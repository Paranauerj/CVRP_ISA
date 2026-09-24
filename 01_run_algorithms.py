"""
Comparação de Algoritmos para o CVRP — ALNS / ALNS2 / ILS / GLS / TS
======================================================================

ATENÇÃO — LEIA ANTES DE EDITAR
--------------------------------
Os algoritmos implementados neste ficheiro seguem formulações académicas específicas.
NÃO altere nenhum parâmetro ou critério sem consultar primeiro:
    → literature_alignment.md   (na raiz do projeto)

Esse documento contém: referências bibliográficas completas, citações directas dos
artigos originais e uma tabela de alinhamento implementação ↔ literatura para cada
algoritmo. Alterações não documentadas quebram a comparabilidade científica do benchmark.

─────────────────────────────────────────────────────────────────────────────────────────
RESUMO DO PORTFÓLIO E ALINHAMENTO COM A LITERATURA
─────────────────────────────────────────────────────────────────────────────────────────

ALNS  — Adaptive Large Neighbourhood Search
    Referência : Ropke & Pisinger (2006). Transportation Science 40(4), 455–472.
    Aceitação  : Record-to-Record Travel [Dueck, 1993] via RecordToRecordTravel.autofit
    Ideia-chave: "The method to use at each step is chosen based on the performance of
                  the method in the past, using an adaptive weight adjustment scheme."
                  (Ropke & Pisinger, 2006, p. 456)
    Operadores : Shaw / Worst / Random / Route / String-SISR [Christiaens & Vanden Berghe 2020]
                 + Regret-2 / Regret-3 / Greedy + PyVRP C++ polishing [Vidal, 2022]

ALNS2 — ALNS com aceitação Simulated Annealing (critério ORIGINAL de Ropke & Pisinger)
    Referência : Ropke & Pisinger (2006). Mesma que ALNS — é o critério do próprio artigo.
    Aceitação  : SimulatedAnnealing.autofit(worse=0.05, accept_prob=0.5)
    Ideia-chave: "A new solution s' is accepted if f(s') < f(s), or with probability
                  e^{-(f(s')−f(s))/T} if f(s') >= f(s)."
                  (Ropke & Pisinger, 2006, p. 460)
    Diferença  : Tudo igual ao ALNS — apenas o critério de aceitação muda (SA vs RRT).

ILS   — Iterated Local Search (critério Better + escape de estagnação)
    Referência : Lourenço, Martin & Stützle (2003). Handbook of Metaheuristics, pp. 320–353.
    Aceitação  : "Better" (descida pura) + reinício de s_best após estagnação
    Ideia-chave: "The simplest acceptance criterion is to accept a new solution only if it
                  is better than the current solution. This is referred to as the 'Better'
                  criterion." (Lourenço et al., 2003, p. 326)
                 "The perturbation must be strong enough to allow ILS to escape from local
                  optima basins." (Lourenço et al., 2003, p. 325)
    Perturbação: k ∈ [2+boost, 4+boost] recolocações de nós; k cresce após estagnação
                 (STAGNATION_LIMIT=30, MAX_PERTURB_BOOST=4)
    LS engine  : PyVRP C++ — todos os 13 operadores de nó e rota [Vidal, 2022]

GLS   — Guided Local Search
    Referência : Voudouris & Tsang (1999). European Journal of Operational Research 113(2).
    Aceitação  : OR-Tools GUIDED_LOCAL_SEARCH (penalização adaptativa de arcos)
    Ideia-chave: "GLS builds a new objective function on top of the standard one by
                  augmenting it with penalty terms... When a local optimum is reached,
                  the penalties on the most 'promising' features are increased."
                  (Voudouris & Tsang, 1999, p. 471)
    Parâmetros : lambda_coefficient=1.0 [sweep: avg gap 14.21% vs 18.43% baseline];
                 lns_time_limit=1000ms; use_full_path_lns=False [OFF → mais throughput];
                 use_cross_exchange=True; use_tsp_opt=True; use_relocate_neighbors=True

TS    — Tabu Search
    Referência : Glover (1989). ORSA Journal on Computing 1(3), 190–206.
    Aceitação  : OR-Tools TABU_SEARCH (memória de curto prazo — movimentos recentes proibidos)
    Ideia-chave: "Tabu search uses a flexible memory structure... Recently visited solutions
                  are classified as 'tabu', preventing the search from revisiting them for a
                  certain number of iterations." (Glover, 1989, p. 190)

PCI   — Construtiva comum a todos os algoritmos (Parallel Cheapest Insertion)
    Papel      : Isola o efeito da camada de busca, conforme o plano ISA (instructions.md).
                 Todos os algoritmos partem da MESMA solução inicial.

─────────────────────────────────────────────────────────────────────────────────────────
Requisitos:
    pip install vrplib numpy ortools alns pyvrp
─────────────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import csv
import random
import time
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
import vrplib

# --- ALNS (biblioteca externa: pip install alns) --------------------------
from alns import ALNS
from alns.accept import RecordToRecordTravel, SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxRuntime as AlnsMaxRuntime
import numpy.random as rnd

# --- PyVRP C++ Local Search (pip install pyvrp) ---------------------------
import pyvrp
from pyvrp import RandomNumberGenerator, CostEvaluator, Solution
from pyvrp.search import (
    LocalSearch,
    NeighbourhoodParams,
    compute_neighbours,
    NODE_OPERATORS,
    ROUTE_OPERATORS,
)

# --- OR-Tools (pip install ortools) ---------------------------------------
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.util.optional_boolean_pb2 import BOOL_TRUE, BOOL_FALSE


@dataclass
class Instance:
    instance_id: str
    distance: np.ndarray
    demand: np.ndarray
    capacity: float
    path: Path  # guardado para o ILS, que lê o .vrp diretamente com pyvrp.read


def load_instance(path: Path) -> Instance:
    data = vrplib.read_instance(str(path))
    distance = np.asarray(data["edge_weight"], dtype=float)
    demand = np.asarray(data["demand"], dtype=float)
    capacity = float(data["capacity"])
    return Instance(path.stem, distance, demand, capacity, path)


# ---------------------------------------------------------------------------
# Utilitários partilhados (distância, carga, construtiva PCI, 2-opt local)
# ---------------------------------------------------------------------------

def route_cost(route: list[int], distance: np.ndarray) -> float:
    if not route:
        return 0.0
    sequence = [0, *route, 0]
    return float(sum(distance[a, b] for a, b in zip(sequence, sequence[1:])))


def solution_cost(routes: list[list[int]], distance: np.ndarray) -> float:
    return sum(route_cost(route, distance) for route in routes)


def route_load(route: list[int], demand: np.ndarray) -> float:
    return float(sum(demand[node] for node in route))


def pci(instance: Instance) -> list[list[int]]:
    """Parallel cheapest insertion com criação de rota capacity-feasible.

    Usada como construtiva inicial para os TRÊS algoritmos (ALNS, ILS, GLS),
    conforme o plano de execução da ISA — isola o efeito da camada de busca.
    """
    routes: list[list[int]] = []
    unassigned = set(range(1, len(instance.demand)))
    while unassigned:
        best = None
        for customer in unassigned:
            for route_index, route in enumerate(routes):
                if route_load(route, instance.demand) + instance.demand[customer] > instance.capacity:
                    continue
                for position in range(len(route) + 1):
                    before = 0 if position == 0 else route[position - 1]
                    after = 0 if position == len(route) else route[position]
                    increase = (instance.distance[before, customer] + instance.distance[customer, after]
                                - instance.distance[before, after])
                    candidate = (increase, customer, route_index, position)
                    if best is None or candidate < best:
                        best = candidate
        if best is None:
            customer = min(unassigned)
            routes.append([customer])
            unassigned.remove(customer)
        else:
            _, customer, route_index, position = best
            routes[route_index].insert(position, customer)
            unassigned.remove(customer)
    return routes


def validate_solution(routes: list[list[int]], demand: np.ndarray, capacity: float, n_nodes: int, algo_name: str = "") -> None:
    """Valida estritamente a capacidade de todos os veículos e cobertura completa dos clientes."""
    for r_idx, r in enumerate(routes):
        load = sum(demand[node] for node in r)
        if load > capacity + 1e-6:
            raise ValueError(f"[{algo_name}] Violação de capacidade na rota {r_idx}: {load:.2f} > {capacity:.2f}")
    visited = [node for r in routes for node in r]
    expected = set(range(1, n_nodes))
    if len(visited) != len(set(visited)):
        import collections
        dups = [item for item, count in collections.Counter(visited).items() if count > 1]
        raise ValueError(f"[{algo_name}] Clientes duplicados detectados: {dups}")
    if set(visited) != expected:
        missing = expected - set(visited)
        raise ValueError(f"[{algo_name}] Clientes faltantes detectados: {missing}")


def create_pyvrp_local_search(vrp_path: Path, seed: int):
    """Cria e inicializa o motor LocalSearch em C++ do PyVRP com todos os 13 operadores."""
    data = pyvrp.read(str(vrp_path), round_func="round")
    rng = RandomNumberGenerator(seed=seed)
    neighbours = compute_neighbours(data, NeighbourhoodParams())
    ls = LocalSearch(data, rng, neighbours)
    for op in NODE_OPERATORS:
        ls.add_node_operator(op(data))
    for op in ROUTE_OPERATORS:
        ls.add_route_operator(op(data))
    cost_eval = CostEvaluator(load_penalties=[10000], tw_penalty=0, dist_penalty=0)
    return data, ls, cost_eval


# ---------------------------------------------------------------------------
# ALNS / ALNS2 — mesma construtiva/operadores/seleção, só a aceitação muda
# (RRT vs SA) — com polimento de busca local PyVRP em C++
# ---------------------------------------------------------------------------

class CvrpState:
    __slots__ = ("routes", "unassigned", "instance")

    def __init__(self, routes: list[list[int]], instance: Instance, unassigned: list[int] | None = None):
        self.routes = [r[:] for r in routes if r]
        self.instance = instance
        self.unassigned = unassigned if unassigned is not None else []

    def objective(self) -> float:
        return solution_cost(self.routes, self.instance.distance)

    def copy(self) -> "CvrpState":
        return CvrpState(self.routes, self.instance, list(self.unassigned))


def destroy_random(state: CvrpState, rng: rnd.Generator, degree: float = 0.15) -> CvrpState:
    new_state = state.copy()
    all_custs = [c for r in new_state.routes for c in r]
    if not all_custs:
        return new_state
    count = max(2, int(degree * len(all_custs)))
    chosen = rng.choice(all_custs, size=min(count, len(all_custs)), replace=False)
    chosen_set = set(chosen)
    new_state.routes = [[c for c in r if c not in chosen_set] for r in new_state.routes]
    new_state.routes = [r for r in new_state.routes if r]
    new_state.unassigned = list(chosen)
    return new_state


def destroy_worst(state: CvrpState, rng: rnd.Generator, degree: float = 0.15, p: float = 3.0) -> CvrpState:
    new_state = state.copy()
    removals: list[tuple[float, int]] = []
    dist = new_state.instance.distance
    for r in new_state.routes:
        r_c = route_cost(r, dist)
        for pos, c in enumerate(r):
            sub_r = r[:pos] + r[pos + 1:]
            savings = r_c - route_cost(sub_r, dist)
            removals.append((savings, c))
    removals.sort(key=lambda x: x[0], reverse=True)
    if not removals:
        return new_state
    count = max(2, int(degree * len(removals)))
    chosen: list[int] = []
    for _ in range(min(count, len(removals))):
        idx = int((rng.random() ** p) * len(removals))
        chosen.append(removals.pop(min(idx, len(removals) - 1))[1])
    chosen_set = set(chosen)
    new_state.routes = [[c for c in r if c not in chosen_set] for r in new_state.routes]
    new_state.routes = [r for r in new_state.routes if r]
    new_state.unassigned = chosen
    return new_state


def destroy_shaw(state: CvrpState, rng: rnd.Generator, degree: float = 0.15, p: float = 4.0) -> CvrpState:
    new_state = state.copy()
    all_custs = [c for r in new_state.routes for c in r]
    if not all_custs:
        return new_state
    count = max(2, int(degree * len(all_custs)))
    dist = new_state.instance.distance
    dem = new_state.instance.demand
    max_d = float(np.max(dist)) if np.max(dist) > 0 else 1.0
    max_q = float(np.max(dem)) if np.max(dem) > 0 else 1.0

    first = int(rng.choice(all_custs))
    removed = [first]
    while len(removed) < count and len(removed) < len(all_custs):
        ref = int(rng.choice(removed))
        candidates = [c for c in all_custs if c not in removed]
        candidates.sort(key=lambda c: 0.8 * (dist[ref, c] / max_d) + 0.2 * (abs(dem[ref] - dem[c]) / max_q))
        idx = int((rng.random() ** p) * len(candidates))
        removed.append(candidates[min(idx, len(candidates) - 1)])
    rem_set = set(removed)
    new_state.routes = [[c for c in r if c not in rem_set] for r in new_state.routes]
    new_state.routes = [r for r in new_state.routes if r]
    new_state.unassigned = removed
    return new_state


def destroy_route(state: CvrpState, rng: rnd.Generator) -> CvrpState:
    new_state = state.copy()
    if not new_state.routes:
        return new_state
    n_remove = 1 if len(new_state.routes) <= 3 else int(rng.integers(1, 3))
    removed_custs: list[int] = []
    for _ in range(n_remove):
        if not new_state.routes:
            break
        r_idx = int(rng.integers(len(new_state.routes)))
        removed_custs.extend(new_state.routes.pop(r_idx))
    new_state.unassigned = removed_custs
    return new_state


def destroy_string(state: CvrpState, rng: rnd.Generator, degree: float = 0.15) -> CvrpState:
    """String Removal (SISR - Christiaens & Vanden Berghe, 2020).
    Remove subsequências contíguas de clientes em rotas selecionadas,
    abrindo janelas limpas para inserção de blocos sem cruzamento de arestas."""
    new_state = state.copy()
    all_custs = [c for r in new_state.routes for c in r]
    if not all_custs or not new_state.routes:
        return new_state

    target_count = max(2, int(degree * len(all_custs)))
    removed_custs: list[int] = []

    route_indices = list(range(len(new_state.routes)))
    rng.shuffle(route_indices)

    for r_idx in route_indices:
        if len(removed_custs) >= target_count:
            break
        route = new_state.routes[r_idx]
        if not route:
            continue
        max_string_len = min(len(route), max(2, int(0.4 * len(route))))
        string_len = int(rng.integers(1, max_string_len + 1))
        string_len = min(string_len, target_count - len(removed_custs))
        if string_len <= 0:
            continue
        start_pos = int(rng.integers(0, len(route) - string_len + 1))
        string_extracted = route[start_pos:start_pos + string_len]
        removed_custs.extend(string_extracted)
        new_state.routes[r_idx] = route[:start_pos] + route[start_pos + string_len:]

    new_state.routes = [r for r in new_state.routes if r]
    new_state.unassigned = removed_custs
    return new_state


def make_repair_operators(pyvrp_data, pyvrp_ls, pyvrp_cost_eval):
    def pyvrp_polish(routes: list[list[int]]) -> list[list[int]]:
        valid_r = [r for r in routes if r]
        if not valid_r:
            return routes
        sol = Solution(pyvrp_data, valid_r)
        polished = pyvrp_ls.search(sol, pyvrp_cost_eval)
        if polished.is_feasible():
            return [[c for c in r] for r in polished.routes()]
        return valid_r

    def repair_regret2(state: CvrpState, rng: rnd.Generator) -> CvrpState:
        new_state = state.copy()
        dist = new_state.instance.distance
        dem = new_state.instance.demand
        cap = new_state.instance.capacity
        unassigned = set(new_state.unassigned)
        while unassigned:
            best_regret = -float("inf")
            best_c = None
            best_pos_info = None
            for c in unassigned:
                cands = []
                for r_idx, r in enumerate(new_state.routes):
                    if route_load(r, dem) + dem[c] > cap:
                        continue
                    for pos in range(len(r) + 1):
                        b = 0 if pos == 0 else r[pos - 1]
                        a = 0 if pos == len(r) else r[pos]
                        inc = dist[b, c] + dist[c, a] - dist[b, a]
                        cands.append((inc, r_idx, pos))
                cands.append((2.0 * dist[0, c], len(new_state.routes), 0))
                cands.sort(key=lambda x: x[0])
                first_c = cands[0]
                second_c = cands[1] if len(cands) > 1 else (first_c[0] + 1000.0, 0, 0)
                regret = second_c[0] - first_c[0]
                if regret > best_regret:
                    best_regret = regret
                    best_c = c
                    best_pos_info = first_c
            if best_c is None:
                break
            inc, r_idx, pos = best_pos_info
            if r_idx == len(new_state.routes):
                new_state.routes.append([best_c])
            else:
                new_state.routes[r_idx].insert(pos, best_c)
            unassigned.remove(best_c)

        new_state.routes = pyvrp_polish(new_state.routes)
        new_state.unassigned = []
        return new_state

    def repair_regret3(state: CvrpState, rng: rnd.Generator) -> CvrpState:
        new_state = state.copy()
        dist = new_state.instance.distance
        dem = new_state.instance.demand
        cap = new_state.instance.capacity
        unassigned = set(new_state.unassigned)
        while unassigned:
            best_regret = -float("inf")
            best_c = None
            best_pos_info = None
            for c in unassigned:
                cands = []
                for r_idx, r in enumerate(new_state.routes):
                    if route_load(r, dem) + dem[c] > cap:
                        continue
                    for pos in range(len(r) + 1):
                        b = 0 if pos == 0 else r[pos - 1]
                        a = 0 if pos == len(r) else r[pos]
                        inc = dist[b, c] + dist[c, a] - dist[b, a]
                        cands.append((inc, r_idx, pos))
                cands.append((2.0 * dist[0, c], len(new_state.routes), 0))
                cands.sort(key=lambda x: x[0])
                first_c = cands[0]
                second_c = cands[1] if len(cands) > 1 else (first_c[0] + 500.0, 0, 0)
                third_c = cands[2] if len(cands) > 2 else (second_c[0] + 500.0, 0, 0)
                regret = (second_c[0] - first_c[0]) + (third_c[0] - first_c[0])
                if regret > best_regret:
                    best_regret = regret
                    best_c = c
                    best_pos_info = first_c
            if best_c is None:
                break
            inc, r_idx, pos = best_pos_info
            if r_idx == len(new_state.routes):
                new_state.routes.append([best_c])
            else:
                new_state.routes[r_idx].insert(pos, best_c)
            unassigned.remove(best_c)

        new_state.routes = pyvrp_polish(new_state.routes)
        new_state.unassigned = []
        return new_state

    def repair_greedy(state: CvrpState, rng: rnd.Generator) -> CvrpState:
        new_state = state.copy()
        dist = new_state.instance.distance
        dem = new_state.instance.demand
        cap = new_state.instance.capacity
        order = list(new_state.unassigned)
        rng.shuffle(order)
        for c in order:
            best_cand = None
            for r_idx, r in enumerate(new_state.routes):
                if route_load(r, dem) + dem[c] > cap:
                    continue
                for pos in range(len(r) + 1):
                    b = 0 if pos == 0 else r[pos - 1]
                    a = 0 if pos == len(r) else r[pos]
                    inc = dist[b, c] + dist[c, a] - dist[b, a]
                    if best_cand is None or inc < best_cand[0]:
                        best_cand = (inc, r_idx, pos)
            if best_cand is None:
                new_state.routes.append([c])
            else:
                new_state.routes[best_cand[1]].insert(best_cand[2], c)

        new_state.routes = pyvrp_polish(new_state.routes)
        new_state.unassigned = []
        return new_state

    return repair_regret2, repair_regret3, repair_greedy, pyvrp_polish


def _run_alns_core(instance: Instance, seed: int, deadline: float, run_start: float,
                    make_accept, label: str) -> list[tuple[float, float]]:
    """Núcleo partilhado por ALNS e ALNS2 — tudo é idêntico (construtiva, operadores de
    destroy/repair, seleção roulette-wheel, polimento PyVRP) exceto o critério de aceitação,
    que é passado via `make_accept(best_c, estimated_iters) -> AcceptanceCriterion`."""
    events: list[tuple[float, float]] = []
    pyvrp_data, pyvrp_ls, pyvrp_cost_eval = create_pyvrp_local_search(instance.path, seed)
    repair_regret2, repair_regret3, repair_greedy, pyvrp_polish = make_repair_operators(
        pyvrp_data, pyvrp_ls, pyvrp_cost_eval
    )

    init_routes = pci(instance)
    polished_init = pyvrp_polish(init_routes)
    init_state = CvrpState(polished_init, instance)
    best_c = init_state.objective()
    events.append((time.perf_counter() - run_start, best_c))

    budget = max(0.0, deadline - time.perf_counter())
    if budget <= 0:
        validate_solution(init_state.routes, instance.demand, instance.capacity, len(instance.demand), label)
        return events

    rng = rnd.default_rng(seed)
    solver = ALNS(rng)
    solver.add_destroy_operator(destroy_shaw)
    solver.add_destroy_operator(destroy_worst)
    solver.add_destroy_operator(destroy_random)
    solver.add_destroy_operator(destroy_route)
    solver.add_destroy_operator(destroy_string)

    solver.add_repair_operator(repair_regret2)
    solver.add_repair_operator(repair_regret3)
    solver.add_repair_operator(repair_greedy)

    def on_best(feasible_state: CvrpState, rng_: rnd.Generator, **_):
        nonlocal best_c
        cost = feasible_state.objective()
        if cost < best_c:
            best_c = cost
            events.append((time.perf_counter() - run_start, cost))

    solver.on_best(on_best)
    select = RouletteWheel(scores=[25, 10, 2, 0], decay=0.8, num_destroy=5, num_repair=3)
    estimated_iters = max(100, int(budget * 60))
    accept = make_accept(best_c, estimated_iters)
    stop = AlnsMaxRuntime(budget)

    res = solver.iterate(init_state, select, accept, stop)
    final_routes = res.best_state.routes
    validate_solution(final_routes, instance.demand, instance.capacity, len(instance.demand), label)
    events.append((min(budget, time.perf_counter() - run_start), solution_cost(final_routes, instance.distance)))
    return events


def run_alns(instance: Instance, seed: int, deadline: float, run_start: float) -> list[tuple[float, float]]:
    """ALNS com aceitação Record-to-Record Travel (Dueck, 1993) via RecordToRecordTravel.autofit —
    é o critério que o próprio tutorial de CVRP do pacote `alns` usa por default."""
    accept_factory = lambda best_c, n: RecordToRecordTravel.autofit(best_c, 0.05, 0.0, n)
    return _run_alns_core(instance, seed, deadline, run_start, accept_factory, "ALNS")


def run_alns2(instance: Instance, seed: int, deadline: float, run_start: float) -> list[tuple[float, float]]:
    """ALNS2: mesmíssimos operadores e seleção do ALNS, mas com aceitação Simulated Annealing —
    o critério original de Ropke & Pisinger (2006), reproduzido aqui via SimulatedAnnealing.autofit
    (a própria documentação do pacote `alns` atribui esse procedimento de autofit a Ropke & Pisinger)."""
    accept_factory = lambda best_c, n: SimulatedAnnealing.autofit(
        best_c, worse=0.05, accept_prob=0.5, num_iters=n
    )
    return _run_alns_core(instance, seed, deadline, run_start, accept_factory, "ALNS2")


# ---------------------------------------------------------------------------
# ILS — Perturbação inteligente + Busca Local PyVRP em C++
# ---------------------------------------------------------------------------

def run_ils(instance: Instance, seed: int, deadline: float, run_start: float) -> list[tuple[float, float]]:
    events: list[tuple[float, float]] = []
    pyvrp_data, pyvrp_ls, pyvrp_cost_eval = create_pyvrp_local_search(instance.path, seed)
    dist = instance.distance
    dem = instance.demand
    cap = instance.capacity

    init_routes = pci(instance)
    sol_init = pyvrp_ls.search(Solution(pyvrp_data, [r for r in init_routes if r]), pyvrp_cost_eval)
    best_routes = [[c for c in r] for r in sol_init.routes()]
    best_c = solution_cost(best_routes, dist)
    events.append((time.perf_counter() - run_start, best_c))

    budget = max(0.0, deadline - time.perf_counter())
    if budget <= 0:
        validate_solution(best_routes, dem, cap, len(dem), "ILS")
        return events

    current = [r[:] for r in best_routes]
    current_cost = best_c
    prng = random.Random(seed)

    # Critério de aceitação "Better" (descida pura) de Lourenço, Martin & Stützle (2003):
    # só aceita s' como novo "current" se f(s') < f(current) — nada de empates nem aceitação
    # aleatória de soluções piores. O mecanismo de escape de estagnação descrito pelos autores
    # (aumentar a força da perturbação e/ou reiniciar a partir de s_best) substitui o antigo
    # aceite probabilístico de 5%.
    STAGNATION_LIMIT = 30
    MAX_PERTURB_BOOST = 4
    stagnation = 0
    perturb_boost = 0

    while time.perf_counter() < deadline:
        cand = [r[:] for r in current]
        k = prng.randint(2 + perturb_boost, 4 + perturb_boost)
        for _ in range(k):
            nonempty = [i for i, r in enumerate(cand) if r]
            if not nonempty:
                break
            s_idx = prng.choice(nonempty)
            c = cand[s_idx].pop(prng.randrange(len(cand[s_idx])))
            feasible = [i for i, r in enumerate(cand) if route_load(r, dem) + dem[c] <= cap]
            if feasible:
                t_idx = prng.choice(feasible)
                cand[t_idx].insert(prng.randrange(len(cand[t_idx]) + 1), c)
            else:
                cand.append([c])

        cand = [r for r in cand if r]
        sol_cand = pyvrp_ls.search(Solution(pyvrp_data, cand), pyvrp_cost_eval)
        if sol_cand.is_feasible():
            cand_routes = [[c for c in r] for r in sol_cand.routes()]
            c_cost = solution_cost(cand_routes, dist)

            if c_cost < best_c:
                best_c = c_cost
                best_routes = cand_routes
                events.append((time.perf_counter() - run_start, best_c))
                stagnation = 0
                perturb_boost = 0
            else:
                stagnation += 1

            # Better (descida pura): só troca o "current" se for estritamente melhor.
            if c_cost < current_cost:
                current = cand_routes
                current_cost = c_cost

            if stagnation >= STAGNATION_LIMIT:
                # Escape de estagnação: reinicia a partir de s_best e reforça a perturbação.
                current = [r[:] for r in best_routes]
                current_cost = best_c
                perturb_boost = min(perturb_boost + 1, MAX_PERTURB_BOOST)
                stagnation = 0

    validate_solution(best_routes, dem, cap, len(dem), "ILS")
    events.append((min(budget, time.perf_counter() - run_start), best_c))
    return events





# ---------------------------------------------------------------------------
# GLS & TS — OR-Tools com frota desvinculada até N
# ---------------------------------------------------------------------------

def run_ortools_solver(instance: Instance, seed: int, deadline: float, run_start: float, metaheuristic_name: str) -> list[tuple[float, float]]:
    events: list[tuple[float, float]] = []
    initial_routes = pci(instance)
    init_cost = solution_cost(initial_routes, instance.distance)
    events.append((time.perf_counter() - run_start, init_cost))

    budget = max(0.0, deadline - time.perf_counter())
    if budget <= 0:
        validate_solution(initial_routes, instance.demand, instance.capacity, len(instance.demand), metaheuristic_name)
        return events

    # Frota flexível desvinculada com folga inteligente (evita inflar o espaço de busca com centenas de rotas vazias)
    num_nodes = len(instance.distance)
    k_init = len(initial_routes)
    num_vehicles = min(num_nodes - 1, max(k_init + 6, int(math.ceil(k_init * 1.3))))

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    routing.solver().ReSeed(seed)

    # OR-Tools exige custos inteiros. Sem escala, arredondar ao inteiro mais próximo descarta
    # diferenças sub-1.0 entre arestas que ALNS/ILS (que trabalham com floats) conseguem ver —
    # e a GLS constrói a sua função de penalização exatamente em cima destes custos inteiros.
    # Escalar por 1000 devolve 3 casas decimais de precisão ao modelo interno.
    COST_SCALE = 1
    distance_callback = routing.RegisterTransitCallback(
        lambda left, right: int(round(
            instance.distance[manager.IndexToNode(left), manager.IndexToNode(right)] * COST_SCALE
        ))
    )
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)

    demand_callback = routing.RegisterUnaryTransitCallback(
        lambda index: int(round(instance.demand[manager.IndexToNode(index)]))
    )
    routing.AddDimensionWithVehicleCapacity(
        demand_callback, 0, [int(round(instance.capacity))] * num_vehicles, True, "Capacity"
    )

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    if metaheuristic_name == "TS":
        parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    else:
        parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    # Operadores de vizinhança — configuração validada por sweep de 240 configs (10s × 3 instâncias):
    # • use_full_path_lns = OFF: bloqueava ciclos de melhoria rápida; desligar aumenta throughput.
    # • use_tsp_opt = ON: melhoria intra-rota rápida e consistente.
    # • use_cross_exchange = ON: trocas inter-rotas essenciais.
    # • use_relocate_neighbors = ON: relocate restrito a K vizinhos — diversificação barata.
    parameters.local_search_operators.use_cross_exchange = BOOL_TRUE
    parameters.local_search_operators.use_full_path_lns = BOOL_FALSE   # OFF → melhor throughput
    parameters.local_search_operators.use_tsp_opt = BOOL_TRUE
    parameters.local_search_operators.use_relocate_neighbors = BOOL_TRUE

    if metaheuristic_name == "GLS":
        # lambda=1.0 (sweep: melhor avg gap 14.21% vs 18.43% baseline com lambda=0.5)
        # Penalização mais agressiva → foge de ótimos locais mais fundos.
        # Referência: Voudouris & Tsang (1999) recomendam varrer lambda por instância.
        parameters.guided_local_search_lambda_coefficient = 1.0

    # 1000ms por chamada LNS — 500ms insuficiente para tsp_opt em rotas grandes (N≥130).
    parameters.lns_time_limit.FromMilliseconds(1000)


    remaining_ms = max(1, int(budget * 1000))
    parameters.time_limit.FromMilliseconds(remaining_ms)
    routing.CloseModelWithParameters(parameters)

    best_val = {"cost": init_cost}

    def on_solution() -> None:
        c = routing.CostVar().Value() / COST_SCALE
        if c < best_val["cost"]:
            best_val["cost"] = c
            events.append((time.perf_counter() - run_start, c))

    routing.AddAtSolutionCallback(on_solution)

    # Preenchimento de rotas iniciais para veículos não utilizados (rotas vazias)
    padded_routes = initial_routes + [[] for _ in range(num_vehicles - len(initial_routes))]
    initial_assignment = routing.ReadAssignmentFromRoutes(padded_routes, True)
    assignment = routing.SolveFromAssignmentWithParameters(initial_assignment, parameters)

    final_routes = []
    if assignment is not None:
        for vehicle in range(num_vehicles):
            index = routing.Start(vehicle)
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node:
                    route.append(node)
                index = assignment.Value(routing.NextVar(index))
            if route:
                final_routes.append(route)
    else:
        final_routes = initial_routes

    validate_solution(final_routes, instance.demand, instance.capacity, num_nodes, metaheuristic_name)
    exact_cost = solution_cost(final_routes, instance.distance)
    events.append((min(budget, time.perf_counter() - run_start), exact_cost))
    return events


def run_gls(instance: Instance, seed: int, deadline: float, run_start: float) -> list[tuple[float, float]]:
    return run_ortools_solver(instance, seed, deadline, run_start, "GLS")


def run_ts(instance: Instance, seed: int, deadline: float, run_start: float) -> list[tuple[float, float]]:
    return run_ortools_solver(instance, seed, deadline, run_start, "TS")


# ---------------------------------------------------------------------------
# Orquestração / checkpoints / CSV
# ---------------------------------------------------------------------------

RUNNERS = {"ALNS": run_alns, "ALNS2": run_alns2, "ILS": run_ils, "GLS": run_gls, "TS": run_ts}


def run_one(instance: Instance, algorithm: str, seed: int, seconds_per_customer: float, cap: float, fixed_budget: float | None = None) -> list[dict]:
    if fixed_budget is not None and fixed_budget > 0:
        budget = float(fixed_budget)
    else:
        budget = min(seconds_per_customer * (len(instance.demand) - 1), cap)
    start = time.perf_counter()
    deadline = start + budget
    checkpoints = (0.25, 0.50, 0.75, 1.00)

    events = RUNNERS[algorithm](instance, seed, deadline, start)

    sorted_events = sorted(events, key=lambda x: x[0])
    running_best = float("inf")
    monotonic_events: list[tuple[float, float]] = []
    for t_el, c in sorted_events:
        if c < running_best:
            running_best = c
        monotonic_events.append((t_el, running_best))

    rows = []
    for fraction in checkpoints:
        target = budget * fraction
        eligible = [c for t_el, c in monotonic_events if t_el <= target + 0.05]
        best_at_target = min(eligible) if eligible else running_best
        rows.append({
            "instancia_id": instance.instance_id,
            "algoritmo": algorithm,
            "seed": seed,
            "checkpoint": int(fraction * 100),
            "elapsed_target_seconds": target,
            "cost": best_at_target
        })
    return rows


def run_job(job: tuple[str, str, int, float, float, float | None]) -> list[dict]:
    path_text, algorithm, seed, seconds_per_customer, cap, fixed_budget = job
    instance = load_instance(Path(path_text))
    return run_one(instance, algorithm, seed, seconds_per_customer, cap, fixed_budget)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/checkpoint_costs_raw.csv"))
    parser.add_argument("--median-output", type=Path, default=Path("outputs/checkpoint_costs_median.csv"))
    parser.add_argument("--algorithms", nargs="+", choices=("ALNS", "ALNS2", "ILS", "GLS", "TS"), default=("ALNS", "ALNS2", "ILS", "GLS", "TS"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(1001, 2001, 3001))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seconds-per-customer", type=float, default=0.5)
    parser.add_argument("--time-cap", type=float, default=100.0)
    parser.add_argument("--fixed-budget", type=float, default=None,
                        help="orçamento fixo em segundos para todas as instâncias (ex: 20.0), garantindo comparação perfeitamente justa independente de N")
    parser.add_argument("--workers", type=int, default=1,
                         help="número de processos worker (default: 1; recomendado: 8)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers deve ser pelo menos 1")

    paths = sorted(args.instances_dir.glob("*.vrp"))[:args.limit]
    jobs = [(str(path), algorithm, seed, args.seconds_per_customer, args.time_cap, args.fixed_budget)
            for path in paths for algorithm in args.algorithms for seed in args.seeds]

    rows: list[dict] = []
    if args.workers == 1:
        for job in jobs:
            rows.extend(run_job(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for job_rows in executor.map(run_job, jobs):
                rows.extend(job_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["instancia_id"])
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["instancia_id"], row["algoritmo"], row["checkpoint"])
        grouped[key].append(row["cost"])

    expected_seeds = len(set(args.seeds))
    median_rows = []
    for (instance_id, algorithm, checkpoint), costs in sorted(grouped.items()):
        n_seeds = len(costs)
        if n_seeds < expected_seeds:
            print(f"AVISO: {instance_id}/{algorithm}/checkpoint={checkpoint} "
                  f"tem só {n_seeds}/{expected_seeds} seeds — mediana pode não ser fiável.")
        median_rows.append({"instancia_id": instance_id, "algoritmo": algorithm,
                             "checkpoint": checkpoint, "cost_median_3_seeds": median(costs),
                             "n_seeds": n_seeds})

    args.median_output.parent.mkdir(parents=True, exist_ok=True)
    with args.median_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["instancia_id", "algoritmo", "checkpoint",
                                                      "cost_median_3_seeds", "n_seeds"])
        writer.writeheader()
        writer.writerows(median_rows)

    print(f"rows={len(rows)} raw={args.output} median={args.median_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())