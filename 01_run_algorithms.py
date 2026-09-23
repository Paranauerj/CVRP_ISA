"""Comparação ALNS (biblioteca `alns`) vs ILS (biblioteca `pyvrp`) vs GLS (OR-Tools).

Requisitos:
    pip install vrplib numpy ortools alns pyvrp

IMPORTANTE — leia antes de correr em produção:
    A classe `IteratedLocalSearch` descrita em https://pyvrp.org/dev/algorithm.html
    ainda NÃO existe na linha estável instalada (testado com pyvrp==0.11.3,
    Python 3.10 — nessa versão o motor de topo é `GeneticAlgorithm`, estilo HGS).
    Por isso o ILS aqui é construído à mão: um laço de perturbação + aceitação
    escrito em Python (igual ao `ils()` original), mas a busca local em si usa
    o motor compilado do PyVRP (`pyvrp.search.LocalSearch` — 2-opt, Or-opt,
    exchange, etc.), muito mais rico e rápido que o `two_opt` caseiro.

    Os nomes exatos de `pyvrp.search` usados abaixo (NODE_OPERATORS,
    ROUTE_OPERATORS, compute_neighbours, NeighbourhoodParams) foram
    reconstruídos a partir dos notebooks/documentação públicos do PyVRP para
    a linha 0.x — NÃO testados localmente contra a tua instalação exata. Se
    der erro, corre primeiro:

        python -c "import pyvrp.search as s; print([n for n in dir(s) if not n.startswith('_')])"
        python -c "import pyvrp; print([n for n in dir(pyvrp) if not n.startswith('_')])"

    e cola-me o output para eu ajustar.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
import vrplib

# --- ALNS (biblioteca externa: pip install alns) --------------------------
from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxRuntime as AlnsMaxRuntime
import numpy.random as rnd

# --- ILS (biblioteca externa: pip install pyvrp) ---------------------------
import pyvrp


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


def two_opt(route: list[int], distance: np.ndarray) -> list[int]:
    """2-opt intra-rota; usado só para polir rotas dentro do repair do ALNS."""
    best = route[:]
    best_cost = route_cost(best, distance)
    improved = True
    while improved:
        improved = False
        for left in range(len(best) - 1):
            for right in range(left + 1, len(best)):
                candidate = best[:left] + best[left:right + 1][::-1] + best[right + 1:]
                cost = route_cost(candidate, distance)
                if cost + 1e-9 < best_cost:
                    best, best_cost, improved = candidate, cost, True
    return best


# ---------------------------------------------------------------------------
# ALNS — biblioteca `alns` (Ropke & Pisinger, via N-Wouda & Lan, JOSS 2023)
# ---------------------------------------------------------------------------

class CvrpState:
    """Estado do ALNS: uma solução (lista de rotas) + clientes por reinserir.

    A biblioteca `alns` espera que os operadores de destroy/repair tenham a
    assinatura `op(state, rng) -> state` (sem devolver a lista de removidos
    à parte) — por isso os clientes removidos ficam guardados em
    `state.unassigned`, e é isso que o repair operator lê.
    """

    __slots__ = ("routes", "unassigned", "instance")

    def __init__(self, routes: list[list[int]], instance: Instance, unassigned: list[int] | None = None):
        self.routes = routes
        self.instance = instance
        self.unassigned = unassigned if unassigned is not None else []

    def objective(self) -> float:
        return solution_cost(self.routes, self.instance.distance)

    def copy(self) -> "CvrpState":
        return CvrpState([route[:] for route in self.routes], self.instance, list(self.unassigned))


def random_route_removal(state: CvrpState, rng: rnd.Generator, degree: float = 0.15) -> CvrpState:
    """Operador de destruição: remove ~`degree` dos clientes de cada rota."""
    new_state = state.copy()
    removed: list[int] = []
    for route in new_state.routes:
        if not route:
            continue
        count = max(1, int(degree * len(route)))
        for _ in range(min(count, len(route))):
            removed.append(route.pop(int(rng.integers(len(route)))))
    new_state.unassigned = removed
    return new_state


def greedy_repair(state: CvrpState, rng: rnd.Generator) -> CvrpState:
    """Operador de reparação: reinsere `state.unassigned` pelo menor custo, com 2-opt final."""
    new_state = state.copy()
    order = list(new_state.unassigned)
    rng.shuffle(order)
    for customer in order:
        best = None
        for route_index, route in enumerate(new_state.routes):
            if route_load(route, new_state.instance.demand) + new_state.instance.demand[customer] > new_state.instance.capacity:
                continue
            for position in range(len(route) + 1):
                before = 0 if position == 0 else route[position - 1]
                after = 0 if position == len(route) else route[position]
                increase = (new_state.instance.distance[before, customer] + new_state.instance.distance[customer, after]
                            - new_state.instance.distance[before, after])
                candidate = (increase, route_index, position)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            new_state.routes.append([customer])
        else:
            _, route_index, position = best
            new_state.routes[route_index].insert(position, customer)
    new_state.routes = [two_opt(route, new_state.instance.distance) for route in new_state.routes if route]
    new_state.unassigned = []
    return new_state


def run_alns(instance: Instance, seed: int, deadline: float, run_start: float) -> list[tuple[float, float]]:
    """Corre o ALNS até `deadline`; devolve [(segundos_desde_run_start, custo_incumbente), ...]."""
    events: list[tuple[float, float]] = []

    init_routes = pci(instance)
    init_state = CvrpState(init_routes, instance)
    events.append((time.perf_counter() - run_start, init_state.objective()))

    budget = max(0.0, deadline - time.perf_counter())
    if budget <= 0:
        return events

    rng = rnd.default_rng(seed)
    solver = ALNS(rng)
    solver.add_destroy_operator(random_route_removal)
    solver.add_repair_operator(greedy_repair)

    def on_best(feasible_state: CvrpState, rng_: rnd.Generator, **_):
        events.append((time.perf_counter() - run_start, feasible_state.objective()))

    solver.on_best(on_best)

    select = RouletteWheel(scores=[25, 5, 1, 0], decay=0.8, num_destroy=1, num_repair=1)
    # `autofit` espera um número de ITERAÇÕES (não segundos) para calcular o
    # decaimento linear do threshold. Não sabemos ao certo quantas iterações
    # cabem no orçamento (varia com o tamanho da instância), por isso usamos
    # uma estimativa grosseira; ajusta o fator se notares o threshold a
    # decair depressa/devagar demais para o teu tamanho de instância típico.
    estimated_iters = max(100, int(budget * 500))
    accept = RecordToRecordTravel.autofit(init_state.objective(), 0.05, 0.0, estimated_iters)
    stop = AlnsMaxRuntime(budget)

    solver.iterate(init_state, select, accept, stop)
    return events


# ---------------------------------------------------------------------------
# ILS — biblioteca `pyvrp` (motor C++, estilo HGS-CVRP / Vidal 2022)
# ---------------------------------------------------------------------------

def routes_from_solution(solution) -> list[list[int]]:
    """Converte um `pyvrp.Solution` em list[list[int]] (mesmo formato do pci())."""
    return [[client for client in route] for route in solution.routes()]


def run_ils(instance: Instance, seed: int, deadline: float, run_start: float) -> list[tuple[float, float]]:
    """ILS construído à mão sobre o `LocalSearch` compilado do pyvrp (ver aviso
    no topo do ficheiro sobre porquê não usamos `pyvrp.IteratedLocalSearch`).
    Estrutura igual ao `ils()` original: perturbação (relocate) -> busca local
    -> aceitação -> repete até `deadline`; só a busca local mudou de motor.
    """
    from pyvrp import RandomNumberGenerator, CostEvaluator, Solution
    from pyvrp.search import (
        LocalSearch,
        NeighbourhoodParams,
        compute_neighbours,
        NODE_OPERATORS,
        ROUTE_OPERATORS,
    )

    events: list[tuple[float, float]] = []

    init_routes = pci(instance)
    events.append((time.perf_counter() - run_start, solution_cost(init_routes, instance.distance)))

    budget = max(0.0, deadline - time.perf_counter())
    if budget <= 0:
        return events

    data = pyvrp.read(instance.path, round_func="round")
    rng = RandomNumberGenerator(seed=seed)
    neighbours = compute_neighbours(data, NeighbourhoodParams())
    local_search = LocalSearch(data, rng, neighbours)
    for op in NODE_OPERATORS:
        local_search.add_node_operator(op(data))
    for op in ROUTE_OPERATORS:
        local_search.add_route_operator(op(data))

    # Penalidade alta para violação de capacidade — a nossa PCI/perturbação já
    # garante rotas feasible; isto é só rede de segurança para a busca local.
    cost_evaluator = CostEvaluator(load_penalties=[1000], tw_penalty=0, dist_penalty=0)

    current = local_search.search(Solution(data, [r for r in init_routes if r]), cost_evaluator)
    best_distance = current.distance() if current.is_feasible() else float("inf")
    if current.is_feasible():
        events.append((time.perf_counter() - run_start, float(best_distance)))

    perturb_rng = random.Random(seed)
    while time.perf_counter() < deadline:
        routes = routes_from_solution(current)
        nonempty = [i for i, r in enumerate(routes) if r]
        if nonempty:
            source = perturb_rng.choice(nonempty)
            customer = routes[source].pop(perturb_rng.randrange(len(routes[source])))
            target = perturb_rng.randrange(len(routes))
            routes[target].insert(perturb_rng.randrange(len(routes[target]) + 1), customer)

        # `pyvrp.Solution` rejeita rotas vazias na lista (ao contrário do
        # nosso `solution_cost`/`route_cost`, que toleram `[]`) — a
        # perturbação pode esvaziar uma rota (ex.: removeu o único cliente
        # dela), por isso filtramos antes de construir o Solution.
        routes = [r for r in routes if r]
        candidate = local_search.search(Solution(data, routes), cost_evaluator)

        # A busca local trabalha com custo PENALIZADO (permite atravessar
        # soluções temporariamente infeasible para escapar de ótimos locais —
        # técnica standard em HGS/PyVRP). A perturbação acima não valida
        # capacidade antes de inserir no `target`, por isso o candidato pode
        # sair infeasible; a busca local tende a corrigi-lo, mas nem sempre.
        # Para os checkpoints (comparáveis ao BKS) só podemos reportar
        # DISTÂNCIA PURA de soluções feasible — nunca o custo penalizado.
        if candidate.is_feasible() and candidate.distance() < best_distance:
            best_distance = candidate.distance()
            events.append((time.perf_counter() - run_start, float(best_distance)))

        if cost_evaluator.cost(candidate) <= cost_evaluator.cost(current) or perturb_rng.random() < 0.05:
            current = candidate

    return events


# ---------------------------------------------------------------------------
# GLS — OR-Tools (inalterado)
# ---------------------------------------------------------------------------

def run_gls(instance: Instance, seed: int, deadline: float, run_start: float) -> list[tuple[float, float]]:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    events: list[tuple[float, float]] = []
    initial_routes = pci(instance)
    events.append((time.perf_counter() - run_start, solution_cost(initial_routes, instance.distance)))
    num_vehicles = len(initial_routes)

    manager = pywrapcp.RoutingIndexManager(len(instance.distance), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    routing.solver().ReSeed(seed)

    distance_callback = routing.RegisterTransitCallback(
        lambda left, right: int(round(instance.distance[manager.IndexToNode(left), manager.IndexToNode(right)]))
    )
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback)

    demand_callback = routing.RegisterUnaryTransitCallback(
        lambda index: int(round(instance.demand[manager.IndexToNode(index)]))
    )
    routing.AddDimensionWithVehicleCapacity(
        demand_callback, 0, [int(round(instance.capacity))] * num_vehicles, True, "Capacity"
    )

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    remaining_ms = max(1, int((deadline - time.perf_counter()) * 1000))
    parameters.time_limit.FromMilliseconds(remaining_ms)
    routing.CloseModelWithParameters(parameters)

    best_cost = {"value": float("inf")}

    def on_solution() -> None:
        cost = routing.CostVar().Value()
        if cost < best_cost["value"]:
            best_cost["value"] = float(cost)
            events.append((time.perf_counter() - run_start, float(cost)))

    routing.AddAtSolutionCallback(on_solution)

    initial_assignment = routing.ReadAssignmentFromRoutes(initial_routes, True)
    assignment = routing.SolveFromAssignmentWithParameters(initial_assignment, parameters)

    if assignment is None:
        events.append((time.perf_counter() - run_start, float("inf")))
        return events

    routes = []
    for vehicle in range(num_vehicles):
        index = routing.Start(vehicle)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node:
                route.append(node)
            index = assignment.Value(routing.NextVar(index))
        if route:
            routes.append(route)

    events.append((time.perf_counter() - run_start, solution_cost(routes, instance.distance)))
    return events


# ---------------------------------------------------------------------------
# Orquestração / checkpoints / CSV
# ---------------------------------------------------------------------------

RUNNERS = {"ALNS": run_alns, "ILS": run_ils, "GLS": run_gls}


def run_one(instance: Instance, algorithm: str, seed: int, seconds_per_customer: float, cap: float) -> list[dict]:
    budget = min(seconds_per_customer * (len(instance.demand) - 1), cap)
    start = time.perf_counter()
    deadline = start + budget
    checkpoints = (0.25, 0.50, 0.75, 1.00)

    events = RUNNERS[algorithm](instance, seed, deadline, start)
    best_final = min((c for _, c in events), default=float("inf"))

    rows = []
    for fraction in checkpoints:
        target = budget * fraction
        values = [cost for elapsed, cost in events if elapsed <= target]
        rows.append({"instancia_id": instance.instance_id, "algoritmo": algorithm, "seed": seed,
                     "checkpoint": int(fraction * 100), "elapsed_target_seconds": target,
                     "cost": min(values) if values else best_final})
    return rows


def run_job(job: tuple[str, str, int, float, float]) -> list[dict]:
    path_text, algorithm, seed, seconds_per_customer, cap = job
    instance = load_instance(Path(path_text))
    return run_one(instance, algorithm, seed, seconds_per_customer, cap)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/checkpoint_costs_raw.csv"))
    parser.add_argument("--median-output", type=Path, default=Path("outputs/checkpoint_costs_median.csv"))
    parser.add_argument("--algorithms", nargs="+", choices=("ALNS", "ILS", "GLS"), default=("ALNS", "ILS", "GLS"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(1001, 2001, 3001))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seconds-per-customer", type=float, default=0.5)
    parser.add_argument("--time-cap", type=float, default=100.0)
    parser.add_argument("--workers", type=int, default=1,
                         help="número de processos worker (default: 1; recomendado: 8)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers deve ser pelo menos 1")

    paths = sorted(args.instances_dir.glob("*.vrp"))[:args.limit]
    jobs = [(str(path), algorithm, seed, args.seconds_per_customer, args.time_cap)
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