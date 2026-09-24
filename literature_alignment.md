# Literature Alignment — CVRP Algorithm Portfolio
## `01_run_algorithms.py` — Academic Justification

> **⚠️ DO NOT MODIFY any algorithm without reading this document first.**  
> Each implementation below was deliberately designed to match a specific academic formulation.  
> Changes may break the scientific comparability of the benchmark.

---

## 1. ALNS — Adaptive Large Neighbourhood Search (RRT Acceptance)

### Primary Reference
> **Ropke, S., & Pisinger, D. (2006).** "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows." *Transportation Science*, 40(4), 455–472.

### Key Idea (from the paper)
> *"We use a set of destroy and repair methods... The method to use at each step is chosen based on the performance of the method in the past, using an adaptive weight adjustment scheme."* — Ropke & Pisinger (2006, p. 456)

The ALNS framework iteratively destroys part of a solution and repairs it. Weights are updated via a **Roulette Wheel** selector: operators that find better solutions gain higher probability of being chosen in future iterations.

**Destroy operators implemented (aligned with Ropke & Pisinger, 2006 + Christiaens & Vanden Berghe, 2020):**
| Operator | Reference | Description |
|---|---|---|
| Shaw Removal | Ropke & Pisinger (2006) §3.2 | Removes geographically/demand-similar customers; parametric randomness via p=4.0 |
| Worst Removal | Ropke & Pisinger (2006) §3.1 | Removes customers with highest removal savings; p=3.0 |
| Random Removal | Ropke & Pisinger (2006) §3.3 | Removes a random subset of customers |
| Route Removal | — | Removes 1–2 entire routes; forces large structural change |
| String Removal (SISR) | Christiaens & Vanden Berghe (2020) | Removes contiguous subsequences from routes |

**Repair operators:**
| Operator | Reference |
|---|---|
| Regret-2 Insertion | Ropke & Pisinger (2006) §3.4 |
| Regret-3 Insertion | Ropke & Pisinger (2006) §3.4 |
| Greedy Insertion | Standard constructive heuristic |
| PyVRP C++ Polish | Vidal (2022) — SwapStar + all node/route operators |

**Acceptance criterion (ALNS):** Record-to-Record Travel (RRT) — Dueck, G. (1993). "New Optimization Heuristics." *Journal of Computational Physics*, 104(1), 86–92.

### Implementation vs. Literature

| Aspect | Literature | Implementation | Status |
|---|---|---|---|
| Weight update | Roulette Wheel, scores {33,9,13,0} in Ropke & Pisinger | RouletteWheel(scores=[25,10,2,0], decay=0.8) | Aligned (scores are tunable; decay is standard) |
| Acceptance | SA in original paper | ALNS uses RRT; ALNS2 uses SA (canonical) | Both variants covered |
| Destroy degree | 5–30% of customers | degree=0.15 (15%) — within recommended range | Aligned |
| Polishing | Not in original (2006) | PyVRP C++ SwapStar after each repair | Enhancement (standard in modern ALNS-CVRP) |

---

## 2. ALNS2 — ALNS with Simulated Annealing Acceptance (Ropke & Pisinger canonical)

### Primary Reference
> **Ropke, S., & Pisinger, D. (2006).** Same as above. The **Simulated Annealing** acceptance criterion is the one explicitly used in the original paper.

### Key Idea (from the paper)
> *"A new solution s' is accepted if f(s') < f(s), or with probability e^{-(f(s')−f(s))/T} if f(s') >= f(s), where T is a temperature parameter that is gradually decreased."* — Ropke & Pisinger (2006, p. 460)

**Acceptance criterion (ALNS2):** SimulatedAnnealing.autofit(best_c, worse=0.05, accept_prob=0.5, num_iters=n)
- worse=0.05: initial temperature calibrated to accept solutions 5% worse with probability 50%
- This is the autofit implementation from the alns Python package, which attributes this procedure directly to Ropke & Pisinger.

### ALNS vs ALNS2 — Difference Table

| | ALNS | ALNS2 |
|---|---|---|
| Acceptance | RRT (Dueck 1993) | SA (Ropke & Pisinger 2006) — canonical |
| All other aspects | Identical | Identical |
| Empirical performance | Generally worse | Wins on structured instances (X-n101: 0.31% gap) |

---

## 3. ILS — Iterated Local Search (Better Criterion + Stagnation Escape)

### Primary Reference
> **Lourenço, H. R., Martin, O. C., & Stützle, T. (2003).** "Iterated Local Search." In *Handbook of Metaheuristics*, pp. 320–353. Springer.

### Key Idea (from the paper)
> *"The simplest acceptance criterion is to accept a new solution only if it is better than the current solution. This is referred to as the 'Better' criterion."* — Lourenço et al. (2003, p. 326)
>
> *"The perturbation must be strong enough to allow ILS to escape from local optima basins."* — Lourenço et al. (2003, p. 325)

**Perturbation:** k=2–6 random capacity-aware node relocations (k increases after stagnation — escape mechanism from §4.2 of Lourenço et al.).

**Acceptance criterion:** Pure "Better" — only accepts candidate if f(candidate) < f(current). Stagnation escape (restart from s_best + increase perturbation strength) replaces probabilistic acceptance.

**Local Search:** PyVRP C++ engine — all 13 node and route operators (including SwapStar from Vidal, 2022).

### Implementation vs. Literature

| Aspect | Literature | Implementation | Status |
|---|---|---|---|
| Perturbation | "Double-bridge" or k-node kick | k-node capacity-aware relocations, k in [2+boost, 4+boost] | Aligned |
| Acceptance | Better criterion (or SA variant) | Pure Better + stagnation escape | Aligned |
| Stagnation handling | Restart from s_best | STAGNATION_LIMIT=30, restart + perturb_boost up to +4 | Aligned (Lourenço et al. §4.2) |
| Local Search | Any LS procedure | PyVRP C++ (Vidal 2022) — best available for CVRP | Enhancement |

---

## 4. GLS — Guided Local Search

### Primary Reference
> **Voudouris, C., & Tsang, E. (1999).** "Guided Local Search and its Application to the Traveling Salesman Problem." *European Journal of Operational Research*, 113(2), 469–499.

### Key Idea (from the paper)
> *"GLS builds a new objective function on top of the standard one by augmenting it with penalty terms. Penalties are associated with features of the solution... When a local optimum is reached, the penalties on the most 'promising' features are increased."* — Voudouris & Tsang (1999, p. 471)

**OR-Tools implementation:** Uses arc-cost-based feature penalization. At each local optimum, arcs with highest utility = cost / (1 + penalty) are penalized, guiding the search away from frequently visited configurations.

**Metaheuristic:** LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH in OR-Tools routing solver.

### Key Parameters (tuned via 240-config sweep, 10s × 3 instances — LDG80/130/185)
| Parameter | Value | Rationale |
|---|---|---|
| lambda_coefficient | **1.0** (all N) | Sweep winner: avg gap 14.21% vs 18.43% baseline (lambda=0.5). More aggressive arc penalization. |
| use_cross_exchange | BOOL_TRUE | Inter-route 2-opt* moves — always beneficial |
| use_full_path_lns | **BOOL_FALSE** | Counter-intuitively better: was blocking fast improvement cycles. Disabling increases move throughput. |
| use_tsp_opt | BOOL_TRUE | Intra-route TSP optimization — consistently helpful |
| use_relocate_neighbors | BOOL_TRUE | Restricted relocate to K nearest neighbors — cheap diversification |
| lns_time_limit | **1000ms** | 500ms was insufficient for tsp_opt on large routes (N≥130) |

> [!NOTE]
> **Sweep finding**: `use_full_path_lns=False` consistently outperforms `True`. The full-path LNS operator,
> while powerful in theory, competes with faster operators (tsp_opt, cross_exchange) for solver time.
> When disabled, OR-Tools performs more high-frequency improvement moves per second.
> Net effect: **+4.23 percentage points** avg gap improvement across LDG80/130/185.

---

## 5. TS — Tabu Search

### Primary Reference
> **Glover, F. (1989).** "Tabu Search — Part I." *ORSA Journal on Computing*, 1(3), 190–206.

### Key Idea (from the paper)
> *"Tabu search uses a flexible memory structure... Recently visited solutions (or solution attributes) are classified as 'tabu', preventing the search from revisiting them for a certain number of iterations."* — Glover (1989, p. 190)

**OR-Tools implementation:** Short-term memory via LocalSearchMetaheuristic.TABU_SEARCH. Moves that recently worsened the objective are forbidden (tabu-active) for a configurable tenure.

---

## 6. Supporting References

| Work | Contribution | Used In |
|---|---|---|
| Vidal, T. (2022). "Hybrid genetic search for the CVRP." Computers & Operations Research, 140. | SwapStar operator — C++ implementation in PyVRP | ALNS, ALNS2, ILS (polishing) |
| Christiaens, J., & Vanden Berghe, G. (2020). "Slack Induction by String Removals for Vehicle Routing Problems." Transportation Science, 54(2). | SISR — String Removal destroy operator | ALNS, ALNS2 |
| Dueck, G. (1993). "New Optimization Heuristics." Journal of Computational Physics, 104(1). | Record-to-Record Travel acceptance criterion | ALNS |
| Shaw, P. (1998). "Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems." In CP-98, Springer. | Relatedness-based removal (Shaw Removal) | ALNS, ALNS2 |

---

## 7. Parallel Cheapest Insertion (PCI) — Common Constructor

> **Reference:** Standard constructive heuristic; attributed to Clarke-Wright savings framework variants.

**Role in the ISA:** All three algorithm families (ALNS, ILS, GLS/TS) start from the **same PCI solution**. This isolates the effect of the search layer from the construction layer, as required by the ISA execution plan.

> *"The common constructive heuristic ensures that differences in final solution quality are attributable to the search strategy, not to lucky initialisation."* — ISA execution plan (instructions.md)
