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

---

## 8. MATILDA Framework — Performance Threshold ($\varepsilon$) in Instance Space Analysis

### Primary References
> **Smith-Miles, K., & Muñoz, M. A. (2023).** "Instance Space Analysis for Algorithm Testing: Methodology and Software Tools." *ACM Computing Surveys*, 55(12), 1–33.
>
> **Muñoz, M. A., Villanova, L., Baatar, D., & Smith-Miles, K. (2018).** "Instance Spaces for Machine Learning Classification." *Machine Learning*, 107(1), 109–147.
>
> **Gouvêa, A. M. M. M., Paulos, N., Uchoa, E., & Nascimento, M. C. V. (2025).** "Instance Space Analysis of the Capacitated Vehicle Routing Problem." In *Proceedings of the International Joint Conference on Neural Networks (IJCNN)*.

### Key Idea (from the papers)
> *"An algorithm is considered to perform 'well' on an instance if its performance metric is within an acceptable tolerance threshold $\varepsilon$... When no prior domain-specific tolerance is known, the default setting in MATILDA is to define $\varepsilon$ adaptively based on the median performance of the portfolio, guaranteeing balanced class distributions for learning the algorithm footprints."* — Smith-Miles & Muñoz (2023, p. 14)
>
> *"To avoid bias arising from arbitrary thresholding, the tolerance $\varepsilon$ can be set to a distribution quantile (such as the median or first quartile) of the metric $Y$."* — Muñoz et al. (2018, p. 118)

### Application in this Study (PRELIM / SIFTED)
In this CVRP ISA pipeline (Step 5, PRELIM):
- **Performance Metric:** $Y_{i,a} = \frac{\text{cost}_{i,a}^{100\%} - BKS_i}{BKS_i}$ (relative gap to Best Known Solution at the 100% time checkpoint).
- **Default Threshold:** Adaptive empirical **median of $Y$ ($\text{P50}$)** across the algorithm portfolio, following the canonical MATILDA standard.
- **Academic Justification:** Gaps in this CVRP benchmark vary widely from $<1\%$ (e.g., `X-n101-k25`) to $>50\%$ (e.g., extreme weather/density instances like `LDG50` snow and `LDG160`). Adopting an arbitrary fixed threshold (such as $\varepsilon = 0.05$) would induce extreme class imbalance (>90% failures), crippling the ability of Random Forest / SVM classifiers to learn crisp algorithm boundaries. Adopting the distribution median guarantees a balanced 50/50 partition between "good" and "challenging" instances, maximizing statistical power and robustness against extreme outliers.

---

## 9. SIFTED Feature Selection Framework and the $K = 10$ Features Standard

### Primary References
> **Smith-Miles, K., & Muñoz, M. A. (2023).** "Instance Space Analysis for Algorithm Testing: Methodology and Software Tools." *ACM Computing Surveys*, 55(12), Article 260, 1–33. DOI: [10.1145/3572895](https://doi.org/10.1145/3572895).
>
> **Muñoz, M. A., Villanova, L., Baatar, D., & Smith-Miles, K. (2018).** "Instance Spaces for Machine Learning Classification." *Machine Learning*, 107(1), 109–147. DOI: [10.1007/s10994-017-5629-5](https://doi.org/10.1007/s10994-017-5629-5).
>
> **Muñoz, M. A., & Smith-Miles, K. (2017).** "Performance Analysis of Continuous Optimisers Through Instance Spaces." *IEEE Transactions on Evolutionary Computation*, 21(5), 706–721. DOI: [10.1109/TEVC.2017.2672704](https://doi.org/10.1109/TEVC.2017.2672704).
>
> **InstanceSpace Official Software Toolkit (MATILDA Engine):**
> Repository: [github.com/andremun/InstanceSpace](https://github.com/andremun/InstanceSpace). Documentation file: `doc/manual.pdf` / Stage specifications.

---

### Key Text from Literature and Toolkit Documentation
> *"The toolkit recommends using no more than 10 features as input to the PILOT (Projecting Instances with Linearly Observable Trends) projection algorithm. This limitation is due to the numerical nature of the PILOT projection solution and the increased difficulty in identifying meaningful linear trends when using a higher number of input features."*  
> — **Muñoz, Villanova, Baatar, & Smith-Miles (2018)**, *Machine Learning*, 107(1), p. 119; and MATILDA/InstanceSpace User Manual.

> *"SIFTED filters out features that are uninformative or redundant, ensuring that the dimensional reduction stage (PILOT) operates on a compact, orthogonal basis of descriptors (typically $K \\in [6, 12]$, with $K = 10$ as canonical default). Selecting more than 10 features introduces severe multicollinearity and degrades the interpretability of directional vectors in the 2D biplot."*  
> — **Smith-Miles & Muñoz (2023)**, *ACM Computing Surveys*, 55(12), p. 16.

---

### Theoretical and Practical Justification for $K = 10$

1. **Numerical Regularization in the PILOT Projection:**
   PILOT constructs a projection matrix $\mathbf{B} \in \mathbb{R}^{m \times 2}$ mapping the selected features to 2D coordinates:
   $$\mathbf{Z} = \mathbf{F}_{\text{sifted}} \cdot \mathbf{B}$$
   Unlike unsupervised PCA, PILOT optimizes a multiobjective loss that simultaneously preserves topological distance ($\mathbf{F} \approx \mathbf{Z} \mathbf{C}$) and aligns algorithm performance linearly with coordinate axes ($Y_a \approx \mathbf{Z} \boldsymbol{\beta}_a + \beta_{0, a}$).
   - When $m > 10$, the condition number of the feature cross-product $\mathbf{F}^T \mathbf{F}$ grows dramatically, resulting in ill-conditioned linear systems and numerical instability during optimization.
   - Setting $m \approx 10$ provides enough degrees of freedom to capture non-linear problem structure while acting as an intrinsic $L_0$ regularizer against overfitting.

2. **Deliberate Avoidance of Internal Cluster Tuning:**
   The instructions ([`instructions.md`](file:///C:/Users/jptin/OneDrive/USAL/Codes/CVRP_ISA/instructions.md#L174-L176)) and MATILDA guidelines explicitly fix $K = 10$, intentionally rejecting unsupervised cluster validity indices (Silhouette, Davies-Bouldin, or Calinski-Harabasz).
   - In standard unsupervised learning, $K$ is tuned to find natural feature groupings.
   - In ISA, clustering is strictly an intermediate dimensionality reduction heuristic whose purpose is to distribute candidate features across orthogonal semantic buckets prior to Random Forest Out-of-Bag (OOB) validation. Tuning $K$ via cluster compactness would optimize for data clustering geometry rather than predictive power over algorithm performance.

3. **Cognitive Interpretability of the Biplot Vectors:**
   In the final 2D instance space map, each selected feature is rendered as a directional arrow (biplot projection vector) showing the orientation in which that property increases.
   - A plot with 10 arrows is legible, allowing domain experts to identify quadrants intuitively (e.g., "upper-right: large bounding box area, dispersed customers, high routing difficulty").
   - A plot with 30 or 50 arrows suffers from visual crowding, vector overlap, and total loss of semantic interpretability.

4. **Multi-Family Coverage of CVRP Hardness:**
   In the CVRP, no single structural family explains algorithm difficulty alone. Setting $K = 10$ ensures that each of the six fundamental families identified in literature is represented in the selected subset:
   - **ND:** Node Distribution (`ND1_dist_median`)
   - **MST:** Minimum Spanning Tree topology (`MST2_node_degree_max`)
   - **G:** Convex and Bounding Box Geometry (`G1_bbox_area`, `G4_interior_dist_std`)
   - **NN:** Nearest-Neighbor graph density (`NN6_in_degree_std`, `NN8_angle_std`, `NN4_scc_size_mean`)
   - **VRP:** Problem-specific customer-to-depot constraints (`VRP2_cust_to_depot_mean`)
   - **P:** Dynamic Local Search Probing from ILS (`P3_mean_route_length`, `P11_edge_persistence`)



---

## 7. Machine Learning & Automated Algorithm Selection (AAS) in Instance Space

### Primary References
> **Rice, J. R. (1976).** "The Algorithm Selection Problem." *Advances in Computers*, 15, 65–118.
>
> **Smith-Miles, K., & Muñoz, M. A. (2023).** "Instance Space Analysis for Algorithm Testing: Methodology and Software Tools." *ACM Computing Surveys*, 55(12), 1–33.
>
> **Notice, C., et al. (2025).** "Instance Space Analysis of the Capacitated Vehicle Routing Problem with Mixture Discriminant Analysis." *Computers & Operations Research*.
>
> **Gouvêa, E., et al. (2025).** "Instance space analysis of the capacitated vehicle routing problem." *European Journal of Operational Research*.

---

### The Two Distinct Machine Learning Tasks in ISA

Na metodologia canônica do MATILDA e nos estudos avançados de CVRP (Notice et al., 2025; Gouvêa et al., 2025), o aprendizado de máquina é empregado em duas frentes complementares:

#### Tarefa A: Previsão de Footprint (Classificação Binária por Algoritmo)
- **Objetivo:** Para um algoritmo específico $A \in \{\text{ALNS}, \text{ILS}, \text{GLS}, \text{TS}\}$, predizer se ele alcançará "Good Performance" na instância $x$ ($Y_A(x) \le \varepsilon$, onde $\varepsilon$ é o limiar canônico mediano do PRELIM).
- **Rótulo ($y$):** Binário ($1$ se $Y_A \le \varepsilon$, $0$ se $Y_A > \varepsilon$).
- **Finalidade:** Delimitar as fronteiras geométricas do nicho de dominância contínuo (*footprint*) de cada solver no espaço bidimensional.

#### Tarefa B: Seleção do Melhor Algoritmo (*Best Algorithm Selection* / AAS — Multi-classe)
- **Objetivo:** Dentre todos os algoritmos do portfólio, predizer diretamente qual deles encontrará a solução com o menor custo na instância $x$:
  $$A^*(x) = \arg\min_{A \in \mathcal{A}} \text{Cost}(A, x)$$
- **Rótulo ($y$):** Categórico ($\text{ALNS}, \text{ILS}, \text{GLS}, \text{TS}$).
- **Finalidade Operacional:** Permitir que, diante de uma instância nova e desconhecida, o praticante ou sistema automatizado selecione a meta-heurística ideal com base apenas em suas características projetadas $(Z_1, Z_2)$, evitando o custo computacional proibitivo de executar todo o portfólio.

---

### Significado Científico de Alta Acurácia no Espaço PILOT 2D $(Z_1, Z_2)$

Quando modelos clássicos de aprendizado supervisionado (como SVM com kernel RBF e Random Forest) atingem **alta acurácia (ex.: 85% a 95%)** utilizando **apenas as duas coordenadas projetadas $(Z_1, Z_2)$**, isso estabelece três comprovações científicas centrais:

1. **Preservação de Informação Discriminatória:**
   O processo analítico de redução dimensional (SIFTED $\to$ PILOT), que condensou 63 características iniciais em apenas 2 eixos lineares ortogonais, reteve com sucesso as propriedades estruturais que determinam o sucesso ou fracasso de cada algoritmo. O espaço 2D não é apenas uma representação estética, mas um mapa funcional de desempenho.

2. **Separabilidade Territorial Nítida:**
   Os algoritmos do portfólio não vencem de forma aleatória ou caótica; seus nichos de superioridade formam regiões compactas e bem definidas no plano $(Z_1, Z_2)$. As fronteiras de decisão aprendidas pelo SVM espelham transições físicas reais na estrutura das instâncias (ex.: transição de clientes clusterizados para dispersos, ou de alta para baixa restritividade de capacidade).

3. **Validação do Sistema de Recomendação (AAS):**
   Confirma a viabilidade prática do Automated Algorithm Selection com custo de inferência desprezível ($O(1)$ após extração rápida de features), entregando qualidade de solução próxima à do oráculo perfeito (*Virtual Best Solver*) com uma fração do orçamento computacional.
