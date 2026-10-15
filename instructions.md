# Plano de Execução — Instance Space Analysis (ISA) do CVRP — Fase 1

## 0. Como este documento deve ser usado

Este plano é uma especificação técnica para um agente de codificação **escrever** o pipeline descrito abaixo — **não para executá-lo**. Quem executa é o autor da pesquisa, de forma **assíncrona, etapa por etapa**. Isso significa:

- Cada etapa numerada abaixo deve ser implementada como um **script/módulo independente**, com entrada e saída bem definidas em disco (CSV/Parquet/JSON), não como uma função monolítica que roda tudo de uma vez.
- A Etapa 5 (definição de $\varepsilon$) é um **ponto de parada obrigatório**: o script correspondente deve terminar sua execução reportando estatísticas e candidatos, **sem prosseguir automaticamente** para a Etapa 6. O usuário revisa o output, decide $\varepsilon$ manualmente, e só então roda o próximo script passando esse valor como parâmetro.
- Todos os parâmetros mencionados como "fixos" abaixo devem ser implementados como **configuráveis** (arquivo de config YAML/JSON ou argumentos de linha de comando), nunca hardcoded.

---

## 1. Escopo e dados de entrada

| Item | Detalhe |
|---|---|
| Instâncias | 10.000 instâncias CVRP artificiais, $N$ (nº de clientes) variando de 30 a 200 |
| BKS | Já existem, em arquivos `.bks` / `.sol`, para as 10.000 instâncias (soluções obtidas via HGS-CVRP, 3 rodadas × 10 min por instância, mediana como referência) |
| Formato de instância | Formato padrão `.vrp` (CVRPLib-like) |
| Métrica de Custo | Padrão TSPLIB `EUC_2D` com distâncias euclidianas arredondadas ao inteiro mais próximo (`nint(hypot(dx, dy))`), calculadas a partir de `NODE_COORD_SECTION`. Seções extras (como `EDGE_WEIGHT_SECTION` com penalidades climáticas hipotéticas) são desconsideradas para preservar equivalência 1:1 com os BKS gerados pelo HGS. |

**Tarefa do agente nesta etapa:** escrever um script de validação (`00_validate_inputs.py`) que:
1. Lê todas as instâncias e todos os `.bks` correspondentes.
2. Verifica correspondência 1:1 entre instância e BKS (sem instância órfã).
3. Reporta estatísticas básicas: distribuição de $N$, nº de instâncias válidas, eventuais arquivos corrompidos/faltantes.

---

## 2. Algoritmos a comparar (geração de $Y$ e das probing features)

O estudo avalia um portfólio de **4 metaheurísticas de trajetória**, **todas inicializadas com a mesma heurística construtiva: PCI (Parallel Cheapest Insertion)**:

1. **ALNS** (Adaptive Large Neighborhood Search):
   - Versão canônica com critério de aceitação por **Simulated Annealing** (Ropke & Pisinger, 2006), parametrizado via `SimulatedAnnealing.autofit(worse=0.05, accept_prob=0.5)` e biblioteca `alns 7.0.0`.
   - Operadores de destruição: Shaw, Worst, Random, Route, String-SISR (Christiaens & Vanden Berghe, 2020).
   - Operadores de reparação: Regret-2, Regret-3, Greedy.
   - Polimento de busca local em C++ via PyVRP (Vidal, 2022).
   - *Nota:* A antiga variante experimental baseada em Record-to-Record Travel (RRT) foi descontinuada/removida, consolidando o ALNS canônico com Simulated Annealing como padrão único.

2. **ILS** (Iterated Local Search):
   - Critério de aceitação "Better" (descida pura) + perturbação dinâmica com escape de estagnação (Lourenço, Martin & Stützle, 2003).
   - Camada de busca local exaustiva em C++ via PyVRP cobrindo 13 operadores de nó e rota (Vidal, 2022).
   - Instrumentado internamente para extração automática das 11 probing features (P1–P11).

3. **GLS** (Guided Local Search, via OR-Tools):
   - Penalização adaptativa de arcos com coeficiente dependente de $N$ e do comprimento médio de rota: $\lambda(N, \bar{L}_{route}) = \text{clamp}(0.45, 1.25, 0.25 + 0.005 N + 0.025 \bar{L}_{route})$.
   - LNS dinâmico: $\text{LNS}(N, \bar{L}_{route}) = \text{clamp}(60, 250, \text{int}(40 + 0.8 N + 8 \bar{L}_{route}))\text{ ms}$.
   - Operadores ativos: `use_cross_exchange=True`, `use_tsp_opt=True`, `use_relocate_neighbors=True`, `use_full_path_lns=False`.

4. **TS** (Tabu Search, via OR-Tools):
   - Metaheurística baseada em memória explícita de curto prazo (Glover, 1989), proibindo arcos recentemente modificados para forçar diversificação contínua.
   - Fornece complementaridade ao portfólio no espaço de instâncias (especialmente em instâncias de alta densidade e rotas curtas).

### 2.1 Justificativa Metodológica do Portfólio (Texto base para o Artigo)

> **Enquadramento para a seção de Metodologia do artigo:**
> *"Diferentemente de estudos de competição voltados a identificar o solver estado-da-arte absoluto (como Gouvêa et al., 2025, com os finalistas do DIMACS, ou Notice et al., 2025, comparando HGS e FILO), este estudo investiga como diferentes **mecanismos clássicos de controle de trajetória em busca local** interagem com a estrutura das instâncias ao longo do orçamento de tempo. Para isolar o efeito do paradigma de diversificação/intensificação, fixamos a solução inicial utilizando a mesma heurística construtiva (Parallel Cheapest Insertion — PCI) em todos os métodos e avaliamos quatro arquiteturas canônicas de trajetória única: (i) destruição e reconstrução em larga escala com aceitação estocástica por Simulated Annealing (ALNS); (ii) perturbação estrutural com critério de aceitação elitista de descida pura (ILS); (iii) modificação dinâmica da função objetivo por penalização de atributos (GLS); e (iv) restrição determinística de movimentos via memória de curto prazo (TS). Esse desenho controlado permite mapear, no espaço de instâncias, sob quais condições topológicas e em qual fração do orçamento de tempo cada estratégia de escape de ótimos locais supera as demais, fornecendo a base analítica para sistemas de Seleção Automática de Algoritmos (AAS) baseados em portfólios de trajetória."*

- **Nota técnica sobre implementação e busca local (a documentar em `params_log.md`):** O portfólio opera sob duas arquiteturas de software sob orçamento *wall-clock*: ALNS e ILS utilizam controle em Python com delegação da busca local para o motor C++ do PyVRP, enquanto GLS e TS executam integralmente sobre o solver em C++ do Google OR-Tools. Essa distinção faz parte da definição operacional dos solvers disponíveis para o sistema de AAS.

### 2.2 Orçamento de tempo

$$T(N) = \min(0.5 \times N,\ \text{cap})$$

- Taxa: 0.5 segundos por cliente (mantida por coerência direta com o sistema AAS do autor, que opera sob este mesmo orçamento).
- Cap: **100 segundos** (configurável via parâmetro; para $N \in [30, 200]$, a função comporta-se de forma puramente linear entre 15s e 100s).

### 2.3 Execução e checkpoints

Para cada instância × cada um dos **4 algoritmos** do portfólio (`ALNS`, `ILS`, `GLS`, `TS`):
- Rodar **3 vezes** com seeds diferentes, até o tempo $T(N)$.
- Durante cada rodada, capturar o **custo da melhor solução incumbente** nos checkpoints de **25%, 50%, 75% e 100%** de $T(N)$ (via registrador temporal / callback de incumbente).
- Ao final das 3 rodadas, calcular a **mediana** do custo incumbente em cada checkpoint, por instância e por algoritmo.

**Output esperado desta etapa (`01_run_algorithms.py`):**
- `outputs/checkpoint_costs_raw.csv`: `instancia_id, algoritmo, seed, checkpoint, elapsed_target_seconds, cost`
- `outputs/checkpoint_costs_median.csv`: `instancia_id, algoritmo, checkpoint, cost_median_3_seeds, n_seeds`

### 2.4 Probing features (P1–P11)

Extraídas durante a fase inicial de busca local do **ILS** (partindo da construção PCI até a convergência ao primeiro mínimo local / janela curta de sondagem, evitando vazamento de dados com o custo final em $T(N)$). O script exporta:
- `outputs/ils_trace.csv`: features brutas por rodada (instância e seed).
- `outputs/ils_probing_median.csv`: features agregadas pela mediana das 3 seeds por instância, prontas para fusão na matriz $F$.

---

## 3. Métrica de dificuldade / desempenho ($Y$)

Para cada instância $i$ e algoritmo $a \in \{\text{ALNS}, \text{ILS}, \text{GLS}, \text{TS}\}$, usando o custo mediano no checkpoint de **100%** (solução final):

$$Y_{i,a} = \frac{\text{custo}_{i,a}^{100\%} - BKS_i}{BKS_i}$$

$Y$ é, portanto, uma **matriz de 4 colunas** (uma para cada algoritmo do portfólio) — é essa matriz que alimenta o PRELIM, o SIFTED e o PILOT.

Os gaps nos checkpoints de 25%, 50% e 75% **não** entram no treinamento da projeção (PRELIM/SIFTED/PILOT) — eles são armazenados em `Y_all_checkpoints.parquet` e usados na Seção 8 para colorir o espaço de instâncias fixo em diferentes fases do orçamento.

---

## 4. Matriz de características ($F$) e Invariância de Escala

Seis categorias, calculadas para cada uma das 10.000 instâncias (`02_extract_features.py`).

### 4.1 Pré-processamento de Invariância Espacial e de Tamanho ($N$)
O PRELIM normaliza colunas entre instâncias (Min-Max/Box-Cox), mas não remove a dependência intrínseca das features em relação ao domínio das coordenadas ou ao número de clientes $N$. Portanto, o extrator de features deve aplicar as seguintes transformações matemáticas:

1. **Normalização do Grid Espacial para $[0, 1000] \times [0, 1000]$ (Notice et al., 2025, p. 4):**
   ```python
   # Escalar coordenadas (depósito + clientes) para grid fixo [0, 1000] x [0, 1000]
   min_c = coords.min(axis=0)
   max_c = coords.max(axis=0)
   span = np.maximum(max_c - min_c, 1e-6)
   norm_coords = ((coords - min_c) / span) * 1000.0
   ```
2. **Normalização de Features Extensivas por $N$:**
   - `ND2_min_edge_sum`: Dividir por $N$ (custo médio da aresta mínima por vértice) **e** gerar variante normalizada por $\sqrt{N}$ (invariante à densidade assintótica de pontos no plano, evitando colinearidade exata com `NN1_mean`).
   - `MST1_edge_cost_sum`: Dividir por $N$ (como o grafo CVRP possui $N$ clientes + 1 depósito = $N+1$ vértices, a MST possui exatamente $N$ arestas; dividir por $N$ fornece o custo médio de aresta da MST).
   - `MST3_tree_depth`: Dividir por $N$ $\implies$ profundidade relativa da árvore em $[0, 1]$.
   - `G1_bounding_box`: Como `norm_coords` fixa a caixa em $1000 \times 1000$, extrair a **razão de aspecto** ($\text{span}_x / \text{span}_y$) das coordenadas originais em vez da área bruta pós-escalonamento.

### 4.2 Catálogo de Features (6 Categorias)

**ND — Distribuição de nós**
- ND1: estatísticas básicas da matriz de distância (média, desvio-padrão, mediana, curtose, assimetria)
- ND2: soma das arestas de menor custo (normalizada por $N$ e por $\sqrt{N}$)
- ND3: fração de distâncias distintas
- ND4: posição (x, y) do centroide no grid normalizado
- ND5: estatísticas da distância dos clientes ao centroide
- ND6: número de clusters (via DBSCAN)
- ND7: estatísticas do tamanho relativo dos clusters
- ND8: estatísticas da distância entre centroides de clusters
- ND9: razão nº de clusters / nº de cidades ($N$)

**MST — Árvore geradora mínima**
- MST1: estatísticas do custo das arestas da MST (incluindo soma dividida por $N$ arestas)
- MST2: estatísticas de dispersão e máximo do grau dos nós na MST (`std`, `max`; a média é omitida por ser identicamente $2N/(N+1)$ para qualquer árvore de $N+1$ vértices)
- MST3: profundidade da MST a partir do depósito dividida por $N$

**P — Probing (via sondagem inicial do ILS a partir do PCI)**
- P1: número de passos de melhora (bruto e dividido por $N$)
- P2: comprimentos de aresta em quartis
- P3: estatísticas do comprimento do segmento de rota
- P4: estatísticas da contagem de arestas no segmento
- P5: estatísticas do comprimento de aresta no segmento
- P6: custo do tour pela heurística construtiva (PCI)
- P7: comprimento do tour no mínimo local (e razão de melhoria P7/P6)
- P8: interseções do tour no plano (bruto e dividido por $N$)
- P9: melhoria média por passo
- P10: número de passos até o mínimo local
- P11: estatísticas da probabilidade/persistência de arestas nos mínimos locais

**G — Geométricas**
- G1: razão de aspecto ($\Delta x / \Delta y$) do retângulo envolvente original (Notice et al., 2025; área redundante pós-normalização é omitida)
- G2: área do fecho convexo no grid $[0, 1000]^2$
- G3: proporção de pontos sobre o fecho convexo
- G4: estatísticas da distância dos pontos internos ao contorno do fecho
- G5: estatísticas dos comprimentos das arestas do fecho convexo

**NN — Vizinhança mais próxima**
- NN1: estatísticas da distância ao 1º vizinho mais próximo
- NN2: número de componentes fortemente conexos no grafo kNN direcionado (dividido por $N$)
- NN3: número de componentes fracamente conexos (dividido por $N$)
- NN4: estatísticas do tamanho dos componentes fortemente conexos
- NN5: estatísticas do tamanho dos componentes fracamente conexos
- NN6: estatísticas do grau de entrada no grafo kNN direcionado (`std`, `max`; a média é identicamente $3.0$ para $k=3$ pelo Lema do Aperto de Mãos e foi omitida por variância zero)
- NN7: razão entre nº de componentes fortes e fracos
- NN8: estatísticas dos ângulos entre um nó e seus dois vizinhos mais próximos

**VRP — Específicas do problema**
- VRP1: distância do centroide ao depósito
- VRP2: estatísticas da distância dos clientes ao depósito
- VRP3: estatísticas das demandas dos clientes (normalizadas pela capacidade $Q$)
- VRP4: razão demanda total / capacidade total da frota
- VRP5: número médio de clientes por veículo ($N / K_{min}$)

**Output esperado:** tabela `F_raw.parquet` (10.000 linhas × todas as sub-features acima), indexada por `instancia_id`.

---

## 5. PRELIM — ponto de decisão manual ($\varepsilon$)

**Sub-etapa 5a — script `03_prelim_analyze.py` (roda até aqui e para):**
1. Recebe $F$ (Seção 4) e $Y$ no checkpoint 100% (Seção 3, **4 colunas**: `ALNS`, `ILS`, `GLS`, `TS`).
2. Calcula e reporta a distribuição de $Y$ (por algoritmo e agregada): média, desvio-padrão, mediana, percentis 10/25/50/75/90, proporção de zeros (empates com o BKS) e histograma.
3. Propõe 2–3 valores candidatos de $\varepsilon$ (limiar de "bom" desempenho) com justificativa estatística.
4. **Termina a execução aqui.** Não prossegue automaticamente para o SIFTED.

**Sub-etapa 5b — execução do pré-processamento (`03b_prelim_transform.py --epsilon <valor>`):**
O usuário revisa o relatório da Sub-etapa 5a, decide $\varepsilon$ manualmente e executa a transformação PRELIM gerando `F_norm.parquet`, `Y_norm.parquet` e a matriz binária `Y_bin.parquet` (10.000 × 4).

Parâmetros do PRELIM (configuráveis, com defaults abaixo):
- $\phi_{max}$ = `False` (minimização)
- $\phi_{bnd}$ = `False` (sem truncamento/winsorização de outliers)
- $\phi_{nrm}$ = `False` (PRELIM aplica Box-Cox/Yeo-Johnson e escalonamento Min-Max internamente)

---

## 6. SIFTED (`04_sifted.py`)

1. **Filtro de Relevância com Regra de Contingência Automática:**
   Calcular a correlação de Pearson $\vert{}r_{\text{feature}, Y_a}\vert{}$ para cada algoritmo $a \in \{\text{ALNS}, \text{ILS}, \text{GLS}, \text{TS}\}$ e reter features com $\vert{}r\vert{} \ge 0.3$ (limiar canônico do toolkit MATILDA, configurável via `--corr-threshold`) com pelo menos um dos 4 algoritmos. O código implementa salvaguarda canônica para garantir que haja candidatos suficientes para os $K=10$ clusters:

   ```python
   # Trecho do 04_sifted.py:
   # 1. Filtra inicialmente pelo limiar canônico (|r| >= corr_threshold, default 0.3)
   filtered = [f for f in sorted_features if corr_table[f]["max_abs"] >= corr_threshold]
   
   # 2. REGRA DE CONTINGÊNCIA CANÔNICA:
   # Se menos de K=10 features passarem, garante exatamente as top min_features (k_clusters) mais correlacionadas!
   min_features = args.k_clusters
   if len(filtered) < min_features:
       print(f"[AVISO] Apenas {len(filtered)} features atingiram |r| >= {corr_threshold}. "
             f"Expandindo para as top {min_features} features mais correlacionadas.")
       filtered = sorted_features[:min_features]
   ```

2. **Agrupamento por Redundância (K-Means):**
   Agrupar as features filtradas usando a matriz de dissimilaridade $1 - \vert{}r_{i,j}\vert{}$ com **$K = 10$ fixo** (valor default do toolkit canônico de Smith-Miles & Muñoz, documentado como escolha deliberada sem otimização via Silhouette/DB/CH), protegido por:

   ```python
   k_eff = min(k_clusters, len(feature_list))
   kmeans = KMeans(n_clusters=k_eff, n_init=30, max_iter=300, random_state=random_state)
   ```

3. **Seleção via Aprendizado Supervisionado (4 Random Forests):**
   Gerar combinações (1 feature por cluster) → projetar temporariamente em 2D via PCA → treinar **uma Random Forest por algoritmo (4 RFs)** prevendo $Y_a$ → calcular o erro OOB de cada uma → escolher a combinação que **minimiza o erro OOB médio entre as 4 RFs**.

**Output esperado:** lista final de 10 features selecionadas (`sifted_selected_features.json`), tabela de correlações, atribuição de clusters e erro OOB da combinação vencedora.

---

## 7. PILOT (`05_pilot.py`)

- $N_{try} = 30$
- $\phi_{num} = \text{False}$ (solução analítica via decomposição espectral/mínimos quadrados)
- **Input:** As 10 features selecionadas na Seção 6, e $Y$ (**4 colunas**, checkpoint 100%).
- **Output:** Matriz de projeção $Z \in \mathbb{R}^{2 \times 10}$ (equivalente à Eq. 1 de Gouvêa et al.) e as coordenadas $(Z_1, Z_2)$ de cada uma das 10.000 instâncias (`projected_coords.parquet`).

Esta é a **única** projeção construída — o espaço $(Z_1, Z_2)$ permanece fixo para a análise temporal da Seção 8.

---

## 8. Overlay dos checkpoints no espaço fixo (`06_visualize_overlays.py`)

Com o espaço $(Z_1, Z_2)$ fixado na Seção 7, para cada um dos 4 checkpoints (25%, 50%, 75%, 100%) e cada um dos **4 algoritmos** (`ALNS`, `ILS`, `GLS`, `TS`):

$$Y_{i,a}^{\text{checkpoint}} = \frac{\text{custo}_{i,a}^{\text{checkpoint}} - BKS_i}{BKS_i}$$

Gerar, para cada checkpoint, um conjunto de gráficos de dispersão no mesmo $(Z_1, Z_2)$:
- Um subplot por algoritmo (4 subplots por checkpoint), colorido pelo gap contínuo e pelo indicador binário de bom desempenho.
- Um plot "Melhor Algoritmo" por instância naquele checkpoint (menor gap entre os **4 algoritmos**, destacando empates).
- Cálculo de área, densidade e pureza dos *footprints* e distribuição do *performance regret* em cada checkpoint.

---

## 9. Outputs finais esperados (checklist de entrega)

1. `F_raw.parquet` e `F_norm.parquet` — matrizes de características bruta e normalizada (10.000 × features).
2. `Y.parquet` — gaps por instância × 4 algoritmos × 4 checkpoints (25/50/75/100%).
3. Relatório da Seção 5 (distribuição de $Y$ nos 4 algoritmos, candidatos de $\varepsilon$).
4. Artefatos do SIFTED: correlações, clusters ($K=10$), combinação vencedora, erros OOB das 4 RFs.
5. `Z_matrix.json` — matriz de projeção PILOT $2 \times 10$.
6. `projected_coords.parquet` — $(Z_1, Z_2)$ por instância.
7. Gráficos do espaço de instâncias — distribuição de features + overlays por checkpoint/algoritmo da Seção 8.
8. `params_log.md` — log com todos os parâmetros usados ($\varepsilon$ escolhido e justificativa, $K=10$, construção PCI fixa, $T(N)$, nº de rodadas, arquitetura dos solvers).

---

## 10. Resumo de parâmetros finais

| Parâmetro | Valor |
|---|---|
| Algoritmos comparados (4) | `ALNS` (SA Ropke & Pisinger), `ILS`, `GLS` (OR-Tools sintonizado), `TS` (OR-Tools Tabu Search) |
| Heurística construtiva (fixa) | PCI (Parallel Cheapest Insertion) em todos os 4 algoritmos |
| Orçamento de tempo | $T(N) = \min(0.5N,\ 100\text{s})$ |
| Rodadas por instância/algoritmo | 3 (mediana) |
| Checkpoints | 25%, 50%, 75%, 100% de $T(N)$ |
| Nº de instâncias | 10.000 ($N \in [30, 200]$) |
| Fonte de $Y$ | Gap vs. BKS no checkpoint 100% (**matriz de 4 colunas**) |
| Fonte das probing features | Sondagem local inicial do ILS (`outputs/ils_probing_median.csv`) |
| $\varepsilon$ | **Decisão manual do usuário após revisar a distribuição de $Y$ (Seção 5)** |
| $\phi_{max} / \phi_{bnd} / \phi_{nrm}$ | `False` / `False` / `False` |
| Filtro SIFTED | $|r| \ge 0.3$ (limiar MATILDA canônico, configurável via `--corr-threshold`) com *fallback* automático para `top k_clusters` |
| $K$ (SIFTED) | 10, fixo (com salvaguarda `k_eff = min(10, len(features))`) |
| Avaliador SIFTED | 4 Random Forests (minimizando erro OOB médio) |
| $N_{try} / \phi_{num}$ (PILOT) | 30 / `False` |

---

## 11. Avaliação Crítica do Projeto e Potencial de Publicação

Esta seção consolida os pontos fortes, as limitações metodológicas e os argumentos de originalidade do estudo para orientar a escrita do artigo e antecipar questionamentos de revisores.

### 11.1 O que traz de novo (Contribuições e Preenchimento de Lacunas)
1. **Dinâmica *Anytime* sobre Topologia Estática:**
   - Gouvêa et al. (2025) reduzem o comportamento temporal do algoritmo a um único escalar (Primal Integral).
   - Notice et al. (2025) avaliam dois cortes de tempo (curto e longo), mas treinam projeções separadas para cada orçamento, alterando as features selecionadas e a geometria dos eixos $(Z_1, Z_2)$ entre os dois cenários.
   - **Diferencial deste trabalho:** Fixar a base de projeção $Z$ no orçamento completo (100%) e sobrepor os estados intermediários (25%, 50%, 75%, 100%) permite rastrear o deslocamento das fronteiras de dominância (*footprints*) ao longo do tempo sobre o mesmo sistema de coordenadas, conectando diretamente a análise de espaço de instâncias (ISA) ao problema de Seleção Automática de Algoritmos (AAS) sensível ao orçamento.
2. **Isolamento Experimental da Fase de Melhoria:**
   - Estudos anteriores comparam pacotes fechados onde a heurística construtiva varia entre os competidores (ex.: Clarke & Wright no FILO vs. inicialização populacional aleatória/construtiva no HGS). Ao fixar a construção via *Parallel Cheapest Insertion* (PCI) nos 4 métodos, este trabalho elimina a variável de confusão da solução inicial, garantindo que a separação no espaço de instâncias reflita exclusivamente a interação entre a topologia da instância e o mecanismo de busca local/escape de ótimos locais.
3. **Densidade e Escala Amostral:**
   - O uso de 10.000 instâncias com BKS conhecidos representa um aumento de 5x a 50x em relação aos estudos recentes de ISA para CVRP (centenas de instâncias do DIMACS em Gouvêa et al., 2025; 1.938 instâncias em Notice et al., 2025), reduzindo a variância do erro *out-of-bag* (OOB) das Random Forests no SIFTED e aumentando a estabilidade da matriz linear do PILOT.

### 11.2 Pontos Fortes do Desenho Experimental
- **Diversidade de paradigmas de trajetória:** O portfólio cobre quatro filosofias distintas de escape de ótimos locais (aceitação estocástica em vizinhança grande via ALNS-SA, perturbação intensiva via ILS, alteração da paisagem de busca via penalidades no GLS e restrição por memória de curto prazo no TS).
- **Prevenção de *Data Leakage* e Escala Espacial:** A extração das *probing features* restrita à sondagem inicial e a padronização das coordenadas para o grid $[0, 1000]^2$ com normalização por $N$ evitam que a projeção linear degrade em um mero preditor do tamanho da instância ($N$) ou incorpore a própria variável resposta $Y$.
- **Reprodutibilidade e Modularidade:** Pipeline assíncrono com parada obrigatória para inspeção da distribuição de $Y$ antes da binarização por $\varepsilon$ e salvaguardas automáticas de código no SIFTED.

### 11.3 Pontos Fracos e Riscos (Como Mitigar na Escrita do Artigo)
1. **Ausência de métodos populacionais estado-da-arte no portfólio avaliado:**
   - *Crítica esperada:* O HGS (usado para gerar os BKS) e o FILO superam consistentemente os solvers do OR-Tools (GLS/TS) e metaheurísticas de trajetória simples em benchmarks padrão.
   - *Mitigação no texto:* Posicionar o artigo explicitamente como uma análise do comportamento de **metaheurísticas clássicas de trajetória única sob inicialização controlada**, e não como uma competição pelo estado-da-arte irrestrito do CVRP (ver texto da Seção 2.1).
2. **Escala moderada do número de clientes ($N \in [30, 200]$):**
   - *Crítica esperada:* Instâncias de até 200 clientes são consideradas de pequeno a médio porte na literatura atual de CVRP, podendo apresentar alta taxa de soluções ótimas (empates em gap = 0%) após 100 segundos de execução.
   - *Mitigação no texto:* Destacar que a análise dos checkpoints prematuros (25% e 50% do tempo, equivalentes a apenas 3.75s–25s para $N=30$ e 25s–50s para $N=200$) captura a fase transitória de convergência onde os algoritmos apresentam maior divergência de desempenho e maior *performance regret*.
3. **Assimetria de linguagem de implementação sob tempo de relógio (*wall-clock*):**
   - *Crítica esperada:* ALNS e ILS possuem o laço de controle em Python (com busca local em C++ via PyVRP), enquanto GLS e TS rodam inteiramente em C++ no OR-Tools.
   - *Mitigação no texto:* Documentar abertamente a arquitetura de software na seção experimental, esclarecendo que na aplicação prática de um sistema de AAS, o seletor escolhe entre implementações concretas disponíveis sob um orçamento de tempo real de CPU, tornando relevante mapear o impacto conjunto do algoritmo e de sua implementação no espaço de instâncias.

### 11.4 Veículos Alvo Recomendados para Submissão
- **Conferências:** GECCO (trilha ECOM/EMOO), PPSN, EvoCOP, IJCNN, LION, IEEE CEC, BRACIS.
- **Periódicos:** *Computers & Operations Research (COR)*, *Journal of Heuristics*, *Engineering Applications of Artificial Intelligence (EAAI)*, *Applied Soft Computing*, *European Journal of Operational Research (EJOR)* (caso a análise teórica das transições entre checkpoints seja aprofundada).