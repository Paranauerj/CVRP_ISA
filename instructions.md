# Plano de Execução — Instance Space Analysis (ISA) do CVRP — Fase 1

## 0. Como este documento deve ser usado

Este plano é uma especificação técnica para um agente de codificação **escrever** o pipeline descrito abaixo — **não para executá-lo**. Quem executa é o autor da pesquisa, de forma **assíncrona, etapa por etapa**. Isso significa:

- Cada etapa numerada abaixo deve ser implementada como um **script/módulo independente**, com entrada e saída bem definidas em disco (CSV/Parquet/JSON), não como uma função monolítica que roda tudo de uma vez.
- A Etapa 5 (definição de ε) é um **ponto de parada obrigatório**: o script correspondente deve terminar sua execução reportando estatísticas e candidatos, **sem prosseguir automaticamente** para a Etapa 6. O usuário revisa o output, decide ε manualmente, e só então roda o próximo script passando esse valor como parâmetro.
- Todos os parâmetros mencionados como "fixos" abaixo devem ser implementados como **configuráveis** (arquivo de config ou argumentos de linha de comando), nunca hardcoded, porque o pipeline será reaplicado a outros datasets no futuro (ver Seção 8).

---

## 1. Escopo e dados de entrada

| Item | Detalhe |
|---|---|
| Instâncias | 10.000 instâncias CVRP artificiais, N (nº de clientes) variando de 30 a 200 |
| BKS | Já existem, em arquivos `.bks`, para as 10.000 instâncias (soluções obtidas via HGS-CVRP, 3 rodadas × 10 min por instância, mediana como referência) |
| Formato de instância | (a definir pelo agente conforme o formato real dos arquivos do usuário — provavelmente `.vrp`/CVRPLib-like) |

**Tarefa do agente nesta etapa:** escrever um script de validação (`00_validate_inputs.py` ou similar) que:
1. Lê todas as instâncias e todos os `.bks` correspondentes.
2. Verifica correspondência 1:1 entre instância e BKS (sem instância órfã).
3. Reporta estatísticas básicas: distribuição de N, nº de instâncias válidas, eventuais arquivos corrompidos/faltantes.

---

## 2. Algoritmos a comparar (Fase 2, mas executados agora para gerar Y e as probing features)

Três metaheurísticas, **todas com a mesma heurística construtiva inicial: PCI (Parallel Cheapest Insertion)**:

1. **ALNS** (Adaptive Large Neighborhood Search)
2. **ILS** (Iterated Local Search — savings/2-opt/Or-opt como base, com critério de aceitação por perturbação)
3. **GLS** (Guided Local Search, via OR-Tools)

Fixar a construtiva isola o efeito da camada de busca local/metaheurística na comparação — a diferença de desempenho entre os três não pode ser atribuída à heurística de construção, porque ela é idêntica nos três.

### 2.1 Orçamento de tempo

$$T(N) = \min(0.5 \times N,\ \text{cap})$$

- Taxa: 0.5 segundos por cliente (reaproveitada do artigo anterior de AAS do usuário, por coerência entre os dois trabalhos — o objetivo final desta ISA é informar decisões do sistema AAS, que opera sob esse mesmo orçamento).
- Cap: **100 segundos** (valor válido para este dataset, onde N_max = 200, portanto o cap nunca é efetivamente atingido dentro do range 30-200 — a função é puramente linear neste dataset). Implementar como parâmetro configurável, pois outros datasets no futuro podem ter N > 200 e o cap passa a ser relevante.

### 2.2 Execução e checkpoints

Para cada instância × cada um dos 3 algoritmos:
- Rodar **3 vezes** com seeds diferentes, até o tempo $T(N)$.
- Durante cada rodada, capturar o **custo da melhor solução incumbente** nos checkpoints de **25%, 50%, 75% e 100%** de $T(N)$ (ex.: via callback/log de solução incumbente — necessário para os três algoritmos, incluindo o GLS do OR-Tools).
- Ao final das 3 rodadas, calcular a **mediana** do custo incumbente em cada checkpoint, por instância e por algoritmo.

**Output esperado desta etapa:** uma tabela/arquivo com colunas:
`instancia_id, algoritmo, checkpoint (25/50/75/100), custo_mediano_3_rodadas`

### 2.3 Probing features (P1–P11)

Extraídas a partir do **traço de execução do ILS** (nº de passos de melhora, comprimento no mínimo local, melhoria por passo etc. — ver Seção 4). O ILS foi escolhido por ser o algoritmo mais simples de instrumentar internamente (o GLS via OR-Tools não expõe esse traço facilmente).

---

## 3. Métrica de dificuldade / desempenho (Y)

Para cada instância $i$ e algoritmo $a \in \{ALNS, ILS, GLS\}$, usando o custo mediano no checkpoint de **100%** (solução final):

$$Y_{i,a} = \frac{\text{custo}_{i,a}^{100\%} - BKS_i}{BKS_i}$$

$Y$ é, portanto, uma **matriz de 3 colunas** (uma por algoritmo) — é essa matriz que alimenta o PRELIM/SIFTED, no mesmo espírito do Gouvêa et al. (que usa múltiplas colunas, uma por algoritmo do portfólio).

Os gaps nos checkpoints de 25/50/75% **não** entram no PRELIM/SIFTED — eles são usados só depois, na Seção 7, para colorir o mesmo espaço de instâncias em diferentes fases do orçamento.

---

## 4. Matriz de características (F)

Seis categorias, calculadas para cada uma das 10.000 instâncias. Todas as features $O(V^2)$ (ND, MST, G, NN, VRP) não dependem de nenhum algoritmo rodando; apenas P depende do traço do ILS.

### ND — Distribuição de nós
- ND1: estatísticas básicas da matriz de distância (média, desvio-padrão, mediana, curtose)
- ND2: soma das arestas de menor custo
- ND3: fração de distâncias distintas
- ND4: posição (x, y) do centroide
- ND5: distância dos clientes ao centroide
- ND6: número de clusters (ex.: via DBSCAN)
- ND7: tamanho dos clusters
- ND8: distância entre centroides de clusters
- ND9: razão nº de clusters / nº de cidades

### MST — Árvore geradora mínima
- MST1: estatísticas do custo das arestas
- MST2: estatísticas do grau dos nós
- MST3: profundidade da MST a partir do depósito

### P — Probing (via traço de execução do ILS)
- P1: número de passos de melhora
- P2: comprimentos de aresta em quartis
- P3: comprimento do segmento de rota
- P4: contagem de arestas no segmento
- P5: comprimento de aresta no segmento
- P6: custo do tour pela heurística construtiva (PCI)
- P7: comprimento do tour no mínimo local
- P8: interseções do tour no plano
- P9: melhoria por passo
- P10: número de passos até o mínimo local
- P11: probabilidade de arestas estarem presentes no mínimo local

### G — Geométricas
- G1: área do retângulo envolvente
- G2: área do fecho convexo
- G3: proporção de pontos sobre o fecho convexo
- G4: distância dos pontos internos ao contorno do fecho
- G5: comprimentos das arestas do fecho convexo

### NN — Vizinhança mais próxima
- NN1: distância ao 1º vizinho mais próximo
- NN2: número de componentes fortemente conexos (grafo kNN direcionado)
- NN3: número de componentes fracamente conexos
- NN4: tamanho dos componentes fortemente conexos
- NN5: tamanho dos componentes fracamente conexos
- NN6: grau de entrada do nó no grafo kNN direcionado
- NN7: razão entre nº de componentes fortes e fracos
- NN8: ângulos entre um nó e seus dois vizinhos mais próximos

### VRP — Específicas do problema
- VRP1: distância do centroide ao depósito
- VRP2: distância dos clientes ao depósito
- VRP3: estatísticas das demandas dos clientes
- VRP4: razão demanda total / capacidade total
- VRP5: número médio de clientes por veículo

**Output esperado:** tabela $F$ (10.000 linhas × todas as sub-features acima, brutas — sem normalizar ainda), indexada por `instancia_id`.

---

## 5. PRELIM — ponto de decisão manual (ε)

**Sub-etapa 5a — script que roda até aqui e para:**
1. Recebe $F$ (Seção 4) e $Y$ (Seção 3, 3 colunas).
2. Calcula e reporta a distribuição de $Y$ (por algoritmo e agregada): média, mediana, percentis 25/50/75/90, histograma.
3. Propõe 2–3 valores candidatos de ε com justificativa textual (ex.: "percentil 75 = X, captura as instâncias com gap acima da maioria").
4. **Termina aqui.** Não prossegue para SIFTED.

**Sub-etapa 5b — o usuário roda o script acima, decide ε manualmente, e então invoca a próxima etapa passando ε como parâmetro.**

Parâmetros do PRELIM (fixos, não sujeitos a decisão neste momento):
- $\phi_{max}$ = False (minimização)
- $\phi_{bnd}$ = False (sem limitação/winsorização de outliers)
- $\phi_{nrm}$ = False (PRELIM normaliza os dados brutos internamente)

---

## 6. SIFTED

1. Correlação de Pearson $|r_{\text{feature}, Y_a}|$ para cada algoritmo $a$; manter apenas features com $|r| \ge 0.5$ com pelo menos um dos 3 algoritmos.
2. K-means sobre a dissimilaridade $1 - |r_{i,j}|$ entre as features restantes.
   - **K = 10, fixo** (valor default do toolkit canônico de Smith-Miles & Muñoz — não otimizar via Silhouette/DB/CH; documentar essa escolha como deliberada).
3. Gerar combinações (1 feature por cluster) → projetar temporariamente em 2D via PCA → treinar uma **Random Forest por algoritmo** (3 RFs) prevendo $Y_a$ → calcular erro OOB de cada uma → escolher a combinação que **minimiza o erro OOB médio entre as 3**.

**Output esperado:** lista final de ~10 features selecionadas, tabela de correlações, atribuição de clusters, erro OOB da combinação vencedora.

---

## 7. PILOT

- $N_{try} = 30$
- $\phi_{num}$ = False (solução analítica, não numérica)
- Input: as features selecionadas na Seção 6, e $Y$ (3 colunas, checkpoint 100%)
- Output: matriz de projeção $Z$ (equivalente à Eq. 1 do Gouvêa et al.) e as coordenadas $(Z_1, Z_2)$ de cada uma das 10.000 instâncias.

Esta é a **única** projeção construída — não se refaz PILOT por checkpoint (ver Seção 8).

---

## 8. Overlay dos checkpoints no espaço fixo

Com o espaço $(Z_1, Z_2)$ fixado na Seção 7:

Para cada um dos 4 checkpoints (25/50/75/100%) e cada um dos 3 algoritmos:
$$Y_{i,a}^{checkpoint} = \frac{\text{custo}_{i,a}^{checkpoint} - BKS_i}{BKS_i}$$

Gerar, para cada checkpoint, um conjunto de gráficos de dispersão no mesmo $(Z_1, Z_2)$:
- Um subplot por algoritmo, colorido pelo gap (gradiente contínuo) — equivalente às Figs. 4/5 do Gouvêa e à Fig. 4 do Notice.
- Um plot "melhor algoritmo" por instância naquele checkpoint (menor gap entre os 3) — equivalente à "Best Selection" da Fig. 3 do Notice.

Isso permite visualizar diretamente se/como a região de domínio de cada algoritmo muda conforme a fase do orçamento avança — a peça central da conexão com o artigo de AAS.

---

## 9. Outputs finais esperados (checklist de entrega)

1. `F.parquet` — matriz de características bruta e normalizada, 10.000 × N features
2. `Y.parquet` — gaps por instância × algoritmo × checkpoint (25/50/75/100%)
3. Relatório da Seção 5 (distribuição de Y, candidatos de ε)
4. Artefatos do SIFTED: correlações, clusters, combinação vencedora, erros OOB
5. `Z_matrix.json` ou `.npy` — matriz de projeção PILOT (o artefato reutilizável mais importante)
6. `projected_coords.parquet` — $(Z_1, Z_2)$ por instância
7. Gráficos do espaço de instâncias — 1 estático (colorido por dificuldade geral) + os overlays por checkpoint/algoritmo da Seção 8
8. `params_log.md` — log com todos os parâmetros usados (ε escolhido e justificativa, K=10, construção PCI fixa, $T(N)$, nº de rodadas etc.) — vira rascunho direto da seção de metodologia do artigo

---

## 10. Extensibilidade para outros datasets (ex.: Set X do Uchoa)

Para permitir reaplicar o espaço já construído a instâncias externas sem refazer SIFTED/PILOT:

- A função de extração de features (Seção 4) deve ser um módulo independente, reutilizável com qualquer arquivo de instância no formato correto (ou um conversor para esse formato).
- Salvar, além da matriz $Z$: os parâmetros de bound/scale do PRELIM e a lista final de features selecionadas pelo SIFTED.
- Projetar uma instância nova = extrair as mesmas features → aplicar o mesmo bound/scale do PRELIM → selecionar as mesmas features do SIFTED → multiplicar por $Z$.
- Isso viabiliza, por exemplo, sobrepor o Set X (ou outros benchmarks públicos) no mesmo mapa já construído, sem rodar nenhum algoritmo neles — só para fins de comparação de cobertura/diversidade do espaço de instâncias.

---

## 11. Resumo de parâmetros finais

| Parâmetro | Valor |
|---|---|
| Algoritmos comparados | ALNS, ILS, GLS |
| Heurística construtiva (fixa) | PCI |
| Orçamento de tempo | $T(N) = \min(0.5N,\ 100s)$ |
| Rodadas por instância/algoritmo | 3 (mediana) |
| Checkpoints | 25%, 50%, 75%, 100% de T |
| Nº de instâncias | 10.000 (N entre 30 e 200) |
| Fonte de Y | Gap vs. BKS, checkpoint 100%, 3 colunas (uma por algoritmo) |
| Fonte das probing features | Traço de execução do ILS |
| ε | **Não fixado aqui — decisão manual do usuário após revisar a distribuição de Y (Seção 5)** |
| φmax / φbnd / φnrm | False / False / False |
| K (SIFTED) | 10, fixo |
| N_try / φnum (PILOT) | 30 / False |