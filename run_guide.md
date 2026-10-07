# Guia de Execucao - ISA CVRP

## 1. O que este projeto executa

O projeto usa Python e trabalha com a fonte Gaetano:

```text
C:\Users\jptin\OneDrive\USAL\Codes\CVRPBenchmark\instances\gaetano
```

Essa pasta contem:

- 10.000 arquivos `.vrp`;
- 10.000 arquivos `.sol`, usados como BKS;
- nenhum arquivo `.bks` separado.

O nome-base dos arquivos e a chave de correspondencia. O arquivo `.vrp` contem a matriz de custos, demandas e capacidade. O `.sol` contem as rotas e a linha `Cost`.

## 2. Pre-requisitos

Usar Python 3.9 ou superior. O ambiente precisa conter, no minimo:

```text
numpy
vrplib
ortools
```

Para a implementacao definitiva do ALNS, instalar tambem:

```powershell
python -m pip install alns==7.0.0
```

A biblioteca oficial e `N-Wouda/ALNS`. Ela fornece o framework ALNS, mas os operadores CVRP, o estado da solucao e a funcao objetivo continuam sendo responsabilidade do projeto.

Verificacao rapida:

```powershell
python -c "import numpy, vrplib, ortools; print('base dependencies ok')"
python -c "import alns; print('alns', alns.__version__)"
```

No estado atual, `alns` ainda nao esta instalado no interpretador usado nos testes.

## 3. Validar as entradas

Executar a validacao antes de qualquer experimento:

```powershell
python .\00_validate_inputs.py `
  --instances-dir "C:\Users\jptin\OneDrive\USAL\Codes\CVRPBenchmark\instances\gaetano" `
  --report .\outputs\input_validation.csv
```

Resultado esperado para Gaetano:

```text
instances=10000
bks_files=10000 (extension=.sol)
valid=10000 invalid=0 orphan_bks=0
```

A validacao tambem imprime a distribuicao do numero de clientes e grava uma linha por instancia em `outputs/input_validation.csv`.

## 4. Smoke test

Sempre testar uma instancia antes de iniciar uma amostra maior:

```powershell
python .\01_run_algorithms.py `
  --instances-dir "C:\Users\jptin\OneDrive\USAL\Codes\CVRPBenchmark\instances\gaetano" `
  --algorithms ALNS ILS GLS `
  --seeds 1001 2001 3001 `
  --workers 8 `
  --limit 1 `
  --seconds-per-customer 0.01 `
  --time-cap 1.0 `
  --output .\outputs\smoke_raw.csv `
  --median-output .\outputs\smoke_median.csv
```

Esse comando executa 9 rodadas: 3 algoritmos x 3 seeds, com 4 checkpoints por rodada.

Nao usar `0.001` segundo por cliente para validar GLS. Em testes anteriores, isso produziu `inf` porque o OR-Tools nao teve tempo suficiente para devolver uma solucao. Esse resultado nao deve ser interpretado como um custo valido.

## 5. Amostra recomendada

Antes da campanha completa, usar uma amostra pequena:

```powershell
python .\01_run_algorithms.py `
  --instances-dir "C:\Users\jptin\OneDrive\USAL\Codes\CVRPBenchmark\instances\gaetano" `
  --algorithms ALNS ILS GLS `
  --seeds 1001 2001 3001 `
  --workers 8 `
  --limit 10 `
  --output .\outputs\sample10_raw.csv `
  --median-output .\outputs\sample10_median.csv
```

Depois conferir:

- nenhum custo `inf`;
- todas as rotas cobrem cada cliente exatamente uma vez;
- nenhuma rota excede a capacidade;
- os quatro checkpoints existem;
- a mediana possui `n_seeds=3`;
- o custo GLS usa a mesma escala dos outros algoritmos.

## 6. Campanha completa

Com os defaults do plano e inclusão de TS e probing features, o comando é:

```powershell
python .\01_run_algorithms.py `
  --instances-dir "C:\Users\jptin\OneDrive\USAL\Codes\CVRPBenchmark\instances\gaetano" `
  --algorithms ALNS ILS GLS TS `
  --seeds 1001 2001 3001 `
  --workers 8 `
  --seconds-per-customer 0.5 `
  --time-cap 100.0 `
  --output .\outputs\checkpoint_costs_raw.csv `
  --median-output .\outputs\checkpoint_costs_median.csv `
  --ils-trace-output .\outputs\ils_trace.csv `
  --ils-median-output .\outputs\ils_probing_median.csv
```

O runner aceita `--workers` e usa processos independentes. Para execução em Linux via SSH, consulte o guia detalhado `phase2_execution_guide.md`.

## 7. Saidas

Com 10.000 instâncias, 4 algoritmos e 3 seeds:

- `checkpoint_costs_raw.csv`: aproximadamente 480.000 linhas (120.000 rodadas x 4 checkpoints);
- `checkpoint_costs_median.csv`: aproximadamente 160.000 linhas (10.000 instâncias x 4 algoritmos x 4 checkpoints);
- `ils_trace.csv`: 30.000 linhas com as 11 probing features brutas do ILS (P1–P11);
- `ils_probing_median.csv`: 10.000 linhas com a mediana das probing features por instância, prontas para fusão na matriz F.

Cada linha raw contem:

```text
instancia_id, algoritmo, seed, checkpoint, elapsed_target_seconds, cost
```

Cada linha agregada contem:

```text
instancia_id, algoritmo, checkpoint, cost_median_3_seeds, n_seeds
```

O custo final sera usado depois para calcular:

```text
gap = (cost_median_3_seeds - bks_cost) / bks_cost
```

## 8. Tempo estimado para Gaetano

A validacao real encontrou:

```text
instancias = 10.000
clientes totais = 1.132.530
clientes medios por instancia = 113,253
orcamento por algoritmo = min(0,5 * N, 100 s)
```

Como todos os N estao entre 30 e 200, o cap de 100 segundos nao reduz o maior caso. O tempo de algoritmo por algoritmo, somando todas as instancias, e:

```text
0,5 * 1.132.530 = 566.265 segundos = 157,30 horas = 6,55 dias
```

A campanha completa possui 9 combinacoes por instancia:

```text
3 algoritmos * 3 seeds = 9 rodadas por instancia
```

Logo, o limite teorico de tempo de algoritmo e:

```text
566.265 * 9 = 5.096.385 segundos
                  = 1.415,66 horas
                  = 58,99 dias
```

Essa e uma estimativa minima para o runner sequencial, sem contar leitura, construcao PCI, calculo de vizinhancas, callbacks, inicializacao do OR-Tools e overhead do sistema.

Estimativa idealizada com paralelismo, apenas para planejamento:

| Workers independentes | Tempo teorico | Planejamento pratico |
|---:|---:|---:|
| 1 | 59,0 dias | 60+ dias |
| 8 | 7,4 dias | 8-10 dias |
| 16 | 3,7 dias | 4-6 dias |
| 32 | 1,8 dias | 2,5-4 dias |

Benchmark observado do runner com 8 instancias, ILS, uma seed, 0,02 s/cliente e cap de 2 s:

| Workers | Tempo | Speedup |
|---:|---:|---:|
| 1 | 16,586 s | 1,00x |
| 2 | 8,942 s | 1,85x |
| 4 | 4,975 s | 3,33x |
| 8 | 3,435 s | 4,83x |

O benchmark mostra que o speedup nao e linear. Nao se deve multiplicar simplesmente por 8 ou 16 sem considerar memoria, processos, escrita concorrente e estabilidade do tempo-limite.

## 9. Estado atual — Todos os bloqueios resolvidos

O runner `01_run_algorithms.py` está pronto para produção e totalmente alinhado com a literatura:

1. **ALNS consolidado:** Implementado com a biblioteca oficial `alns 7.0.0`, usando aceitação por Simulated Annealing (`SimulatedAnnealing.autofit`, Ropke & Pisinger 2006) e operadores de destruição/reparação com polimento PyVRP C++. A antiga variante RRT foi removida.
2. **ILS com 13 operadores e probing:** Implementado com critério "Better" e perturbação dinâmica com escape de estagnação (Lourenço et al. 2003) + busca local exaustiva PyVRP C++. Extrai automaticamente as 11 probing features (P1–P11) para `outputs/ils_trace.csv` e `outputs/ils_probing_median.csv`.
3. **GLS com escala float unificada e fórmula sintonizada:** Custos com escala unificada via float com escala 100 no OR-Tools, penalização dependente de $N$ e comprimento médio de rota ($\lambda(N, \bar{L}_{route})$) e LNS dinâmico.
4. **TS adicionado ao portfólio:** Tabu Search via OR-Tools integrado com memória de curto prazo e construtiva PCI idêntica.
5. **Probing features integradas:** Gera simultaneamente `checkpoint_costs_raw.csv`, `checkpoint_costs_median.csv`, `ils_trace.csv` e `ils_probing_median.csv`.

## 10. Execucao das Etapas ISA (Pipeline de Ponta a Ponta)

Uma vez concluída a execução do benchmark (Etapa 1), as fases subsequentes de Instance Space Analysis são executadas sequencialmente:

### Etapa 2 — Extracao da Matriz de Caracteristicas (F)
Extrai 50 características estáticas estruturais (ND, MST, G, NN, VRP) e mescla as 13 colunas de probing dinâmico do ILS:
```powershell
python .\02_extract_features.py `
  --instances-dir "C:\Users\jptin\OneDrive\USAL\Codes\CVRPBenchmark\instances\gaetano" `
  --probing-file .\outputs\ils_probing_median.csv `
  --workers 8 `
  --output .\outputs\F.parquet `
  --csv-output .\outputs\F.csv
```

### Etapa 3 — PRELIM (Matriz de Gaps Y e Ponto de Decisao de epsilon)
Calcula os gaps relativos vs. BKS a 100% de tempo, avalia a distribuição empírica (P10 a P90) e aplica o critério mediano canônico do MATILDA:
```powershell
python .\03_prelim.py `
  --instances-dir "C:\Users\jptin\OneDrive\USAL\Codes\CVRPBenchmark\instances\gaetano" `
  --median-costs .\outputs\checkpoint_costs_median.csv `
  --features .\outputs\F.parquet `
  --output-y .\outputs\Y.parquet `
  --csv-output .\outputs\Y.csv `
  --epsilon auto
```
*Gera: `outputs/prelim_distribution.png`, `outputs/prelim_report.md` e `outputs/prelim_summary.json`.*

### Etapa 4 — SIFTED (Selecao de K=10 Features)
Aplica filtro de correlação de Pearson ($|r| \ge 0.5$), agrupa em $K=10$ clusters ortogonais e seleciona o subconjunto minimizando o erro Out-of-Bag (OOB) do Random Forest:
```powershell
python .\04_sifted.py `
  --features .\outputs\F.parquet `
  --performance .\outputs\Y.parquet `
  --k-clusters 10 `
  --output-parquet .\outputs\F_sifted.parquet `
  --output-csv .\outputs\F_sifted.csv `
  --output-features .\outputs\sifted_features.json
```
*Gera: `outputs/F_sifted.parquet`, `outputs/sifted_clusters.png` e `outputs/sifted_report.md`.*

### Etapa 5 — PILOT (Projecao Linear Analitica e Rotacao Otimizada)
Resolve a decomposição espectral generalizada em forma fechada ($\phi_{\text{num}} = \text{False}$), otimiza $N_{\text{try}} = 30$ rotações e gera a Equação 1 canônica:
```powershell
python .\05_pilot.py `
  --sifted-features .\outputs\F_sifted.parquet `
  --performance .\outputs\Y.parquet `
  --n-try 30 `
  --output-z .\outputs\Z_matrix.json `
  --output-coords .\outputs\projected_coords.parquet `
  --output-plot .\outputs\instance_space_pilot.png
```
*Gera: `outputs/Z_matrix.json`, `outputs/projected_coords.parquet`, `outputs/instance_space_pilot.png` e `outputs/pilot_report.md`.*

### Etapa 6 — Overlay de Checkpoints e Migracao de Footprint (25%, 50%, 75%, 100%)
Projeta os 4 checkpoints no espaço $(Z_1, Z_2)$ fixo, gerando mapas de Best Selection e curvas de migração de footprint para o artigo de AAS:
```powershell
python .\06_overlay_checkpoints.py `
  --raw-costs .\outputs\checkpoint_costs_raw.csv `
  --median-costs .\outputs\checkpoint_costs_median.csv `
  --instances-dir "C:\Users\jptin\OneDrive\USAL\Codes\CVRPBenchmark\instances\gaetano" `
  --coords .\outputs\projected_coords.parquet `
  --z-matrix .\outputs\Z_matrix.json
```
*Gera: `checkpoint_overlays_algorithms.png`, `checkpoint_overlays_best_selection.png`, `checkpoint_evolution_summary.png` e `params_log.md`.*

### Etapa 7 — Extensibilidade para Novos Datasets (ex.: Uchoa Set X)
Projeta novas instâncias no mesmo mapa geométrico SEM refazer SIFTED nem PILOT e sem executar algoritmos adicionais:
```powershell
python .\07_project_new_instances.py `
  --instances-dir "C:\caminho\para\novas_instancias" `
  --z-matrix .\outputs\Z_matrix.json `
  --ref-coords .\outputs\projected_coords.parquet `
  --output-coords .\outputs\projected_new_instances.parquet `
  --output-plot .\outputs\projected_new_instances_overlay.png
```
*Gera: `projected_new_instances.parquet`, `projected_new_instances_overlay.png` e `projected_new_instances_report.md`.*

---

## 11. Referencia Rapida de Scripts e Modulos

| Script | Etapa | Descrição |
|---|---|---|
| `00_validate_inputs.py` | 1 | Validação 1:1 entre instâncias `.vrp` e soluções `.sol` |
| `01_run_algorithms.py` | 2 | Solvers (ALNS, ILS, GLS, TS) com PCI e probing features $P_1-P_{11}$ |
| `cvrp_features.py` | 2/4/10 | Módulo independente com as 50 funções estáticas de features CVRP |
| `02_extract_features.py` | 4 | Extrai features estáticas e faz join com probing median $\to \mathbf{F}$ |
| `03_prelim.py` | 5 | Computa matriz de gaps $\mathbf{Y}$ e análise de $\varepsilon$ (Ponto de parada 5a) |
| `04_sifted.py` | 6 | Filtragem Pearson, clustering $K=10$ e seleção via Random Forest OOB $\to \mathbf{F}_{\text{sifted}}$ |
| `05_pilot.py` | 7 | Projeção linear analítica fechada e rotação $N_{\text{try}}=30 \to \mathbf{Z}$ |
| `06_overlay_checkpoints.py` | 8 | Sobreposição multi-checkpoint (25-100%) e evolução de Best Selection |
| `07_project_new_instances.py` | 10 | Reaplicação do mapa fixado para datasets externos (ex.: Uchoa Set X) |
| `literature_alignment.md` | — | Documentação com citações acadêmicas para $\varepsilon$, $K=10$ e operadores |
| `params_log.md` | 9 | Log consolidado de todos os parâmetros para a seção de metodologia do paper |
