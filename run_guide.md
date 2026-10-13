# Guia de Execução — Instance Space Analysis (ISA) para CVRP

Este documento orienta a execução completa do pipeline de Instance Space Analysis (ISA) para o problema de Roteamento de Veículos Capacitado (CVRP), integrando os 4 algoritmos do portfólio canônico (**ALNS**, **ILS**, **GLS**, **TS**), a extração de 50 features estruturais e 11 probing features, a projeção linear analítica PILOT ($Z_1, Z_2$), e o overlay multi-checkpoint (25%, 50%, 75%, 100%) com delimitação de footprints e seleção de algoritmos (AAS).

---

## 1. Ambientes de Execução e Diretórios de Instâncias

O benchmark utiliza o conjunto com 10.000 instâncias do Gaetano (`.vrp` pareados com arquivos `.sol` que contêm a linha `Cost` do BKS).

### Execução Local (Windows)
```text
C:\Users\jptin\OneDrive\USAL\Codes\CVRPBenchmark\instances\gaetano
```

### Execução Remota via Servidor Linux (SSH)
Conforme configurado para execução remota no cluster/servidor SSH:
```bash
--instances-dir "../../joao/instances/gaetano"
```

---

## 2. Consumo de Memória RAM e Perfil Computacional

### A execução gasta muita memória RAM?
**Não.** O pipeline é expressivamente leve em memória RAM, sendo estritamente **CPU-bound** (limitado por ciclos de processamento dos solvers e da busca local):

1. **Pegada por Processo Worker:**
   - Runtime Python + bibliotecas (`numpy`, `scipy`, `vrplib`): ~60 MB a 80 MB.
   - Instâncias CVRP ($N \le 200$): a matriz de distâncias $200 \times 200$ em ponto flutuante ocupa menos de **320 KB** em memória.
   - Motores C++ do OR-Tools e PyVRP: ~40 MB a 70 MB por processo durante a exploração de vizinhanças.
   - **Total estimado por worker:** **100 MB a 150 MB de RAM**.

2. **Consumo Total com Paralelismo:**
   - Com 8 workers (`--workers 8`): **~1,0 GB a 1,4 GB de RAM**.
   - Com 16 workers (`--workers 16`): **~2,0 GB a 2,5 GB de RAM**.
   - Com 32 workers (`--workers 32`): **~4,0 GB a 5,0 GB de RAM**.

3. **Arquivos de Saída e CSVs:**
   - A tabela bruta completa de 10.000 instâncias x 4 algoritmos x 3 seeds x 4 checkpoints totaliza ~480.000 linhas.
   - Todas as 480.000 linhas em memória ocupam aproximadamente **150 MB**.
   - A escrita em disco é realizada via *flush* atômico contínuo a cada job concluído, evitando acúmulo de buffers.

> [!NOTE]
> Um servidor modesto com 8 GB ou 16 GB de RAM opera com folga confortável mesmo alocando 16 ou 32 threads de processamento. O gargalo do pipeline é tempo de CPU, não memória.

---

## 3. Sistema Resiliente de Retomada (`--resume`)

O benchmark massivo de algoritmos (Etapa 1) pode demandar dias de execução contínua. Por esse motivo, o script [`01_run_algorithms.py`](file:///C:/Users/jptin/OneDrive/USAL/Codes/CVRP_ISA/01_run_algorithms.py) e o orquestrador [`run_pipeline.py`](file:///C:/Users/jptin/OneDrive/USAL/Codes/CVRP_ISA/run_pipeline.py) possuem mecanismo de retomada tolerante a falhas (ativado por padrão):

### Como funciona a resiliência no `01_run_algorithms.py`
1. **Verificação de Integridade de Jobs:**
   - Cada job consiste em uma combinação `(instancia_id, algoritmo, seed)` que precisa emitir **4 checkpoints** (25%, 50%, 75%, 100%).
   - Ao iniciar com `--resume`, o script analisa os arquivos `checkpoint_costs_raw.csv` e `ils_trace.csv`.
   - Se o processo foi interrompido abruptamente (SIGINT, queda de energia, queda de conexão SSH) enquanto gravava um job pela metade (ex.: apenas 2 checkpoints gravados), essas **linhas órfãs são automaticamente expurgadas** do arquivo raw, e o job incompleto é reiniciado do zero.
   - Jobs que já possuem todos os 4 checkpoints válidos são classificados como concluídos e pulados instantaneamente.
2. **Escrita Atômica Contínua:**
   - A cada término de job por qualquer worker, os dados são descarregados imediatamente no arquivo de saída com `.flush()`.
3. **Features:**
   - A extração de características em [`02_extract_features.py`](file:///C:/Users/jptin/OneDrive/USAL/Codes/CVRP_ISA/02_extract_features.py) não mantém estado intermediário e executa do zero em poucos segundos/minutos sobre o conjunto final de instâncias, garantindo pureza dos dados.

Para forçar um re-run completo do zero descartando arquivos anteriores, use a flag `--no-resume`.

---

## 4. Onde Encontrar a Acurácia de Algorithm Selection (AAS) e Modelos ML?

Ao término da execução do pipeline, os modelos de Machine Learning (SVM com kernel RBF e Random Forest com 50 estimadores) são avaliados em validação cruzada estratificada (5-fold CV).

Os resultados de acurácia são consolidados em três locais:

### 1. Arquivo Estruturado JSON
- Caminho: [`outputs/<run_name>/ml_models_accuracy.json`](file:///C:/Users/jptin/OneDrive/USAL/Codes/CVRP_ISA/outputs/clean_verification_run/ml_models_accuracy.json)
- Contém:
  - `cv_method`: Método de validação cruzada e número de dobras.
  - `epsilon`: Limiar $\varepsilon$ de Good Performance adotado.
  - `footprint_models`: Acurácia 2D do SVM em $(Z_1, Z_2)$, acurácia 2D do RF em $(Z_1, Z_2)$ e acurácia do RF nas 10 features selecionadas ($F_{\text{sifted}}$) para cada um dos 4 algoritmos.
  - `best_selection_model`: Distribuição de classes do melhor algoritmo e acurácias do classificador multi-classe (SVM 2D, RF 2D e RF $F_{\text{sifted}}$).

### 2. Relatório Executivo Markdown
- Caminho: [`outputs/<run_name>/pipeline_summary.md`](file:///C:/Users/jptin/OneDrive/USAL/Codes/CVRP_ISA/outputs/clean_verification_run/pipeline_summary.md)
- Seção 2: Apresenta a tabela formal com as métricas de acurácia de previsão de footprint e a acurácia de seleção do solver vencedor.

### 3. Saída Padrão no Terminal (stdout)
- O orquestrador imprime ao final o resumo formatado:
  ```text
  --- 2. Seleção do Melhor Algoritmo (Best Solver Selection / AAS) ---
   Distribuição de Vencedores: {'ALNS': 3, 'ILS': 1, 'GLS': 1}
     * SVM RBF em (Z1, Z2)       : 80.0% acurácia
     * Random Forest em (Z1, Z2) : 80.0% acurácia
     * Random Forest em F_sifted : 100.0% acurácia
  ```

---

## 5. Fundamentação dos Polígonos de Footprint na Literatura de ISA

### O que são esses polígonos?
Na literatura canônica de Instance Space Analysis (*Smith-Miles & Muñoz, 2023*; *Notice et al., 2025*; *Gouvêa et al., 2025*):

1. **Espaço Discreto vs. Espaço Contínuo:**
   - As instâncias empíricas testadas são pontos discretos no plano $(Z_1, Z_2)$.
   - Porém, o conceito teórico de **Footprint** ($\mathcal{F}$) define uma sub-região contínua (um envelope de dominância) dentro do espaço bidimensional onde determinado algoritmo apresenta desempenho superior (*Best Solver*) ou satisfatório (*Good Performance*, com $\text{gap} \le \varepsilon$).
2. **Construção Geométrica via Convex Hull:**
   - Para estimar essa região contínua no plano 2D, calcula-se o **Convex Hull** (envoltório convexo bidimensional) que circunscreve todas as instâncias onde aquele algoritmo obteve o melhor resultado.
   - A área desse polígono relativa à área do Convex Hull global de todas as instâncias define a métrica formal **Normalized Area ($A$)** apresentada na Tabela 3 de Notice et al. (2025).
3. **Por que gerar "com polígonos" e "sem polígonos"?**
   - **Com Polígonos** (`checkpoint_overlays_best_selection.png`): Mostra as fronteiras territoriais contínuas de cada heurística, permitindo visualizar como o nicho de dominância do ALNS, ILS, GLS ou TS se expande ou retrai temporalmente nos 4 checkpoints.
   - **Sem Polígonos** (`checkpoint_overlays_best_selection_points.png`): Exibe unicamente a dispersão dos pontos individuais coloridos por solver e com rótulos de instância, eliminando qualquer sobreposição visual ou ilusão de dominância em regiões esparsas.

---

## 6. Execução do Pipeline Ponta a Ponta

Para executar o pipeline completo de uma só vez com todos os passos encadeados e saídas centralizadas:

### Em Ambiente Local (PowerShell):
```powershell
python .\run_pipeline.py `
  --run-name "campanha_piloto" `
  --instances-dir "C:\Users\jptin\OneDrive\USAL\Codes\CVRPBenchmark\instances\gaetano" `
  --workers 8 `
  --resume
```

### Em Servidor Remoto Linux via SSH (Bash):
```bash
python3 run_pipeline.py \
  --run-name "campanha_gaetano_ssh" \
  --instances-dir "../../joao/instances/gaetano" \
  --workers 16 \
  --resume
```

---

## 7. Execução Manual Etapa por Etapa (Para Depuração ou SSH Direto)

Se preferir rodar cada script individualmente:

### Etapa 0 — Validação de Entradas
```bash
python3 00_validate_inputs.py \
  --instances-dir "../../joao/instances/gaetano" \
  --report outputs/campanha_ssh/input_validation.csv
```

### Etapa 1 — Benchmark dos 4 Algoritmos (ALNS, ILS, GLS, TS) com Probing
```bash
python3 01_run_algorithms.py \
  --instances-dir "../../joao/instances/gaetano" \
  --algorithms ALNS ILS GLS TS \
  --seeds 1001 2001 3001 \
  --workers 16 \
  --seconds-per-customer 0.5 \
  --time-cap 100.0 \
  --resume \
  --output outputs/campanha_ssh/checkpoint_costs_raw.csv \
  --median-output outputs/campanha_ssh/checkpoint_costs_median.csv \
  --ils-trace-output outputs/campanha_ssh/ils_trace.csv \
  --ils-median-output outputs/campanha_ssh/ils_probing_median.csv
```

### Etapa 2 — Extração das Features CVRP e Fusão com Probing
```bash
python3 02_extract_features.py \
  --instances-dir "../../joao/instances/gaetano" \
  --probing-file outputs/campanha_ssh/ils_probing_median.csv \
  --workers 16 \
  --output outputs/campanha_ssh/F.parquet \
  --csv-output outputs/campanha_ssh/F.csv
```

### Etapa 3 — PRELIM (Matriz de Gaps Y e Ponto de Decisão de ε)
```bash
python3 03_prelim.py \
  --features outputs/campanha_ssh/F.parquet \
  --costs outputs/campanha_ssh/checkpoint_costs_median.csv \
  --instances-dir "../../joao/instances/gaetano" \
  --bks-file outputs/campanha_ssh/input_validation.csv \
  --epsilon auto \
  --out-y-parquet outputs/campanha_ssh/Y.parquet \
  --out-y-csv outputs/campanha_ssh/Y.csv \
  --out-report-json outputs/campanha_ssh/prelim_summary.json \
  --out-report-md outputs/campanha_ssh/prelim_report.md \
  --out-plot outputs/campanha_ssh/prelim_distribution.png
```

### Etapa 4 — SIFTED (Seleção de K=10 Features via Clustering e Random Forest OOB)
```bash
python3 04_sifted.py \
  --features outputs/campanha_ssh/F.parquet \
  --performance outputs/campanha_ssh/Y.parquet \
  --k-clusters 10 \
  --out-f-sifted-parquet outputs/campanha_ssh/F_sifted.parquet \
  --out-f-sifted-csv outputs/campanha_ssh/F_sifted.csv \
  --out-selected-json outputs/campanha_ssh/sifted_features.json \
  --out-selected-csv outputs/campanha_ssh/sifted_features.csv \
  --out-report-md outputs/campanha_ssh/sifted_report.md \
  --out-plot outputs/campanha_ssh/sifted_clusters.png
```

### Etapa 5 — PILOT (Projeção Linear Analítica Z1, Z2 e Rotação N_try=30)
```bash
python3 05_pilot.py \
  --features outputs/campanha_ssh/F_sifted.parquet \
  --performance outputs/campanha_ssh/Y.parquet \
  --n-try 30 \
  --out-coords-parquet outputs/campanha_ssh/projected_coords.parquet \
  --out-coords-csv outputs/campanha_ssh/projected_coords.csv \
  --out-z-json outputs/campanha_ssh/Z_matrix.json \
  --out-z-npy outputs/campanha_ssh/Z_matrix.npy \
  --out-report-md outputs/campanha_ssh/pilot_report.md \
  --out-plot outputs/campanha_ssh/instance_space_pilot.png \
  --out-feature-maps outputs/campanha_ssh/feature_distribution_maps.png
```

### Etapa 6 — Overlay Multi-Checkpoint e Métricas de Footprint
```bash
python3 06_overlay_checkpoints.py \
  --coords outputs/campanha_ssh/projected_coords.parquet \
  --costs outputs/campanha_ssh/checkpoint_costs_median.csv \
  --instances-dir "../../joao/instances/gaetano" \
  --bks-file outputs/campanha_ssh/input_validation.csv \
  --epsilon 0.05 \
  --out-checkpoints-parquet outputs/campanha_ssh/Y_checkpoints.parquet \
  --out-checkpoints-csv outputs/campanha_ssh/Y_checkpoints.csv \
  --out-footprint-metrics outputs/campanha_ssh/footprint_metrics.csv \
  --out-plot-algorithms outputs/campanha_ssh/checkpoint_overlays_algorithms.png \
  --out-plot-best outputs/campanha_ssh/checkpoint_overlays_best_selection.png \
  --out-plot-best-points outputs/campanha_ssh/checkpoint_overlays_best_selection_points.png \
  --out-plot-summary outputs/campanha_ssh/checkpoint_evolution_summary.png \
  --out-report-md outputs/campanha_ssh/checkpoint_overlay_report.md \
  --out-params-log outputs/campanha_ssh/params_log.md
```

---

## 8. Catálogo Completo de Saídas por Diretório de Execução

Ao rodar uma campanha, todos os artefatos são gravados dentro de `outputs/<run_name>/`:

| Arquivo | Categoria | Descrição |
|---|---|---|
| `input_validation.csv` | Validação | Registro 1:1 de integridade das instâncias e BKS |
| `checkpoint_costs_raw.csv` | Benchmark | Custos das 4 marcas temporais para todas as seeds |
| `checkpoint_costs_median.csv` | Benchmark | Custos consolidados pela mediana de 3 seeds |
| `ils_trace.csv` | Probing | Trajetória bruta de 11 métricas de landscape do ILS |
| `ils_probing_median.csv` | Probing | Mediana das 11 métricas de probing por instância |
| `F.parquet` / `F.csv` | Features | Matriz completa de 63 características (50 estáticas + 13 probing) |
| `Y.parquet` / `Y.csv` | Performance | Matriz de gaps relativos a 100% de tempo vs BKS |
| `prelim_distribution.png` | Diagnóstico | Curvas de densidade, boxplot e ECDF de gaps |
| `prelim_report.md` | Relatório | Análise do percentil $\varepsilon$ canônico do MATILDA |
| `F_sifted.parquet` / `.csv` | Seleção | Matriz com as 10 características ortogonais selecionadas |
| `sifted_clusters.png` | Diagnóstico | Dendrograma e correlação dos 10 clusters |
| `sifted_report.md` | Relatório | Justificativa e relevância de cada feature selecionada |
| `Z_matrix.json` / `Z_matrix.npy` | Projeção | Matriz linear de projeção analítica $B$ e pesos |
| `projected_coords.parquet` | Coordenadas | Coordenadas canônicas $(Z_1, Z_2)$ das instâncias |
| `instance_space_pilot.png` | Visualização | Mapa bidimensional do Instance Space com biplot loadings |
| `feature_distribution_maps.png` | Visualização | Mapas de calor das top features no plano Z (Notice Fig. 5) |
| `Y_checkpoints.parquet` / `.csv` | Checkpoints | Gaps consolidados nos 4 checkpoints temporais |
| `footprint_metrics.csv` | Métricas ISA | Tabela 3 (Notice et al., 2025): Area, Density e Purity por solver |
| `checkpoint_overlays_algorithms.png` | Visualização | Grade comparativa de redução de gaps por algoritmo |
| `checkpoint_overlays_best_selection.png` | Visualização | Mapa de Best Selection com envoltórios de Footprint (Convex Hull) |
| `checkpoint_overlays_best_selection_points.png` | Visualização | Mapa de Best Selection sem polígonos (dispersão pura de pontos) |
| `checkpoint_evolution_summary.png` | Visualização | Dashboard executivo com curvas de convergência |
| `checkpoint_overlay_report.md` | Relatório | Transição do melhor algoritmo e análise metodológica |
| `ml_models_accuracy.json` | ML / AAS | Acurácias de validação cruzada dos modelos SVM e Random Forest |
| `pipeline_summary.md` | Relatório | Sumário executivo consolidando tempos, métricas de ML e catálogo |
| `params_log.md` | Metodologia | Checklist formal com todos os hiperparâmetros para o paper |
