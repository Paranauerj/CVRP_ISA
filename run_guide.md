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

Com os defaults do plano, o comando e:

```powershell
python .\01_run_algorithms.py `
  --instances-dir "C:\Users\jptin\OneDrive\USAL\Codes\CVRPBenchmark\instances\gaetano" `
  --algorithms ALNS ILS GLS `
  --seeds 1001 2001 3001 `
  --workers 8 `
  --seconds-per-customer 0.5 `
  --time-cap 100.0 `
  --output .\outputs\checkpoint_costs_raw.csv `
  --median-output .\outputs\checkpoint_costs_median.csv
```

O runner aceita `--workers` e usa processos independentes. Para esta campanha, o valor recomendado e `--workers 8`.

## 7. Saidas

Com 10.000 instancias, 3 algoritmos e 3 seeds:

- `checkpoint_costs_raw.csv`: aproximadamente 360.000 linhas, pois sao 90.000 rodadas x 4 checkpoints;
- `checkpoint_costs_median.csv`: aproximadamente 120.000 linhas, pois sao 10.000 instancias x 3 algoritmos x 4 checkpoints.

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

## 9. Estado atual e bloqueios antes da campanha final

O runner atual e util para smoke tests, mas ainda nao e o executor final do estudo:

1. A funcao `alns()` atual e um ALNS manual simplificado; ela ainda nao usa a biblioteca oficial `alns`.
2. O ILS atual usa relocacao aleatoria e 2-opt. Ainda faltam savings, Or-opt, perturbacao definida e o traco P1-P11 do plano.
3. O GLS converte custos float para inteiros antes de entrega-los ao OR-Tools, enquanto ALNS/ILS usam floats. A funcao objetivo precisa ser unificada.
4. A frota do GLS esta modelada como ilimitada, usando um veiculo por cliente. Isso deve ser confirmado contra a definicao experimental do CVRP.
5. A campanha completa ainda nao gera `Y` nem `ils_trace.csv`; esses artefatos pertencem as etapas seguintes.

Portanto, a ordem segura e:

```text
validacao -> smoke de 1 instancia -> amostra de 10/100 -> corrigir bloqueios -> campanha completa
```

## 10. Referencia rapida

- Validacao: `00_validate_inputs.py`
- Runner atual: `01_run_algorithms.py`
- Decisoes tecnicas: `isa_implementation_notes.md`
- Plano original: `instructions.md` (nao editar durante esta etapa)
