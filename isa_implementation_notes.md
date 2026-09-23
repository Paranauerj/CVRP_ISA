# ISA CVRP - Notas de Implementacao (itens 1 e 2)

## Dados confirmados

- Fonte: `C:\Users\jptin\OneDrive\USAL\Codes\CVRPBenchmark\instances\gaetano`.
- Contagem confirmada: 10.000 arquivos `.vrp` e 10.000 arquivos `.sol`.
- Nao ha arquivos `.bks` nessa pasta. O `.sol` e o BKS efetivo: cada par contem rotas e uma linha `Cost`.
- O nome-base e a chave de correspondencia 1:1.
- `DIMENSION` inclui o deposito; numero de clientes = `DIMENSION - 1`.
- O validador desta pasta e `00_validate_inputs.py`; ele aceita `.bks` se forem fornecidos em outra fonte e, na ausencia deles, usa `.sol`.

## Ambiente

Sim, a implementacao e Python. O ambiente de referencia ja usa OR-Tools 9.14, `vrplib`, NumPy, pandas, PyArrow e HGS. Para ALNS, a biblioteca recomendada e `alns` 7.0.0 (MIT, Python >= 3.9), com operadores destroy/repair definidos pelo projeto. A biblioteca oferece selecao de operadores, criterios de aceitacao e parada, mas nao fornece a modelagem CVRP pronta.

Para o ILS, nao sera usada uma biblioteca generica: um modulo proprio permite manter exatamente PCI, savings, 2-opt, Or-opt e o traco P1-P11. O pacote PyPI chamado `ils` nao e necessario para este estudo.

Dependencia adicional planejada:

```text
alns==7.0.0
```

## Itens 1 e 2: contrato de execucao

1. Validar entradas antes de qualquer rodada:
   - correspondencia 1:1 entre `.vrp` e BKS;
   - leitura de `DIMENSION`, `CAPACITY` e `Cost` do BKS;
   - relatorio CSV com status por instancia;
   - distribuicao de clientes e erros.

2. Rodar os tres algoritmos com a mesma construtiva PCI:
   - ALNS: `alns` com operadores CVRP do projeto;
   - ILS: implementacao propria e instrumentada;
   - GLS: OR-Tools, `PARALLEL_CHEAPEST_INSERTION` e `GUIDED_LOCAL_SEARCH`.

Parametros configuraveis, com defaults do plano:

```text
algorithms = ALNS, ILS, GLS
runs = 3
seeds = 1001, 2001, 3001
checkpoint_fractions = 0.25, 0.50, 0.75, 1.00
seconds_per_customer = 0.5
time_cap_seconds = 100.0
construction = PCI
workers = 8
```

O tempo da instancia e `min(seconds_per_customer * customers, time_cap_seconds)`. O resultado de cada algoritmo/seed deve registrar o melhor custo incumbente em cada checkpoint. Depois, a tabela agregada calcula a mediana das 3 rodadas.

## Coleta ao longo do tempo

- ILS/ALNS: o loop proprio chama um registrador monotonicamente por tempo sempre que uma nova melhor solucao aparece; no checkpoint, congela-se o ultimo melhor custo conhecido. Isso e suficiente para a tabela de Y e para o traco do ILS.
- GLS/OR-Tools: usar o callback de nova solucao (`AddAtSolutionCallback`) para atualizar o melhor custo e o tempo decorrido; apos o `SolveWithParameters`, preencher checkpoints sem evento com o ultimo incumbente.
- O callback registra somente o essencial: `elapsed_seconds`, `best_cost` e, no ILS, dados do movimento para P1-P11. Nao copiar toda a infraestrutura de monitoramento do CVRPBenchmark.
- O custo final usado em Y e o custo mediano no checkpoint de 100%, comparado com `BKS` por `(ost - BKS) / BKS`.

## Execucao paralela

O runner aceita `--workers` e usa `ProcessPoolExecutor`. Cada processo executa uma combinacao independente de instancia, algoritmo e seed. Somente o processo principal escreve os CSVs, evitando escrita concorrente no mesmo arquivo.

Benchmark observado em 8 instancias, ILS, uma seed, 0,02 s/cliente e cap de 2 s:

| Workers | Tempo | Speedup | Eficiencia |
|---:|---:|---:|---:|
| 1 | 16,586 s | 1,00x | 100,0% |
| 2 | 8,942 s | 1,85x | 92,7% |
| 4 | 4,975 s | 3,33x | 83,3% |
| 8 | 3,435 s | 4,83x | 60,4% |

Os tempos incluem inicializacao dos processos e escrita dos resultados. O speedup nao e linear; `--workers 8` e o ponto de partida recomendado para Gaetano.

## Saidas dos itens 1 e 2

- `outputs/input_validation.csv`: uma linha por instancia.
- `outputs/checkpoint_costs_raw.csv`: uma linha por instancia, algoritmo, seed e checkpoint.
- `outputs/checkpoint_costs_median.csv`: mediana das 3 seeds.
- `outputs/ils_trace.csv`: eventos essenciais do ILS para probing features.

O script de validacao nao executa algoritmos. O runner da etapa 2 deve ser executado explicitamente pelo autor, inicialmente em uma amostra pequena para conferir custos, tempos e callbacks antes das 10.000 instancias.

## Referencias consultadas

- ALNS: https://pypi.org/project/alns/ (versao 7.0.0; documentacao e exemplo CVRP).
- Repositorio: https://github.com/N-Wouda/ALNS.
- OR-Tools Routing: https://developers.google.com/optimization/routing.
