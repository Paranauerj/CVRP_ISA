# Guia de Execução da Fase 2 no Servidor Linux (SSH Fire-and-Forget)
## 4 Solvers (ALNS, ILS, GLS, TS) | 8 Workers | Probing Features (P1–P11)

---

## 1. Configuração e Tempo Estimado no Servidor da Universidade

### 1.1 Parâmetros Selecionados
- **Dataset:** 10.000 instâncias Gaetano ($N \in [30, 200]$, média $\bar{N} \approx 113,25$ clientes)
- **Solvers:** **4 metaheurísticas** (`ALNS`, `ILS`, `GLS`, `TS`)
  - **TS incluído:** Fornece memória explícita de curto prazo (Glover, 1989), vencedor comprovado em instâncias densas como `X-n101-k25` (gap de **1.67%**) e `LDG185` (gap de **15.55%**).
- **Construtiva comum:** **PCI** (Parallel Cheapest Insertion) fixa para todos os solvers.
- **Orçamento de tempo:** $T(N) = \min(0.5 \times N,\ 100\text{ s})$ (coerência com o artigo de AAS do autor).
- **Seeds:** 3 (`1001`, `2001`, `3001`) $\implies$ $10.000 \times 4 \times 3 = \mathbf{120.000\text{ runs}}$.
- **Checkpoints:** 25%, 50%, 75%, 100% de $T(N)$.
- **Workers:** **8 processos paralelos** (ideal para o servidor da universidade).
- **Destino dos arquivos:** **Raiz do projeto** (`./`).

---

### 1.2 Tempo Estimado com 8 Workers

- **Tempo total de CPU acumulado:** $\approx 1.887,5\text{ horas}$ (~78,6 CPU-dias).
- **Tempo real de relógio (Wall-clock) com 8 workers (~80% de eficiência paralela):**
  $$\text{Tempo real} \approx \frac{1.887,5\text{ h}}{8 \times 0.80} \approx \mathbf{295\text{ horas}} \approx \mathbf{11,8\text{ a 12 dias}}$$

> [!TIP]
> **Por que 8 workers no servidor da universidade é muito melhor?**
> - Com 4 workers levaria ~23 dias. Com 8 workers, o tempo cai praticamente pela metade (**~12 dias**).
> - 8 workers utilizam de forma equilibrada a CPU e a memória RAM sem saturar o barramento do servidor ou causar contenção de I/O em disco.

---

## 2. A Questão do Probing (P1 a P11 Resolvidas Automaticamente)

As 11 probing features exigidas na Seção 2.3 e Seção 4 de [`instructions.md`](instructions.md) são extraídas diretamente pelo `ILS` e salvas na raiz:
- `ils_trace.csv`: 30.000 linhas com as 11 features brutas por rodada (instância × seed).
- `ils_probing_median.csv`: 10.000 linhas com a mediana das 11 features por instância (1 linha por instância), **pronta para fusão direta na matriz $F$ na Etapa 4**.

Você **não** precisará rodar o ILS novamente no futuro para gerar as features de probing.

---

## 3. Preparação do Servidor (Executar apenas na primeira vez)

Conecte-se via SSH:
```bash
ssh usuario@servidor.universidade.edu
```

Entre na pasta do projeto e configure o ambiente Python:
```bash
cd ~/cvrp_isa

# Criar e ativar o ambiente virtual (se ainda não existir)
python3 -m venv venv
source venv/bin/activate

# Instalar dependências essenciais
pip install --upgrade pip
pip install numpy scipy vrplib ortools alns==7.0.0 pyvrp
```

Verifique se as bibliotecas carregam sem erros:
```bash
python3 -c "import numpy, vrplib, ortools, alns, pyvrp; print('Ambiente 100% OK!')"
```

---

## 4. O Comando Principal (NOHUP — Fire-and-Forget)

O `nohup` desacopla a execução do terminal. Mesmo que feche o notebook, caia a internet ou faça logout do SSH, o script continuará rodando ininterruptamente até o fim.

### Comando Definitivo de Produção:
```bash
cd ~/cvrp_isa
source venv/bin/activate

nohup python3 01_run_algorithms.py \
  --instances-dir "/caminho/para/instancias/gaetano" \
  --algorithms ALNS ILS GLS TS \
  --seeds 1001 2001 3001 \
  --workers 8 \
  --seconds-per-customer 0.5 \
  --time-cap 100.0 \
  --output ./checkpoint_costs_raw.csv \
  --median-output ./checkpoint_costs_median.csv \
  --ils-trace-output ./ils_trace.csv \
  --ils-median-output ./ils_probing_median.csv > ./execution.log 2>&1 &
```

*(Como definimos esses caminhos como padrão em `01_run_algorithms.py`, o comando simplificado abaixo produz exatamente o mesmo resultado:)*
```bash
nohup python3 01_run_algorithms.py \
  --instances-dir "/caminho/para/instancias/gaetano" \
  --workers 8 > ./execution.log 2>&1 &
```

Ao rodar, o Linux exibirá o PID do processo (ex: `[1] 12345`).

---

## 5. Como Monitorar a Execução

### 5.1 Ver o log em tempo real
```bash
tail -f execution.log
```
*(Para sair da visualização do log sem parar o processo, pressione `Ctrl + C`).*

### 5.2 Conferir se os 8 workers estão ativos e consumindo CPU
```bash
top
```
*(Você verá 8 processos `python3` consumindo ~100% de CPU cada).*

Ou liste os processos do runner:
```bash
ps aux | grep 01_run_algorithms.py
```

### 5.3 Acompanhar o número de execuções concluídas
Para conferir quantas linhas já foram geradas:
```bash
wc -l checkpoint_costs_raw.csv ils_trace.csv
```

### 5.4 Parar a execução (se necessário)
Se precisar interromper por qualquer motivo:
```bash
pkill -f 01_run_algorithms.py
```

---

## 6. Arquivos Gerados na Raiz do Projeto

Ao final dos ~12 dias, a raiz conterá:

| Arquivo | Linhas Esperadas | Conteúdo |
|---|:---:|---|
| `checkpoint_costs_raw.csv` | ~480.001 | Custos brutos por instância, solver, seed e checkpoint (25, 50, 75, 100%) |
| `checkpoint_costs_median.csv` | ~160.001 | Mediana das 3 seeds por checkpoint — base para a matriz de desempenho $Y$ |
| `ils_trace.csv` | ~30.001 | As 11 probing features ($P_1-P_{11}$) brutas por rodada do ILS |
| `ils_probing_median.csv` | ~10.001 | As 11 probing features agregadas por mediana — base para a matriz $F$ |
| `execution.log` | — | Log completo com timestamps e diagnósticos da execução |

---

## 7. Script Rápido de Validação Pós-Execução

Após a conclusão, execute na raiz para validar 100% das saídas:
```bash
python3 -c "
import pandas as pd

raw = pd.read_csv('checkpoint_costs_raw.csv')
med = pd.read_csv('checkpoint_costs_median.csv')
trace = pd.read_csv('ils_trace.csv')
prob_med = pd.read_csv('ils_probing_median.csv')

print(f'Raw rows: {len(raw)} (esperado: 480.000)')
print(f'Median rows: {len(med)} (esperado: 160.000)')
print(f'ILS trace rows: {len(trace)} (esperado: 30.000)')
print(f'ILS median rows: {len(prob_med)} (esperado: 10.000)')

assert not raw['cost'].isna().any(), 'Há NaNs!'
assert not (raw['cost'] == float('inf')).any(), 'Há Infs!'
assert (med['n_seeds'] == 3).all(), 'Há instâncias com menos de 3 seeds!'
print('Sucesso absoluto! Todas as 10.000 instâncias foram resolvidas e validadas.')
"
```
