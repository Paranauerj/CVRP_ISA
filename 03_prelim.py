"""
03_prelim.py
============
Etapa 5: PRELIM — Análise da Distribuição de Desempenho (Y) e Ponto de Decisão de Tolerância (ε)
Conforme especificação da Seção 5 do instructions.md e diretrizes do MATILDA (Smith-Miles & Muñoz, 2023).

Funções:
  1. Carrega a matriz de características F (F.parquet / F.csv).
  2. Carrega os custos medianos do checkpoint 100% (checkpoint_costs_median.csv) e os valores de BKS.
  3. Calcula a matriz de desempenho Y (gap relativo em relação ao BKS):
         Y_{i, a} = (cost_{i, a}^{100%} - BKS_i) / BKS_i
  4. Analisa a distribuição estatística de Y (individual por algoritmo e agregada do portfólio):
     - Média, desvio-padrão, mínimo, máximo
     - Percentis 10, 25, 50 (mediana), 75, 90
  5. Avalia e propõe candidatos fundamentados para a tolerância ε:
     - Candidato 1 (P25 - Estrito): Focado na excelência máxima de convergência.
     - Candidato 2 (P50 - Mediana, MATILDA Padrão): Padrão recomendado pela literatura para balanceamento 50/50.
     - Candidato 3 (P75 - Permissivo): Tolera perturbações e penaliza apenas falhas severas.
  6. Gera gráficos diagnósticos de distribuição (prelim_distribution.png):
     - Boxplot por algoritmo
     - Histograma / Densidade agregada com corte de ε
     - Perfis de Desempenho Empíricos (ECDF)
     - Taxa de sucesso por algoritmo sob os limiares candidatos
  7. Exporta Y (Y.parquet e Y.csv), o sumário estatístico (prelim_summary.json) e o relatório (prelim_report.md).
  8. Encerra na Sub-etapa 5a sem avançar para o SIFTED, permitindo inspeção e aprovação do usuário.

Parâmetros canônicos do PRELIM (fixos conforme instructions.md):
  - phi_max = False (minimização de custo/gap)
  - phi_bnd = False (sem limitação/winsorização de outliers)
  - phi_nrm = False (normalização tratada internamente no pipeline MATILDA)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Configurar stdout para UTF-8 no Windows para suportar caracteres gregos e acentos
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")  # Backend não-interativo para geração de figuras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Funções de Carregamento e BKS
# ---------------------------------------------------------------------------

COST_RE = re.compile(r"^COST\s*:?\s*([-+0-9.eE]+)", re.IGNORECASE)


def load_bks_lookup(bks_csv: Path | None = None, instances_dir: Path | None = None) -> dict[str, float]:
    """
    Carrega tabela de BKS (Best Known Solution) a partir de arquivo de validação CSV
    ou lendo diretamente os arquivos .sol / .bks do diretório de instâncias.
    """
    bks: dict[str, float] = {}

    # 1. Tentar ler do CSV de validação consolidado se existir
    if bks_csv and bks_csv.is_file():
        try:
            df_val = pd.read_csv(bks_csv)
            if "instancia_id" in df_val.columns and "bks_cost" in df_val.columns:
                for _, row in df_val.iterrows():
                    val = row["bks_cost"]
                    if pd.notna(val) and float(val) > 0:
                        bks[str(row["instancia_id"])] = float(val)
        except Exception as e:
            print(f"[AVISO] Não foi possível ler BKS de {bks_csv}: {e}")

    # 2. Complementar ou carregar varrendo o diretório de instâncias (.sol e .bks)
    if instances_dir and instances_dir.is_dir():
        for ext in (".sol", ".bks"):
            for sol_path in instances_dir.glob(f"*{ext}"):
                inst_id = sol_path.stem
                if inst_id not in bks:
                    try:
                        with sol_path.open(encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                match = COST_RE.match(line.strip())
                                if match:
                                    bks[inst_id] = float(match.group(1))
                                    break
                    except Exception:
                        pass

    return bks


def compute_performance_matrix_y(
    df_costs: pd.DataFrame,
    bks_lookup: dict[str, float],
    target_algorithms: list[str] | None = None,
) -> pd.DataFrame:
    """
    Computa a matriz Y de gaps relativos no checkpoint 100%:
        Y_{i, a} = (cost_{i, a}^{100%} - BKS_i) / BKS_i
    """
    # Filtrar checkpoint de 100%
    if "checkpoint" not in df_costs.columns:
        raise ValueError("DataFrame de custos não possui a coluna 'checkpoint'.")

    df_100 = df_costs[df_costs["checkpoint"] == 100].copy()
    if df_100.empty:
        raise ValueError("Nenhum dado encontrado para checkpoint == 100.")

    # Identificar coluna de custo
    cost_col = None
    for cand in ["cost_median_3_seeds", "cost_median", "cost"]:
        if cand in df_100.columns:
            cost_col = cand
            break
    if cost_col is None:
        raise ValueError(f"Coluna de custo não encontrada. Colunas disponíveis: {list(df_100.columns)}")

    # Filtrar algoritmos desejados se especificado
    if target_algorithms:
        df_100 = df_100[df_100["algoritmo"].isin(target_algorithms)].copy()

    # Mapear BKS
    df_100["bks"] = df_100["instancia_id"].map(bks_lookup)
    missing_bks = df_100[df_100["bks"].isna()]["instancia_id"].unique()
    if len(missing_bks) > 0:
        raise ValueError(
            f"BKS não encontrado para {len(missing_bks)} instâncias. Exemplos: {missing_bks[:5]}"
        )

    # Calcular Gap relativo
    df_100["gap"] = (df_100[cost_col] - df_100["bks"]) / df_100["bks"]

    # Pivotar para matriz [instancia_id x algoritmo]
    y_matrix = df_100.pivot(index="instancia_id", columns="algoritmo", values="gap")
    y_matrix.index.name = "instancia_id"

    return y_matrix


# ---------------------------------------------------------------------------
# Análise Estatística e Seleção de Epsilon
# ---------------------------------------------------------------------------

def compute_statistics(y_matrix: pd.DataFrame) -> dict:
    """Calcula estatísticas descritivas completas para cada algoritmo e agregadas."""
    stats = {
        "per_algorithm": {},
        "aggregate": {},
    }

    percentiles = [10, 25, 50, 75, 90]

    for col in y_matrix.columns:
        s = y_matrix[col].dropna()
        q_vals = np.percentile(s, percentiles)
        stats["per_algorithm"][col] = {
            "count": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p10": float(q_vals[0]),
            "p25": float(q_vals[1]),
            "median_p50": float(q_vals[2]),
            "p75": float(q_vals[3]),
            "p90": float(q_vals[4]),
            "max": float(s.max()),
        }

    # Agregado de todas as observações do portfólio
    all_values = y_matrix.values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    q_agg = np.percentile(all_values, percentiles)
    stats["aggregate"] = {
        "count": int(len(all_values)),
        "mean": float(np.mean(all_values)),
        "std": float(np.std(all_values)),
        "min": float(np.min(all_values)),
        "p10": float(q_agg[0]),
        "p25": float(q_agg[1]),
        "median_p50": float(q_agg[2]),
        "p75": float(q_agg[3]),
        "p90": float(q_agg[4]),
        "max": float(np.max(all_values)),
    }

    return stats


def generate_plots(
    y_matrix: pd.DataFrame,
    stats: dict,
    chosen_epsilon: float,
    output_png: Path,
) -> None:
    """Gera painel visual de 4 quadrantes para diagnóstico de Y e limiares de ε."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PRELIM — Análise da Distribuição de Desempenho (Y) e Limiares ε", fontsize=15, fontweight="bold")

    algs = list(y_matrix.columns)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # 1. Boxplot por Algoritmo (Painel Superior Esquerdo)
    ax1 = axes[0, 0]
    box_data = [y_matrix[c].dropna().values for c in algs]
    ax1.boxplot(box_data, tick_labels=algs, patch_artist=True,
                boxprops=dict(facecolor="#e6f2ff", color="#1f77b4"),
                medianprops=dict(color="#d62728", linewidth=2))
    ax1.axhline(stats["aggregate"]["p25"], color="gray", linestyle=":", label=f"P25 = {stats['aggregate']['p25']:.3f}")
    ax1.axhline(stats["aggregate"]["median_p50"], color="blue", linestyle="--", label=f"P50 (Mediana) = {stats['aggregate']['median_p50']:.3f}")
    ax1.axhline(stats["aggregate"]["p75"], color="purple", linestyle=":", label=f"P75 = {stats['aggregate']['p75']:.3f}")
    ax1.axhline(chosen_epsilon, color="red", linestyle="-", linewidth=2, label=f"ε Selecionado = {chosen_epsilon:.3f}")
    ax1.set_title("Distribuição do Gap Relativo (Y) por Algoritmo", fontweight="bold")
    ax1.set_ylabel("Gap Relativo ao BKS (Y)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8)

    # 2. Histograma Agregado com Linha de ε (Painel Superior Direito)
    ax2 = axes[0, 1]
    all_vals = y_matrix.values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    ax2.hist(all_vals, bins=25, color="#4682b4", edgecolor="black", alpha=0.7, density=True)
    ax2.axvline(chosen_epsilon, color="red", linestyle="-", linewidth=2.5,
                label=f"ε = {chosen_epsilon:.3f} ({'Padrão MATILDA' if np.isclose(chosen_epsilon, stats['aggregate']['median_p50']) else 'Custom'})")
    ax2.axvline(stats["aggregate"]["median_p50"], color="blue", linestyle="--", label="Mediana P50")
    ax2.set_title("Histograma Agregado de Y e Limiar ε", fontweight="bold")
    ax2.set_xlabel("Gap Relativo (Y)")
    ax2.set_ylabel("Densidade")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    # 3. ECDF / Perfil de Desempenho Empírico (Painel Inferior Esquerdo)
    ax3 = axes[1, 0]
    x_grid = np.linspace(0, max(all_vals.max(), 1.0), 300)
    for idx, alg in enumerate(algs):
        vals = np.sort(y_matrix[alg].dropna().values)
        ecdf = np.searchsorted(vals, x_grid, side="right") / len(vals)
        c = colors[idx % len(colors)]
        ax3.plot(x_grid, ecdf, label=alg, color=c, linewidth=2)
    ax3.axvline(chosen_epsilon, color="red", linestyle="--", alpha=0.8, label=f"ε = {chosen_epsilon:.3f}")
    ax3.set_title("Perfis de Desempenho Empíricos (ECDF)", fontweight="bold")
    ax3.set_xlabel("Tolerância de Gap (ε)")
    ax3.set_ylabel("Proporção de Instâncias Resolvidas")
    ax3.set_ylim(-0.02, 1.05)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower right", fontsize=9)

    # 4. Taxas de Sucesso por Algoritmo nos Limiares Candidatos (Painel Inferior Direito)
    ax4 = axes[1, 1]
    cands = [
        ("P25 (Estrito)", stats["aggregate"]["p25"]),
        ("P50 (Mediana)", stats["aggregate"]["median_p50"]),
        ("P75 (Permissivo)", stats["aggregate"]["p75"]),
    ]
    x = np.arange(len(algs))
    width = 0.25
    cand_colors = ["#6baed6", "#3182bd", "#08519c"]
    for i, (label, eps_val) in enumerate(cands):
        success_rates = [(y_matrix[alg] <= eps_val).mean() * 100 for alg in algs]
        ax4.bar(x + (i - 1) * width, success_rates, width, label=f"{label} ({eps_val:.2f})", color=cand_colors[i])

    ax4.set_title("Taxa de Sucesso (%) sob Candidatos de ε", fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(algs)
    ax4.set_ylabel("% Instâncias com Y <= ε")
    ax4.set_ylim(0, 105)
    ax4.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax4.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[OK] Gráficos de distribuição salvos em: {output_png}")


# ---------------------------------------------------------------------------
# CLI e Orquestração
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="03_prelim.py — Análise da Distribuição de Desempenho Y e Ponto de Decisão ε (MATILDA/ISA)"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("F.parquet"),
        help="Caminho para a matriz de características F (F.parquet ou F.csv). Padrão: F.parquet",
    )
    parser.add_argument(
        "--costs",
        type=Path,
        default=Path("checkpoint_costs_median.csv"),
        help="Caminho para o CSV de custos medianos. Padrão: checkpoint_costs_median.csv",
    )
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=Path("test_instances"),
        help="Diretório contendo as instâncias e arquivos .sol/.bks. Padrão: test_instances",
    )
    parser.add_argument(
        "--bks-file",
        type=Path,
        default=Path("outputs/input_validation.csv"),
        help="Arquivo CSV opcional contendo bks_cost por instancia_id.",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=None,
        help="Lista de algoritmos separados por vírgula (ex: ALNS,GLS,ILS). Padrão: detecta todos no CSV de custos.",
    )
    parser.add_argument(
        "--epsilon",
        type=str,
        default="auto",
        help="Tolerância ε de desempenho. 'auto' utiliza a mediana de Y (MATILDA standard), ou passe um float (ex: 0.15). Padrão: auto",
    )
    parser.add_argument(
        "--out-y-parquet",
        type=Path,
        default=Path("Y.parquet"),
        help="Arquivo Parquet de saída para a matriz Y. Padrão: Y.parquet",
    )
    parser.add_argument(
        "--out-y-csv",
        type=Path,
        default=Path("Y.csv"),
        help="Arquivo CSV de saída para a matriz Y. Padrão: Y.csv",
    )
    parser.add_argument(
        "--out-report-json",
        type=Path,
        default=Path("prelim_summary.json"),
        help="Arquivo JSON com o sumário estatístico e metadados. Padrão: prelim_summary.json",
    )
    parser.add_argument(
        "--out-report-md",
        type=Path,
        default=Path("prelim_report.md"),
        help="Relatório textual em Markdown. Padrão: prelim_report.md",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=Path("prelim_distribution.png"),
        help="Gráfico diagnóstico de distribuição. Padrão: prelim_distribution.png",
    )

    args = parser.parse_args()

    # 1. Resolver caminhos com fallback automático entre raiz e outputs/
    features_path = args.features
    if not features_path.is_file():
        alt_features = Path("outputs") / features_path.name
        if alt_features.is_file():
            features_path = alt_features
        elif Path("F.csv").is_file():
            features_path = Path("F.csv")

    costs_path = args.costs
    if not costs_path.is_file():
        alt_costs = Path("outputs") / costs_path.name
        if alt_costs.is_file():
            costs_path = alt_costs

    if not features_path.is_file():
        print(f"[ERRO] Matriz F não encontrada em {args.features} nem em {features_path}.")
        print("Execute primeiro a etapa 02: python 02_extract_features.py")
        sys.exit(1)

    if not costs_path.is_file():
        print(f"[ERRO] Arquivo de custos não encontrado em {args.costs} nem em {costs_path}.")
        sys.exit(1)

    print("=" * 70)
    print(" 03_prelim.py — PRELIM: Análise de Desempenho e Ponto de Decisão ε")
    print("=" * 70)
    print(f" Matriz F : {features_path}")
    print(f" Custos   : {costs_path}")

    # 2. Carregar F e obter instâncias válidas
    if features_path.suffix.lower() == ".parquet":
        df_f = pd.read_parquet(features_path)
    else:
        df_f = pd.read_csv(features_path, index_col=0)

    f_instances = set(df_f.index)
    print(f" Instâncias em F: {len(f_instances)} (Dimensão: {df_f.shape[0]}x{df_f.shape[1]})")

    # 3. Carregar BKS
    bks_lookup = load_bks_lookup(bks_csv=args.bks_file, instances_dir=args.instances_dir)
    print(f" BKS carregados : {len(bks_lookup)} entradas disponíveis")

    # 4. Carregar Custos e computar Y
    df_costs = pd.read_csv(costs_path)
    target_algs = [a.strip() for a in args.algorithms.split(",")] if args.algorithms else None
    y_matrix = compute_performance_matrix_y(df_costs, bks_lookup, target_algs)

    # Alinhar instâncias: interseção entre F e Y
    common_instances = sorted(list(f_instances.intersection(set(y_matrix.index))))
    if len(common_instances) == 0:
        print("[ERRO] Nenhuma instância comum entre F e os custos calculados!")
        sys.exit(1)

    if len(common_instances) < len(f_instances):
        print(f"[AVISO] Alinhando matrizes: {len(common_instances)} instâncias comuns de {len(f_instances)} em F.")

    y_matrix = y_matrix.loc[common_instances]
    algorithms = list(y_matrix.columns)
    print(f" Matriz Y : {y_matrix.shape[0]} instâncias x {y_matrix.shape[1]} algoritmos ({algorithms})")

    # 5. Calcular estatísticas de distribuição
    stats = compute_statistics(y_matrix)

    # 6. Avaliar candidatos de ε
    p25 = stats["aggregate"]["p25"]
    p50 = stats["aggregate"]["median_p50"]
    p75 = stats["aggregate"]["p75"]

    print("\n" + "-" * 70)
    print(" DISTRIBUIÇÃO ESTATÍSTICA DO GAP RELATIVO (Y)")
    print("-" * 70)
    print(f"{'Algoritmo':<10} | {'Média':>7} | {'Std':>7} | {'Min':>7} | {'P25':>7} | {'P50 (Med)':>9} | {'P75':>7} | {'P90':>7} | {'Max':>7}")
    print("-" * 70)
    for alg in algorithms:
        st = stats["per_algorithm"][alg]
        print(f"{alg:<10} | {st['mean']:>7.4f} | {st['std']:>7.4f} | {st['min']:>7.4f} | {st['p25']:>7.4f} | {st['median_p50']:>9.4f} | {st['p75']:>7.4f} | {st['p90']:>7.4f} | {st['max']:>7.4f}")
    print("-" * 70)
    st_agg = stats["aggregate"]
    print(f"{'AGREGADO':<10} | {st_agg['mean']:>7.4f} | {st_agg['std']:>7.4f} | {st_agg['min']:>7.4f} | {st_agg['p25']:>7.4f} | {st_agg['median_p50']:>9.4f} | {st_agg['p75']:>7.4f} | {st_agg['p90']:>7.4f} | {st_agg['max']:>7.4f}")
    print("-" * 70)

    # 7. Decisão de ε
    candidates = {
        "Candidato 1 (P25 - Estrito)": {
            "valor": p25,
            "descricao": "Focado na alta excelência de convergência (top 25% melhores gaps). Muito seletivo.",
        },
        "Candidato 2 (P50 - Mediana, MATILDA Padrão)": {
            "valor": p50,
            "descricao": "Padrão recomendado pela literatura MATILDA (Smith-Miles & Muñoz, 2023). Divide as instâncias em 50% boas e 50% desafiadoras, maximizando a variância e a separabilidade para aprendizado de máquina.",
        },
        "Candidato 3 (P75 - Permissivo)": {
            "valor": p75,
            "descricao": "Tolera perturbações moderadas de clima/recurso; penaliza apenas instâncias com falha severa (> P75).",
        },
    }

    if args.epsilon.lower() == "auto":
        chosen_epsilon = p50
        epsilon_mode = "auto (P50 Mediana - Literatura MATILDA)"
    else:
        try:
            chosen_epsilon = float(args.epsilon)
            epsilon_mode = f"manual ({chosen_epsilon})"
        except ValueError:
            print(f"[ERRO] Valor inválido para --epsilon: {args.epsilon}. Use 'auto' ou um número decimal.")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(" PROPOSTA DE VALORES CANDIDATOS PARA ε (TOLERÂNCIA DE DESEMPENHO)")
    print("=" * 70)
    for name, cinfo in candidates.items():
        rec_tag = " [PADRÃO RECOMENDADO]" if "Mediana" in name else ""
        print(f" * {name}{rec_tag}: ε = {cinfo['valor']:.4f} ({cinfo['valor']*100:.2f}%)")
        print(f"   Justificativa: {cinfo['descricao']}\n")

    print(f" >> ε Decidido para esta execução: {chosen_epsilon:.4f} (Modo: {epsilon_mode})")

    # Avaliar taxa de sucesso sob ε decidido
    success_rates = {}
    print(f"\n Taxa de Sucesso por Algoritmo (Y <= {chosen_epsilon:.4f}):")
    for alg in algorithms:
        rate = float((y_matrix[alg] <= chosen_epsilon).mean())
        success_rates[alg] = rate
        print(f"   - {alg:<6}: {rate*100:>5.1f}% das instâncias qualificadas como 'boas'")

    # 8. Exportar Matriz Y (Parquet e CSV)
    args.out_y_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.out_y_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_report_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_report_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_plot.parent.mkdir(parents=True, exist_ok=True)

    y_matrix.to_parquet(args.out_y_parquet)
    y_matrix.to_csv(args.out_y_csv)
    print(f"\n[OK] Matriz Y gravada em Parquet: {args.out_y_parquet}")
    print(f"[OK] Matriz Y gravada em CSV    : {args.out_y_csv}")

    # 9. Gerar Gráficos Diagnósticos
    generate_plots(y_matrix, stats, chosen_epsilon, args.out_plot)

    # 10. Salvar Sumário JSON com parâmetros canônicos do PRELIM
    summary_data = {
        "n_instances": len(common_instances),
        "algorithms": algorithms,
        "statistics": stats,
        "epsilon_candidates": {
            "P25": p25,
            "P50_median": p50,
            "P75": p75,
        },
        "chosen_epsilon": chosen_epsilon,
        "epsilon_mode": epsilon_mode,
        "success_rates_at_chosen_epsilon": success_rates,
        "matilda_parameters": {
            "phi_max": False,
            "phi_bnd": False,
            "phi_nrm": False,
        },
    }
    with args.out_report_json.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[OK] Sumário PRELIM salvo em JSON: {args.out_report_json}")

    # 11. Salvar Relatório Markdown
    md_content = f"""# PRELIM — Relatório de Distribuição de Desempenho e Seleção de ε

## 1. Contexto Metodológico
Conforme a metodologia de Instance Space Analysis (ISA) de Smith-Miles & Muñoz (2023) e Gouvêa et al. (2025), o desempenho relativo de cada algoritmo $a$ sobre a instância $i$ é medido pelo gap normalizado em relação ao Best Known Solution (BKS):
$$Y_{{i, a}} = \\frac{{\\text{{custo}}_{{i, a}}^{{100\\%}} - BKS_i}}{{BKS_i}}$$

## 2. Parâmetros Canônicos do PRELIM
- $\\phi_{{max}}$ = False (problema de minimização)
- $\\phi_{{bnd}}$ = False (sem winsorização/corte de outliers)
- $\\phi_{{nrm}}$ = False (normalização das features executada internamente no toolkit)

## 3. Distribuição Estatística de Y
- **Total de instâncias avaliadas:** {len(common_instances)}
- **Algoritmos no portfólio:** {", ".join(algorithms)}

| Algoritmo | Média | Desvio-Padrão | Mínimo | P25 | Mediana (P50) | P75 | P90 | Máximo |
|---|---|---|---|---|---|---|---|---|
"""
    for alg in algorithms:
        st = stats["per_algorithm"][alg]
        md_content += f"| **{alg}** | {st['mean']:.4f} | {st['std']:.4f} | {st['min']:.4f} | {st['p25']:.4f} | {st['median_p50']:.4f} | {st['p75']:.4f} | {st['p90']:.4f} | {st['max']:.4f} |\n"

    st_agg = stats["aggregate"]
    md_content += f"| **AGREGADO** | **{st_agg['mean']:.4f}** | **{st_agg['std']:.4f}** | **{st_agg['min']:.4f}** | **{st_agg['p25']:.4f}** | **{st_agg['median_p50']:.4f}** | **{st_agg['p75']:.4f}** | **{st_agg['p90']:.4f}** | **{st_agg['max']:.4f}** |\n\n"

    md_content += f"""## 4. Candidatos a Tolerância (ε)
1. **Candidato 1 (P25 - Estrito):** $\\varepsilon = {p25:.4f}$ ({p25*100:.2f}%)
   - *Justificativa:* Seleciona apenas os 25% melhores desempenhos. Exige convergência quase exata.
2. **Candidato 2 (P50 - Mediana, MATILDA Padrão - RECOMENDADO):** $\\varepsilon = {p50:.4f}$ ({p50*100:.2f}%)
   - *Justificativa:* Padrão da literatura (Smith-Miles & Muñoz, 2023). Divide as instâncias balanceadamente em 50% boas e 50% desafiadoras, garantindo máxima variância e estabilidade para treino do Random Forest no SIFTED.
3. **Candidato 3 (P75 - Permissivo):** $\\varepsilon = {p75:.4f}$ ({p75*100:.2f}%)
   - *Justificativa:* Tolera dificuldades climáticas e penaliza apenas falhas severas de otimização.

## 5. Limiar Adotado
- **ε Selecionado:** `{chosen_epsilon:.4f}` ({chosen_epsilon*100:.2f}%)
- **Modo:** `{epsilon_mode}`
- **Taxas de Sucesso ($Y \\le \\varepsilon$):**
"""
    for alg in algorithms:
        md_content += f"  - **{alg}:** {success_rates[alg]*100:.1f}%\n"

    md_content += f"""
## 6. Próximo Passo
A Sub-etapa 5a do PRELIM encerra aqui conforme `instructions.md`.
Para prosseguir para o SIFTED (Etapa 6), execute `04_sifted.py` passando `--epsilon {chosen_epsilon:.4f}`.
"""
    with args.out_report_md.open("w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] Relatório Markdown salvo em: {args.out_report_md}")

    print("\n" + "=" * 70)
    print(" Sub-etapa 5a CONCLUÍDA com sucesso!")
    print(f" Arquivo Y gerado: {args.out_y_parquet}")
    print(f" Limiar ε definido: {chosen_epsilon:.4f}")
    print(" Conforme instructions.md, o pipeline para aqui para inspeção antes do SIFTED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
