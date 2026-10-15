"""
06_overlay_checkpoints.py
=========================
Etapa 8: Overlay dos Checkpoints no Espaço Fixo de Instâncias (Z1, Z2)
Conforme especificação da Seção 8 do instructions.md.

Objetivo Científico:
  Mapear e visualizar no espaço fixo (Z1, Z2) construído pelo PILOT como a região de domínio
  e o desempenho de cada algoritmo evolui à medida que o orçamento computacional avança:
    Checkpoints: 25% -> 50% -> 75% -> 100% de T(N)
  Isso conecta diretamente o Instance Space Analysis (ISA) ao problema de Dynamic Algorithm Selection (AAS).

Entradas:
  - projected_coords.parquet (ou .csv): Coordenadas fixas (Z1, Z2) da Etapa 7.
  - checkpoint_costs_median.csv: Custos medianos dos 4 checkpoints.
  - inputs/BKS: Carregados de outputs/input_validation.csv ou test_instances/*.sol.

Saídas:
  - Y_checkpoints.parquet / Y_checkpoints.csv: Tabela consolidada com gaps por checkpoint e coordenadas.
  - checkpoint_overlays_algorithms.png: Painel comparativo contínuo por algoritmo e checkpoint.
  - checkpoint_overlays_best_selection.png: Evolução temporal do mapa de 'Best Selection' (Footprint Dinâmico).
  - checkpoint_evolution_summary.png: Painel consolidado para publicação científica.
  - checkpoint_overlay_report.md: Relatório analítico da evolução temporal dos algoritmos.
  - params_log.md: Checklist consolidado de todos os parâmetros metodológicos do estudo (Seção 9 e 11).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Configurar stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# Funções de Carregamento e Processamento de Gaps
# ---------------------------------------------------------------------------

COST_RE = re.compile(r"^COST\s*:?\s*([-+0-9.eE]+)", re.IGNORECASE)


def load_bks(bks_csv: Path | None, instances_dir: Path | None) -> dict[str, float]:
    """Carrega BKS consolidado."""
    bks: dict[str, float] = {}
    if bks_csv and bks_csv.is_file():
        try:
            df = pd.read_csv(bks_csv)
            if "instancia_id" in df.columns and "bks_cost" in df.columns:
                for _, r in df.iterrows():
                    if pd.notna(r["bks_cost"]) and float(r["bks_cost"]) > 0:
                        bks[str(r["instancia_id"])] = float(r["bks_cost"])
        except Exception:
            pass

    if instances_dir and instances_dir.is_dir():
        for ext in (".sol", ".bks"):
            for p in instances_dir.glob(f"*{ext}"):
                if p.stem not in bks:
                    try:
                        with p.open(encoding="utf-8", errors="ignore") as f:
                            for l in f:
                                m = COST_RE.match(l.strip())
                                if m:
                                    bks[p.stem] = float(m.group(1))
                                    break
                    except Exception:
                        pass
    return bks


def compute_checkpoint_gaps(
    df_costs: pd.DataFrame,
    df_coords: pd.DataFrame,
    bks_lookup: dict[str, float],
    target_algorithms: list[str] | None = None,
) -> tuple[pd.DataFrame, list[int], list[str]]:
    """
    Computa os gaps relativos por checkpoint e alinha com as coordenadas Z1, Z2.
    """
    cost_col = None
    for cand in ["cost_median_3_seeds", "cost_median", "cost"]:
        if cand in df_costs.columns:
            cost_col = cand
            break
    if cost_col is None:
        raise ValueError("Coluna de custo mediano não encontrada.")

    # Filtrar instâncias com coordenadas projetadas
    df = df_costs[df_costs["instancia_id"].isin(df_coords.index)].copy()

    if target_algorithms:
        df = df[df["algoritmo"].isin(target_algorithms)].copy()

    # Mapear BKS e calcular gap
    df["bks"] = df["instancia_id"].map(bks_lookup)
    df["gap"] = (df[cost_col] - df["bks"]) / df["bks"]

    checkpoints = sorted(df["checkpoint"].unique())
    algorithms = sorted(df["algoritmo"].unique())

    # Construir tabela agregada formatada
    records = []
    for inst_id, inst_df in df.groupby("instancia_id"):
        z1 = df_coords.loc[inst_id, "Z1"]
        z2 = df_coords.loc[inst_id, "Z2"]
        bks_val = bks_lookup.get(str(inst_id), np.nan)

        for cp in checkpoints:
            cp_df = inst_df[inst_df["checkpoint"] == cp]
            row = {
                "instancia_id": inst_id,
                "checkpoint": cp,
                "Z1": z1,
                "Z2": z2,
                "bks": bks_val,
            }
            gaps_dict = {}
            for alg in algorithms:
                alg_match = cp_df[cp_df["algoritmo"] == alg]
                if not alg_match.empty:
                    gap_val = float(alg_match["gap"].values[0])
                    cost_val = float(alg_match[cost_col].values[0])
                    row[f"{alg}_cost"] = cost_val
                    row[f"{alg}_gap"] = gap_val
                    if not np.isnan(gap_val):
                        gaps_dict[alg] = gap_val
                else:
                    row[f"{alg}_cost"] = np.nan
                    row[f"{alg}_gap"] = np.nan

            if gaps_dict:
                best_gap = min(gaps_dict.values())
                # Empate numérico com tolerância de 1e-6 (evita viés posicional de argmin para o 1º algoritmo)
                tied_algs = [alg for alg, g in gaps_dict.items() if abs(g - best_gap) <= 1e-6]
                if len(tied_algs) > 1:
                    row["best_algo"] = "Tie"
                else:
                    row["best_algo"] = tied_algs[0]
                row["best_gap"] = best_gap
            else:
                row["best_algo"] = None
                row["best_gap"] = np.nan
            records.append(row)

    df_consolidated = pd.DataFrame(records)
    return df_consolidated, checkpoints, algorithms


# ---------------------------------------------------------------------------
# Visualizações Científicas
# ---------------------------------------------------------------------------

def plot_algorithm_gap_overlays(
    df_consolidated: pd.DataFrame,
    checkpoints: list[int],
    algorithms: list[str],
    output_png: Path,
) -> None:
    """
    Gera grade de dispersão contínua: Linhas = Algoritmos, Colunas = Checkpoints (25%, 50%, 75%, 100%).
    Utiliza escala de cores idêntica normalizada para evidenciar a redução temporal dos gaps.
    """
    n_rows = len(algorithms)
    n_cols = len(checkpoints)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 1, 3.5 * n_rows), squeeze=False)
    fig.suptitle("Overlay de Gaps no Espaço Fixo (Z1, Z2) por Checkpoint e Algoritmo", fontsize=15, fontweight="bold")

    # Determinar limites globais de gap para normalização uniforme da barra de cores
    all_gaps = [df_consolidated[f"{alg}_gap"].dropna().values for alg in algorithms]
    flattened_gaps = np.concatenate(all_gaps)
    v_min = max(0.0, float(np.percentile(flattened_gaps, 1)))
    v_max = float(np.percentile(flattened_gaps, 99))

    norm = matplotlib.colors.Normalize(vmin=v_min, vmax=v_max)
    cmap = plt.get_cmap("viridis")

    for r, alg in enumerate(algorithms):
        for c, cp in enumerate(checkpoints):
            ax = axes[r, c]
            sub = df_consolidated[df_consolidated["checkpoint"] == cp]

            z1 = sub["Z1"].values
            z2 = sub["Z2"].values
            gaps = sub[f"{alg}_gap"].values

            sc = ax.scatter(z1, z2, c=gaps, cmap=cmap, norm=norm, s=90, edgecolor="black", alpha=0.9, zorder=3)

            # Rótulos das instâncias nos subplots (apenas para bases pequenas <= 12)
            if len(sub) <= 12:
                for _, row in sub.iterrows():
                    short_name = str(row["instancia_id"]).split("_")[0] + "_" + str(row["instancia_id"]).split("_")[-1]
                    ax.annotate(short_name, (row["Z1"], row["Z2"]), fontsize=6, alpha=0.75,
                                xytext=(3, 3), textcoords="offset points")

            ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
            ax.axvline(0, color="gray", linestyle=":", alpha=0.3)
            ax.grid(True, linestyle=":", alpha=0.4)

            # Títulos das colunas apenas na primeira linha
            if r == 0:
                ax.set_title(f"Checkpoint {cp}% de T(N)", fontweight="bold", fontsize=11)
            # Rótulo do algoritmo apenas na primeira coluna
            if c == 0:
                ax.set_ylabel(f"{alg}\nCoordenada Z2", fontweight="bold", fontsize=11)
            if r == n_rows - 1:
                ax.set_xlabel("Coordenada Z1", fontweight="bold", fontsize=10)

    # Barra de cores compartilhada à direita
    fig.subplots_adjust(right=0.88, top=0.92, hspace=0.25, wspace=0.2)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Gap Relativo ao BKS (Escala Normalizada)", fontweight="bold", fontsize=11)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[OK] Painel de gaps por algoritmo salvo em: {output_png}")


def filter_footprint_outliers(pts: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:
    """
    Filtra outliers espaciais isolados no plano (Z1, Z2) antes do Convex Hull,
    conforme a metodologia de densidade e pureza de Notice et al. (2025).
    Mantém apenas pontos que possuem vizinhança de suporte da mesma classe.
    """
    if len(pts) < 4:
        return pts
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(pts))
    dists_sorted = np.sort(dists, axis=1)
    k_idx = min(min_cluster_size, len(pts) - 1)
    k_dist = dists_sorted[:, k_idx]
    med_k = float(np.median(k_dist))
    if med_k <= 1e-6:
        return pts
    threshold = med_k * 2.5
    inlier_mask = k_dist <= threshold
    if np.sum(inlier_mask) >= 3:
        return pts[inlier_mask]
    return pts


def plot_best_selection_evolution(
    df_consolidated: pd.DataFrame,
    checkpoints: list[int],
    algorithms: list[str],
    output_png: Path,
    show_polygons: bool = True,
) -> None:
    """
    Gera o mapa dinâmico de 'Best Selection' (Footprint Temporal do Vencedor) nos 4 checkpoints.
    Equivalente à Fig. 3 do Notice e à Fig. 4 do Gouvêa et al. (2025).
    Se show_polygons=True, desenha os envoltórios convexos (Convex Hull) representando o Footprint contínuo.
    Se show_polygons=False, exibe dispersão pura de pontos sem polígonos sobrepostos.
    """
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(4.5 * len(checkpoints), 4.8), sharex=True, sharey=True)
    title_suffix = "com Footprints Poligonais" if show_polygons else "(Dispersão Pura de Pontos)"
    fig.suptitle(f"Evolução Temporal do Melhor Algoritmo ('Best Selection') {title_suffix}",
                 fontsize=13, fontweight="bold")

    algo_palette = {
        "ALNS": "#1f77b4",  # Azul
        "GLS":  "#ff7f0e",  # Laranja
        "ILS":  "#2ca02c",  # Verde
        "TS":   "#d62728",  # Vermelho
        "Tie":  "#7f7f7f",  # Cinza neutro para empates múltiplos
    }

    for c, cp in enumerate(checkpoints):
        ax = axes[c]
        sub = df_consolidated[df_consolidated["checkpoint"] == cp]

        all_cats = list(algorithms) + (["Tie"] if "Tie" in sub["best_algo"].values else [])
        for alg in all_cats:
            mask = (sub["best_algo"] == alg)
            if np.any(mask):
                c_color = algo_palette.get(alg, "#7f7f7f")
                pts = sub.loc[mask, ["Z1", "Z2"]].values
                pts_hull = filter_footprint_outliers(pts)
                if show_polygons and alg != "Tie" and len(pts_hull) >= 3:
                    try:
                        hull = ConvexHull(pts_hull)
                        poly = MplPolygon(
                            pts_hull[hull.vertices],
                            facecolor=c_color,
                            edgecolor=c_color,
                            alpha=0.18,
                            linestyle="--",
                            linewidth=1.5,
                            zorder=2,
                        )
                        ax.add_patch(poly)
                    except Exception:
                        pass
                ax.scatter(sub.loc[mask, "Z1"], sub.loc[mask, "Z2"],
                           color=c_color, label=f"{alg} ({mask.sum()} inst)",
                           s=120, edgecolor="black", zorder=3)

        if len(sub) <= 12:
            for _, row in sub.iterrows():
                short_name = str(row["instancia_id"]).split("_")[0] + "_" + str(row["instancia_id"]).split("_")[-1]
                ax.annotate(short_name, (row["Z1"], row["Z2"]), fontsize=6.5, alpha=0.8,
                            xytext=(3, 3), textcoords="offset points")

        ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(f"Checkpoint {cp}% do Tempo", fontweight="bold", fontsize=11)
        ax.set_xlabel("Coordenada Z1", fontweight="bold")
        if c == 0:
            ax.set_ylabel("Coordenada Z2", fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[OK] Painel de 'Best Selection' ({'com polígonos' if show_polygons else 'sem polígonos'}) salvo em: {output_png}")


def plot_summary_dashboard(
    df_consolidated: pd.DataFrame,
    checkpoints: list[int],
    algorithms: list[str],
    output_png: Path,
) -> None:
    """
    Gera painel consolidado de 4 quadrantes para artigo científico:
      1. Curvas de convergência média dos algoritmos por checkpoint
      2. Evolução das vitórias de cada algoritmo (% de instâncias dominadas)
      3. Best Selection final a 100%
      4. Dispersão da melhora relativa (Delta Gap de 25% para 100%)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Dinâmica Temporal da Busca Local no Espaço de Instâncias (Checkpoints 25% a 100%)",
                 fontsize=14, fontweight="bold")

    algo_palette = {"ALNS": "#1f77b4", "GLS": "#ff7f0e", "ILS": "#2ca02c", "TS": "#d62728", "Tie": "#7f7f7f"}

    # 1. Convergência Média por Checkpoint (Superior Esquerdo)
    ax1 = axes[0, 0]
    for alg in algorithms:
        mean_gaps = [df_consolidated[df_consolidated["checkpoint"] == cp][f"{alg}_gap"].mean() * 100 for cp in checkpoints]
        c = algo_palette.get(alg, "gray")
        ax1.plot(checkpoints, mean_gaps, marker="o", linewidth=2.2, label=alg, color=c)
    ax1.set_title("Evolução do Gap Médio (%) ao Longo dos Checkpoints", fontweight="bold")
    ax1.set_xlabel("Checkpoint de Tempo (%)", fontweight="bold")
    ax1.set_ylabel("Gap Médio ao BKS (%)", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # 2. Participação do Vencedor (% de Vitórias) por Checkpoint (Superior Direito)
    ax2 = axes[0, 1]
    n_insts = len(df_consolidated["instancia_id"].unique())
    x = np.arange(len(checkpoints))
    all_summary_cats = list(algorithms) + (["Tie"] if "Tie" in df_consolidated["best_algo"].values else [])
    width = 0.8 / len(all_summary_cats)
    for i, alg in enumerate(all_summary_cats):
        counts = [(df_consolidated[df_consolidated["checkpoint"] == cp]["best_algo"] == alg).sum() / n_insts * 100 for cp in checkpoints]
        c = algo_palette.get(alg, "gray")
        ax2.bar(x + (i - len(all_summary_cats) / 2 + 0.5) * width, counts, width, label=alg, color=c, edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{cp}%" for cp in checkpoints], fontweight="bold")
    ax2.set_title("Proporção de Instâncias Dominadas por Checkpoint", fontweight="bold")
    ax2.set_ylabel("% de Vitórias (Menor Gap)", fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend(loc="upper right", fontsize=8)

    # 3. Best Selection Final a 100% (Inferior Esquerdo)
    ax3 = axes[1, 0]
    sub100 = df_consolidated[df_consolidated["checkpoint"] == 100]
    sub100_cats = list(algorithms) + (["Tie"] if "Tie" in sub100["best_algo"].values else [])
    for alg in sub100_cats:
        mask = (sub100["best_algo"] == alg)
        if np.any(mask):
            c = algo_palette.get(alg, "gray")
            ax3.scatter(sub100.loc[mask, "Z1"], sub100.loc[mask, "Z2"],
                        color=c, label=f"{alg} ({mask.sum()} inst)", s=120, edgecolor="black", zorder=3)
    if len(sub100) <= 12:
        for _, row in sub100.iterrows():
            short_name = str(row["instancia_id"]).split("_")[0] + "_" + str(row["instancia_id"]).split("_")[-1]
            ax3.annotate(short_name, (row["Z1"], row["Z2"]), fontsize=6.5, alpha=0.8,
                         xytext=(3, 3), textcoords="offset points")
    ax3.set_title("Espaço Final de Instâncias (100% de Tempo) — Best Selection", fontweight="bold")
    ax3.set_xlabel("Coordenada Z1", fontweight="bold")
    ax3.set_ylabel("Coordenada Z2", fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="upper left", fontsize=9)

    # 4. Magnitude da Melhoria Obtida de 25% para 100% (Inferior Direito)
    ax4 = axes[1, 1]
    sub25 = df_consolidated[df_consolidated["checkpoint"] == 25].set_index("instancia_id")
    sub100_idx = sub100.set_index("instancia_id")
    delta_gap = (sub25["best_gap"] - sub100_idx["best_gap"]) * 100

    sc4 = ax4.scatter(sub100["Z1"], sub100["Z2"], c=delta_gap.values, cmap="plasma", s=110, edgecolor="black", zorder=3)
    cbar4 = fig.colorbar(sc4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.set_label("Redução no Gap (% entre 25% e 100%)", fontweight="bold")

    if len(sub100) <= 12:
        for _, row in sub100.iterrows():
            short_name = str(row["instancia_id"]).split("_")[0] + "_" + str(row["instancia_id"]).split("_")[-1]
            ax4.annotate(short_name, (row["Z1"], row["Z2"]), fontsize=6.5, alpha=0.8,
                         xytext=(3, 3), textcoords="offset points")
    ax4.set_title("Melhoria de Qualidade ao Estender o Tempo (Delta Gap)", fontweight="bold")
    ax4.set_xlabel("Coordenada Z1", fontweight="bold")
    ax4.set_ylabel("Coordenada Z2", fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[OK] Painel consolidado salvo em: {output_png}")


# ---------------------------------------------------------------------------
# Geração do params_log.md (Checklist da Seção 9 e 11)
# ---------------------------------------------------------------------------

def generate_params_log(
    output_md: Path,
    algorithms: list[str],
    checkpoints: list[int],
    n_instances: int,
) -> None:
    """Gera o arquivo params_log.md detalhando todos os parâmetros canônicos do estudo."""
    # Ler metadados do PRELIM e SIFTED se existirem
    eps_val = "0.1717 (auto / Mediana P50 de Y)"
    prelim_candidates = [output_md.parent / "prelim_summary.json", Path("prelim_summary.json"), Path("outputs/prelim_summary.json")]
    for p_cand in prelim_candidates:
        if p_cand.is_file():
            try:
                with open(p_cand, encoding="utf-8") as f:
                    pj = json.load(f)
                    eps_val = f"{pj.get('chosen_epsilon', 0.1717):.4f} ({pj.get('epsilon_mode', 'auto')})"
                break
            except Exception:
                pass

    content = f"""# PARAMS LOG — Especificação Consolidada dos Parâmetros do Estudo
*Documento gerado automaticamente para servir como rascunho direto da seção de Metodologia do artigo científico.*

---

## 1. Algoritmos e Busca Local (Etapa 2)

| Parâmetro | Configuração Adotada | Fundamentação Acadêmica |
|---|---|---|
| **Portfólio de Solvers** | {", ".join(algorithms)} | Cobertura de meta-heurísticas clássicas de trajetória (Seções 1 a 5 de `literature_alignment.md`) |
| **Heurística Construtiva Comum** | **Parallel Cheapest Insertion (PCI)** | Garante que diferenças de convergência decorram da busca local e não da solução inicial |
| **ALNS** | Adaptive Large Neighborhood Search | Ropke & Pisinger (2006); operadores Shaw, Random e SISR com Simulated Annealing canônico |
| **ILS** | Iterated Local Search | Lourenço, Martin & Stützle (2003); descida estrita com escape por perturbação dinâmica |
| **GLS** | Guided Local Search | Voudouris & Tsang (1999); penalização contínua adaptativa via OR-Tools |
| **TS** | Tabu Search | Glover (1989); memória de curto prazo com tenure configurável via OR-Tools |
| **Polishing C++** | PyVRP Local Search | SwapStar (Vidal, COR 2022), Relocate, 2-Opt, Exchange |
| **Orçamento Temporal** | $T(N) = \\min(0.5N, 100\\text{{s}})$ | Escalonamento suave proporcional ao número de clientes |
| **Repetições Estocásticas** | 3 sementes aleatórias por instância | Mediana de custos das 3 rodadas |
| **Checkpoints Temporais** | {", ".join([f"{cp}%" for cp in checkpoints])} de $T(N)$ | Rastreamento da evolução da busca ao longo do tempo |

---

## 2. Extração de Características e Probing (Etapa 4)

- **Total de Características Extraídas:** 63 features brutas
- **Invariância de Escala e Grid Canônico (Notice et al., 2025, p. 4):**
  - Coordenadas espaciais normalizadas para grid $[0, 1000] \\times [0, 1000]$ preservando a proporção de aspecto espacial.
  - Todas as métricas extensivas (ND2, MST1_sum, MST3, NN2, NN3) convertidas em taxas e médias relativas a N.
- **Famílias Estruturais:**
  1. **ND (Distribuição de Nós):** 9 sub-features (distâncias médias, medianas, clusters DBSCAN)
  2. **MST (Árvore Geradora Mínima):** 3 sub-features (grau de nós, custos de arestas)
  3. **G (Geometria):** 5 sub-features (área do retângulo envolvente, fecho convexo)
  4. **NN (k-NN Graph):** 8 sub-features (componentes conexos, ângulos de vizinhança)
  5. **VRP (Específicas de Roteamento):** 5 sub-features (demandas, capacidade, depósito)
  6. **P (ILS Probing):** 11 features extraídas em tempo real (<2ms overhead) do traço do ILS

---

## 3. PRELIM — Tolerância de Desempenho (Etapa 5)

- **Métrica de Desempenho ($Y$):** Gap relativo ao Best Known Solution no checkpoint de 100%:
  $$Y_{{i, a}} = \\frac{{\\text{{custo}}_{{i, a}}^{{100\\%}} - BKS_i}}{{BKS_i}}$$
- **Tolerância $\\varepsilon$:** `{eps_val}`
- **Fundamentação:** Padrão canônico do framework MATILDA (**Smith-Miles & Muñoz, 2023**; **Muñoz et al., 2018**). A adoção da mediana garante classes balanceadas 50/50 e máxima separabilidade para aprendizado de máquina.
- **Parâmetros Canônicos Fixos:**
  - $\\phi_{{\\max}} = \\text{{False}}$ (problema de minimização)
  - $\\phi_{{\\text{{bnd}}}} = \\text{{False}}$ (sem winsorização/corte de outliers)
  - $\\phi_{{\\text{{nrm}}}} = \\text{{False}}$ (normalização tratada internamente)

---

## 4. SIFTED — Seleção Inteligente de Features (Etapa 6)

- **Filtro Linear:** Pearson $|r| \\ge 0.50$ com pelo menos um algoritmo
- **Controle de Multicolinearidade:** $K$-Means sobre dissimilaridade $D = 1 - |r|$
- **Número de Clusters ($K$):** **$K = 10$, fixo** (padrão deliberado do MATILDA; evita overfitting de métricas como Silhouette)
- **Critério de Seleção Final:** Projeção temporária 2D via PCA + Random Forest regressora (100 árvores) por algoritmo minimizando o erro Out-of-Bag (OOB MSE)
- **Features Finais Selecionadas:** 10 características representando todas as 6 famílias do CVRP

---

## 5. PILOT — Projeção Linear no Espaço 2D (Etapa 7)

- **Solver:** Analítico em Forma Fechada ($\\phi_{{\\text{{num}}}} = \\text{{False}}$) baseado em decomposição espectral generalizada
- **Ponderação $\\alpha$:** $0.5$ (equilíbrio exato entre preservação de features e alinhamento de performance)
- **Otimização de Rotação ($N_{{\\text{{try}}}}$):** $30$ avaliações angulares para orientar a dificuldade no quadrante superior
- **Equação Definitiva de Projeção:** Salva em `Z_matrix.json` e `Z_matrix.npy` (Eq. 1 de Gouvêa et al., 2025)

---

## 6. Overlay dos Checkpoints (Etapa 8)

- **Espaço $(Z_1, Z_2)$:** Rigorosamente fixo (a projeção não é recalculada entre checkpoints)
- **Visualizações Geradas:**
  - `checkpoint_overlays_algorithms.png` (gaps contínuos por solver)
  - `checkpoint_overlays_best_selection.png` (footprint temporal do vencedor)
  - `checkpoint_evolution_summary.png` (painel consolidado)
"""
    with output_md.open("w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Log metodológico consolidado salvo em: {output_md}")


def compute_footprint_metrics(
    df_consolidated: pd.DataFrame,
    checkpoints: list[int],
    algorithms: list[str],
    epsilon: float = 0.05,
) -> pd.DataFrame:
    """
    Calcula as métricas de Footprint canônicas para Good e Best performance:
      - Normalized Area (A): Área do polígono / Área do Convex Hull do espaço total.
      - Density (D): Proporção de instâncias dentro do polígono normalizada pela densidade do espaço.
      - Purity (P): Proporção de instâncias com desempenho alvo contidas no polígono.
    Alinhado estritamente à Tabela 3 de Notice et al. (2025) e Seção 3.4 de Smith-Miles & Muñoz (2023).
    """
    df_unique = df_consolidated.drop_duplicates(subset=["instancia_id"]).copy()
    Z_all = df_unique[["Z1", "Z2"]].values
    n_total = len(Z_all)

    if n_total >= 3:
        try:
            hull_space = ConvexHull(Z_all)
            area_space = float(hull_space.volume)
        except Exception:
            area_space = 1.0
    else:
        area_space = 1.0
    density_space = n_total / area_space if area_space > 0 else 1.0

    records = []
    for cp in checkpoints:
        sub_cp = df_consolidated[df_consolidated["checkpoint"] == cp]
        for alg in algorithms:
            # 1. Good Performance
            good_mask = sub_cp[f"{alg}_gap"] <= epsilon
            Z_good = sub_cp.loc[good_mask, ["Z1", "Z2"]].values
            good_ids = set(sub_cp.loc[good_mask, "instancia_id"])

            Z_good_hull = filter_footprint_outliers(Z_good)
            if len(Z_good_hull) >= 3:
                try:
                    hull_good = ConvexHull(Z_good_hull)
                    area_good = float(hull_good.volume)
                    norm_area_good = area_good / area_space
                    path_good = MplPath(Z_good_hull[hull_good.vertices])
                    inside_mask = path_good.contains_points(Z_all)
                    n_inside_total = int(np.sum(inside_mask))
                    inside_good = sum(
                        1
                        for _, row in df_unique.iterrows()
                        if row["instancia_id"] in good_ids and path_good.contains_point((row["Z1"], row["Z2"]))
                    )
                    purity_good = inside_good / n_inside_total if n_inside_total > 0 else 1.0
                    density_good = (n_inside_total / area_good) / density_space if area_good > 0 else 0.0
                except Exception:
                    norm_area_good = 0.0
                    density_good = len(Z_good) / n_total
                    purity_good = 1.0
            elif len(Z_good) > 0:
                norm_area_good = 0.0
                density_good = len(Z_good) / n_total
                purity_good = 1.0
            else:
                norm_area_good = 0.0
                density_good = 0.0
                purity_good = 0.0

            # 2. Best Performance
            best_mask = sub_cp["best_algo"] == alg
            Z_best = sub_cp.loc[best_mask, ["Z1", "Z2"]].values
            best_ids = set(sub_cp.loc[best_mask, "instancia_id"])

            Z_best_hull = filter_footprint_outliers(Z_best)
            if len(Z_best_hull) >= 3:
                try:
                    hull_best = ConvexHull(Z_best_hull)
                    area_best = float(hull_best.volume)
                    norm_area_best = area_best / area_space
                    path_best = MplPath(Z_best_hull[hull_best.vertices])
                    inside_mask = path_best.contains_points(Z_all)
                    n_inside_total = int(np.sum(inside_mask))
                    inside_best = sum(
                        1
                        for _, row in df_unique.iterrows()
                        if row["instancia_id"] in best_ids and path_best.contains_point((row["Z1"], row["Z2"]))
                    )
                    purity_best = inside_best / n_inside_total if n_inside_total > 0 else 1.0
                    density_best = (n_inside_total / area_best) / density_space if area_best > 0 else 0.0
                except Exception:
                    norm_area_best = 0.0
                    density_best = len(Z_best) / n_total
                    purity_best = 1.0
            elif len(Z_best) > 0:
                norm_area_best = 0.0
                density_best = len(Z_best) / n_total
                purity_best = 1.0
            else:
                norm_area_best = 0.0
                density_best = 0.0
                purity_best = 0.0

            records.append({
                "checkpoint": cp,
                "algoritmo": alg,
                "good_area": round(norm_area_good, 3),
                "good_density": round(density_good, 3),
                "good_purity": round(purity_good, 3),
                "best_area": round(norm_area_best, 3),
                "best_density": round(density_best, 3),
                "best_purity": round(purity_best, 3),
                "n_good": int(np.sum(good_mask)),
                "n_best": int(np.sum(best_mask)),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI e Execução Principal
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="06_overlay_checkpoints.py — Etapa 8: Overlay dos Checkpoints no Espaço Fixo (Z1, Z2)"
    )
    parser.add_argument(
        "--coords",
        type=Path,
        default=Path("projected_coords.parquet"),
        help="Arquivo com as coordenadas fixas (Z1, Z2). Padrão: projected_coords.parquet",
    )
    parser.add_argument(
        "--costs",
        type=Path,
        default=Path("checkpoint_costs_median.csv"),
        help="Arquivo de custos dos checkpoints. Padrão: checkpoint_costs_median.csv",
    )
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=Path("test_instances"),
        help="Diretório contendo instâncias e arquivos .sol. Padrão: test_instances",
    )
    parser.add_argument(
        "--bks-file",
        type=Path,
        default=Path("outputs/input_validation.csv"),
        help="Arquivo CSV de validação contendo bks_cost.",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=None,
        help="Lista de algoritmos separados por vírgula. Padrão: detecta todos disponíveis.",
    )
    parser.add_argument(
        "--out-checkpoints-parquet",
        type=Path,
        default=Path("Y_checkpoints.parquet"),
        help="Arquivo Parquet consolidado com gaps por checkpoint. Padrão: Y_checkpoints.parquet",
    )
    parser.add_argument(
        "--out-checkpoints-csv",
        type=Path,
        default=Path("Y_checkpoints.csv"),
        help="Arquivo CSV consolidado com gaps por checkpoint. Padrão: Y_checkpoints.csv",
    )
    parser.add_argument(
        "--out-plot-algorithms",
        type=Path,
        default=Path("checkpoint_overlays_algorithms.png"),
        help="Gráfico de dispersão de gaps por algoritmo. Padrão: checkpoint_overlays_algorithms.png",
    )
    parser.add_argument(
        "--out-plot-best",
        type=Path,
        default=Path("checkpoint_overlays_best_selection.png"),
        help="Gráfico de Best Selection por checkpoint com polígonos de footprint. Padrão: checkpoint_overlays_best_selection.png",
    )
    parser.add_argument(
        "--out-plot-best-points",
        type=Path,
        default=Path("checkpoint_overlays_best_selection_points.png"),
        help="Gráfico de Best Selection por checkpoint sem polígonos (apenas dispersão). Padrão: checkpoint_overlays_best_selection_points.png",
    )
    parser.add_argument(
        "--out-plot-summary",
        type=Path,
        default=Path("checkpoint_evolution_summary.png"),
        help="Painel consolidado para publicação. Padrão: checkpoint_evolution_summary.png",
    )
    parser.add_argument(
        "--out-report-md",
        type=Path,
        default=Path("checkpoint_overlay_report.md"),
        help="Relatório técnico em Markdown. Padrão: checkpoint_overlay_report.md",
    )
    parser.add_argument(
        "--out-params-log",
        type=Path,
        default=Path("params_log.md"),
        help="Checklist e log completo dos parâmetros metodológicos. Padrão: params_log.md",
    )
    parser.add_argument(
        "--out-footprint-metrics",
        type=Path,
        default=Path("footprint_metrics.csv"),
        help="Arquivo CSV para as métricas de Footprint (Area, Density, Purity). Padrão: footprint_metrics.csv",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Tolerância de gap para Good Performance no Footprint (padrão: 0.05 = 5%%)",
    )

    args = parser.parse_args()

    # 1. Resolver caminhos com fallback automático
    coords_path = args.coords
    if not coords_path.is_file():
        alt_c = Path("outputs") / coords_path.name
        if alt_c.is_file():
            coords_path = alt_c
        elif Path("projected_coords.csv").is_file():
            coords_path = Path("projected_coords.csv")

    costs_path = args.costs
    if not costs_path.is_file():
        alt_costs = Path("outputs") / costs_path.name
        if alt_costs.is_file():
            costs_path = alt_costs

    if not coords_path.is_file():
        print(f"[ERRO] Coordenadas não encontradas em {args.coords} nem em {coords_path}.")
        print("Execute primeiro a etapa 05: python 05_pilot.py")
        sys.exit(1)

    if not costs_path.is_file():
        print(f"[ERRO] Custos dos checkpoints não encontrados em {args.costs} nem em {costs_path}.")
        sys.exit(1)

    print("=" * 70)
    print(" 06_overlay_checkpoints.py — Overlay dos Checkpoints no Espaço Fixo (Z1, Z2)")
    print("=" * 70)
    print(f" Coordenadas fixas : {coords_path}")
    print(f" Custos temporais  : {costs_path}")

    # 2. Carregar coordenadas e custos
    df_coords = pd.read_parquet(coords_path) if coords_path.suffix.lower() == ".parquet" else pd.read_csv(coords_path, index_col=0)
    df_costs = pd.read_csv(costs_path)

    # 3. Carregar BKS
    bks_lookup = load_bks(bks_csv=args.bks_file, instances_dir=args.instances_dir)
    print(f" BKS disponíveis    : {len(bks_lookup)} instâncias")

    # 4. Processar gaps dos checkpoints
    target_algs = [a.strip() for a in args.algorithms.split(",")] if args.algorithms else None
    df_consolidated, checkpoints, algorithms = compute_checkpoint_gaps(
        df_costs=df_costs,
        df_coords=df_coords,
        bks_lookup=bks_lookup,
        target_algorithms=target_algs,
    )

    print(f" Checkpoints        : {checkpoints}%")
    print(f" Algoritmos         : {algorithms}")
    print(f" Total de instâncias: {len(df_consolidated['instancia_id'].unique())}")

    # 5. Exportar dados consolidados (Parquet e CSV)
    args.out_checkpoints_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.out_checkpoints_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_plot_algorithms.parent.mkdir(parents=True, exist_ok=True)
    args.out_plot_best.parent.mkdir(parents=True, exist_ok=True)
    args.out_plot_best_points.parent.mkdir(parents=True, exist_ok=True)
    args.out_plot_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_params_log.parent.mkdir(parents=True, exist_ok=True)
    args.out_report_md.parent.mkdir(parents=True, exist_ok=True)

    df_consolidated.to_parquet(args.out_checkpoints_parquet)
    df_consolidated.to_csv(args.out_checkpoints_csv, index=False)
    print(f"\n[OK] Tabela consolidada de gaps salva em Parquet: {args.out_checkpoints_parquet}")
    print(f"[OK] Tabela consolidada de gaps salva em CSV    : {args.out_checkpoints_csv}")

    # 6. Calcular Métricas de Footprint (Notice et al., 2025, Tabela 3 & Smith-Miles & Muñoz, 2023)
    df_footprints = compute_footprint_metrics(
        df_consolidated=df_consolidated,
        checkpoints=checkpoints,
        algorithms=algorithms,
        epsilon=args.epsilon,
    )
    args.out_footprint_metrics.parent.mkdir(parents=True, exist_ok=True)
    df_footprints.to_csv(args.out_footprint_metrics, index=False)
    print(f"[OK] Métricas de Footprint salvas em CSV        : {args.out_footprint_metrics}")

    # 7. Gerar Gráficos
    plot_algorithm_gap_overlays(df_consolidated, checkpoints, algorithms, args.out_plot_algorithms)
    plot_best_selection_evolution(df_consolidated, checkpoints, algorithms, args.out_plot_best, show_polygons=True)
    plot_best_selection_evolution(df_consolidated, checkpoints, algorithms, args.out_plot_best_points, show_polygons=False)
    plot_summary_dashboard(df_consolidated, checkpoints, algorithms, args.out_plot_summary)

    # 8. Gerar params_log.md (Checklist de Entrega)
    generate_params_log(
        output_md=args.out_params_log,
        algorithms=algorithms,
        checkpoints=checkpoints,
        n_instances=len(df_consolidated["instancia_id"].unique()),
    )

    # 9. Gerar Relatório Técnico Markdown
    md_report = f"""# Relatório de Overlay dos Checkpoints no Espaço Fixo de Instâncias (Etapa 8)

## 1. Contexto Científico e Conexão com AAS
O espaço linear de instâncias $(Z_1, Z_2)$ foi construído e fixado na Etapa 7 pelo PILOT. Nesta etapa, sobrepomos os desempenhos obtidos em quatro checkpoints do orçamento temporal ($25\\%, 50\\%, 75\\%$ e $100\\%$ de $T(N)$) sobre as **mesmas coordenadas**, permitindo investigar:
1. Como os algoritmos de busca local refinam suas rotas ao longo do tempo.
2. Como a região de domínio (*footprint*) do algoritmo vencedor se altera conforme a busca amadurece.

## 2. Resumo da Distribuição do Vencedor por Checkpoint

| Checkpoint | Vencedor Majoritário | Instâncias Dominadas | Gap Médio do Vencedor |
|---|---|---|---|
"""
    n_insts = len(df_consolidated["instancia_id"].unique())
    for cp in checkpoints:
        sub_cp = df_consolidated[df_consolidated["checkpoint"] == cp]
        top_alg = sub_cp["best_algo"].value_counts().index[0]
        top_count = sub_cp["best_algo"].value_counts().values[0]
        mean_best_gap = sub_cp["best_gap"].mean() * 100
        md_report += f"| **{cp}% de T** | **{top_alg}** | {top_count}/{n_insts} ({top_count/n_insts*100:.1f}%) | {mean_best_gap:.2f}% |\n"

    md_report += f"""
### Detalhamento de Vitórias por Solver e Checkpoint:
"""
    for cp in checkpoints:
        sub_cp = df_consolidated[df_consolidated["checkpoint"] == cp]
        counts = sub_cp["best_algo"].value_counts().to_dict()
        md_report += f"- **Checkpoint {cp}%:** " + ", ".join([f"{alg}: {cnt} inst" for alg, cnt in counts.items()]) + "\n"

    md_report += f"""
## 3. Avaliação Quantitativa dos Footprints (Notice et al., 2025, Tabela 3 & Smith-Miles & Muñoz, 2023)

- **Critério de Good Performance:** $\\text{{Gap}} \\le {args.epsilon*100:.1f}\\%$
- **Critério de Best Performance:** Algoritmo com melhor solução na instância
- **Normalized Area ($A$):** Fração da área do Convex Hull do espaço total dominada pelo algoritmo
- **Density ($D$):** Concentração de instâncias no footprint relativa à densidade média do espaço
- **Purity ($P$):** Proporção de instâncias alvo contidas dentro do polígono de fronteira

| Checkpoint | Algoritmo | Good Area ($A$) | Good Density ($D$) | Good Purity ($P$) | Best Area ($A$) | Best Density ($D$) | Best Purity ($P$) | N Good | N Best |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for _, r in df_footprints.iterrows():
        md_report += (
            f"| **{r['checkpoint']}%** | **{r['algoritmo']}** | "
            f"{r['good_area']:.3f} | {r['good_density']:.3f} | {r['good_purity']:.3f} | "
            f"{r['best_area']:.3f} | {r['best_density']:.3f} | {r['best_purity']:.3f} | "
            f"{r['n_good']} | {r['n_best']} |\n"
        )

    md_report += f"""
## 4. Transição do Melhor Algoritmo por Instância ao Longo dos Checkpoints

| Instância | Vencedor 25% | Vencedor 50% | Vencedor 75% | Vencedor 100% | Gap 100% | Estado |
|---|---|---|---|---|---:|:---:|
"""
    for i_id in sorted(df_consolidated["instancia_id"].unique()):
        inst_df = df_consolidated[df_consolidated["instancia_id"] == i_id].sort_values("checkpoint")
        bests = {r["checkpoint"]: r["best_algo"] for _, r in inst_df.iterrows()}
        final_rows = inst_df[inst_df["checkpoint"] == 100]
        final_gap = final_rows.iloc[0]["best_gap"] * 100 if len(final_rows) > 0 else 0.0
        has_changed = len(set(bests.values())) > 1
        change_str = "Mudou" if has_changed else "Estável"
        md_report += f"| `{i_id}` | {bests.get(25, '-')} | {bests.get(50, '-')} | {bests.get(75, '-')} | **{bests.get(100, '-')}** | {final_gap:.2f}% | {change_str} |\n"

    md_report += f"""
## 5. Fundamentação dos Polígonos de Footprint na Literatura de ISA

Na metodologia de Instance Space Analysis (Smith-Miles & Muñoz, 2023; Notice et al., 2025; Gouvêa et al., 2025):
- As instâncias empíricas testadas são pontos discretos no espaço projetado $(Z_1, Z_2)$.
- O conceito matemático de **Footprint** ($\mathcal{{F}}$) define a região contínua do espaço onde um determinado algoritmo apresenta desempenho superior (*Best Solver*) ou aceitável (*Good Performance*, $\\text{{Gap}} \\le \\varepsilon$).
- Para calcular métricas contínuas a partir de uma amostra discreta, a literatura calcula o **Convex Hull** (envoltório convexo bidimensional) que circunscreve as instâncias onde o algoritmo vence.
- A área normalizada desse polígono ($A = \\text{{Área}}(\\mathcal{{H}}_{{\\text{{algo}}}}) / \\text{{Área}}(\\mathcal{{H}}_{{\\text{{total}}}})$) quantifica o tamanho do nicho de dominância de cada heurística (Notice et al., 2025, Tabela 3).
- **Formatos de Visualização Gerados:**
  1. `checkpoint_overlays_best_selection.png`: Exibe os envoltórios poligonais semitransparentes sobrepostos aos pontos, delimitando as fronteiras contínuas de cada algoritmo.
  2. `checkpoint_overlays_best_selection_points.png`: Exibe dispersão pura de pontos sem polígonos, permitindo inspeção nítida de cada instância individual sem oclusão geométrica.

## 6. Artefatos de Visualização Produzidos
| Arquivo | Descrição |
|---|---|
| `checkpoint_overlays_algorithms.png` | Grade comparativa de todos os algoritmos nos 4 checkpoints |
| `checkpoint_overlays_best_selection.png` | Evolução visual do Best Selection com polígonos de footprint (Convex Hull) |
| `checkpoint_overlays_best_selection_points.png` | Evolução visual do Best Selection sem polígonos (dispersão pura de pontos) |
| `checkpoint_evolution_summary.png` | Painel consolidado com curvas de convergência e ganhos de qualidade |
| `footprint_metrics.csv` | Tabela formal de métricas de footprint (Area, Density, Purity) |
| `Y_checkpoints.parquet` / `Y_checkpoints.csv` | Tabela estruturada para análises econométricas posteriores |
| `params_log.md` | Checklist completo dos parâmetros metodológicos |
"""
    with args.out_report_md.open("w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"[OK] Relatório Markdown salvo em: {args.out_report_md}")

    print("\n" + "=" * 70)
    print(" Etapa 8 (Overlay dos Checkpoints) CONCLUÍDA com sucesso!")
    print(" Pipeline completo de Instance Space Analysis (ISA) finalizado!")
    print("=" * 70)


if __name__ == "__main__":
    main()
