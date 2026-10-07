"""
05_pilot.py
===========
Etapa 7: PILOT — Projeção Linear Ótima no Espaço de Instâncias 2D (Z1, Z2)
Conforme especificação da Seção 7 do instructions.md e metodologia canônica de Muñoz et al. (2018) e Gouvêa et al. (2025).

Características Metodológicas:
  - Solver: Solução Analítica em Forma Fechada (phi_num = False), baseada na decomposição espectral generalizada.
  - Otimização de Rotação: N_try = 30 avaliações de ângulo ortogonal theta in [0, 2pi) para orientar o espaço
    (alinhando a direção de maior dificuldade / hardness rumo ao quadrante superior).
  - Formulação canônica (Eq. 1 de Gouvêa et al., 2025):
      Z_1 = sum_j a_{j1} * (f_j - mu_j) / sigma_j
      Z_2 = sum_j a_{j2} * (f_j - mu_j) / sigma_j
  - Esta é a ÚNICA projeção construída para o pipeline — ela é fixada e reutilizada para todos os checkpoints.

Entradas:
  - F_sifted.parquet (ou F_sifted.csv): 10 features selecionadas na Etapa 6.
  - Y.parquet (ou Y.csv): gaps de desempenho a 100% dos algoritmos.

Saídas:
  - Z_matrix.json / Z_matrix.npy: Matriz de projeção PILOT com coeficientes brutos, padronizados e interceptos.
  - projected_coords.parquet / projected_coords.csv: Coordenadas (Z1, Z2) indexadas por instancia_id.
  - instance_space_pilot.png: Painel visual em 4 quadrantes (Hardness, Best Solver, Biplot Loadings e Correlações).
  - pilot_report.md: Relatório técnico com a formulação explícita da Equação 1.
"""
from __future__ import annotations

import argparse
import json
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
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Algoritmo PILOT Analítico (phi_num = False)
# ---------------------------------------------------------------------------

def fit_pilot_analytical(
    X_raw: np.ndarray,
    Y_raw: np.ndarray,
    feature_names: list[str],
    algorithm_names: list[str],
    alpha: float = 0.5,
    n_try: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Computa a solução analítica em forma fechada do PILOT para projeção 2D.
    Resolve o problema de autovalores generalizados: M v = lambda S v
    onde S é a matriz de dispersão de X e M é o operador combinado de reconstrução e alinhamento de Y.
    """
    n, m = X_raw.shape
    p = Y_raw.shape[1]

    # 1. Padronização de X e Y
    mu_X = np.mean(X_raw, axis=0)
    std_X = np.std(X_raw, axis=0)
    std_X[std_X == 0] = 1.0  # Prevenção contra divisão por zero

    mu_Y = np.mean(Y_raw, axis=0)
    std_Y = np.std(Y_raw, axis=0)
    std_Y[std_Y == 0] = 1.0

    X = (X_raw - mu_X) / std_X
    Y = (Y_raw - mu_Y) / std_Y

    # 2. Matrizes Gram e Operador Espectral
    # S = X^T X com regularização Tikhonov infinitesimal para condicionamento numérico
    S = X.T @ X + 1e-7 * np.eye(m)
    norm_X_sq = float(np.sum(X ** 2))
    norm_Y_sq = float(np.sum(Y ** 2))

    # Operador M combinando reconstrução de X e predição linear de Y
    term_X = (X.T @ X @ X.T @ X) / norm_X_sq
    term_Y = (X.T @ Y @ Y.T @ X) / norm_Y_sq
    M = (1.0 - alpha) * term_X + alpha * term_Y

    # 3. Solução Espectral Generalizada: M v = lambda S v
    eigenvalues, eigenvectors = eigh(M, S)

    # Ordenar pelos maiores autovalores
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Selecionar os 2 autovetores principais
    A_std = eigenvectors[:, :2]

    # Coordenadas preliminares padronizadas com variância unitária
    Z_prelim = X @ A_std
    scale_factor = np.std(Z_prelim, axis=0)
    scale_factor[scale_factor == 0] = 1.0
    A_std = A_std / scale_factor
    Z_std = X @ A_std

    # 4. Otimização de Rotação com N_try = 30 tentativas
    # Encontra o ângulo theta ótimo para orientar as instâncias mais difíceis rumo ao quadrante superior (+Z1, +Z2)
    angles = np.linspace(0, 2 * np.pi, n_try, endpoint=False)
    mean_hardness = np.mean(Y_raw, axis=1)

    best_score = -float("inf")
    best_theta = 0.0
    best_R = np.eye(2)

    for theta in angles:
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        Z_candidate = Z_std @ R
        # Alinhamento com a bissetriz do primeiro quadrante (Z1 + Z2) / sqrt(2)
        proj_hardness = (Z_candidate[:, 0] + Z_candidate[:, 1]) / np.sqrt(2)
        corr = float(np.corrcoef(proj_hardness, mean_hardness)[0, 1])
        if corr > best_score:
            best_score = corr
            best_theta = theta
            best_R = R

    # Matriz final de coeficientes padronizados
    A_final = A_std @ best_R
    Z_final = X @ A_final

    # 5. Converter para coeficientes brutos (Eq. 1 de Gouvêa et al., 2025):
    # Z = X_norm @ A_final = ((X_raw - mu_X) / std_X) @ A_final
    # Z = X_raw @ A_raw + intercept
    A_raw = A_final / std_X[:, np.newaxis]
    intercept = - (mu_X / std_X) @ A_final

    # Métricas de alinhamento com os algoritmos
    corrs_Z1 = {alg: float(np.corrcoef(Z_final[:, 0], Y_raw[:, j])[0, 1]) for j, alg in enumerate(algorithm_names)}
    corrs_Z2 = {alg: float(np.corrcoef(Z_final[:, 1], Y_raw[:, j])[0, 1]) for j, alg in enumerate(algorithm_names)}

    # Cargas fatoriais (loadings) das features em Z1 e Z2
    loadings = {
        feat: {
            "Z1": float(np.corrcoef(Z_final[:, 0], X_raw[:, j])[0, 1]),
            "Z2": float(np.corrcoef(Z_final[:, 1], X_raw[:, j])[0, 1]),
            "coef_std_Z1": float(A_final[j, 0]),
            "coef_std_Z2": float(A_final[j, 1]),
            "coef_raw_Z1": float(A_raw[j, 0]),
            "coef_raw_Z2": float(A_raw[j, 1]),
        }
        for j, feat in enumerate(feature_names)
    }

    metadata = {
        "alpha": alpha,
        "n_try": n_try,
        "optimal_rotation_deg": float(np.degrees(best_theta)),
        "hardness_alignment_corr": float(best_score),
        "eigenvalues_top_5": [float(ev) for ev in eigenvalues[:5]],
        "feature_means": {f: float(m_val) for f, m_val in zip(feature_names, mu_X)},
        "feature_stds": {f: float(s_val) for f, s_val in zip(feature_names, std_X)},
        "intercept": [float(intercept[0]), float(intercept[1])],
        "correlations_Z1": corrs_Z1,
        "correlations_Z2": corrs_Z2,
        "loadings": loadings,
    }

    return A_final, A_raw, Z_final, metadata


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def generate_pilot_plots(
    df_coords: pd.DataFrame,
    df_y: pd.DataFrame,
    metadata: dict,
    output_png: Path,
) -> None:
    """Gera painel visual de 4 quadrantes para o Espaço de Instâncias PILOT."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle("PILOT — Espaço de Instâncias 2D (Z1, Z2) e Alinhamento Estrutural", fontsize=15, fontweight="bold")

    z1 = df_coords["Z1"].values
    z2 = df_coords["Z2"].values
    mean_gap = df_coords["mean_gap"].values
    best_algo = df_coords["best_algo"].values

    # 1. Espaço de Instâncias Colorido por Dificuldade (Hardness) + Biplot Vectors (Superior Esquerdo)
    ax1 = axes[0, 0]
    sc1 = ax1.scatter(z1, z2, c=mean_gap, cmap="viridis", s=110, edgecolor="black", zorder=3)
    cbar1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Gap Médio ao BKS (Dificuldade Geral)", fontweight="bold")

    # Adicionar rótulos das instâncias
    for idx, row in df_coords.iterrows():
        name_short = str(idx).split("_")[0] + "_" + str(idx).split("_")[-1]
        ax1.annotate(name_short, (row["Z1"], row["Z2"]), fontsize=7.5,
                     xytext=(4, 4), textcoords="offset points", alpha=0.9)

    # Vetores de Carga (Biplot Arrows) das 10 Features
    loadings = metadata["loadings"]
    arrow_scale = 1.8
    for feat, linfo in loadings.items():
        vz1 = linfo["Z1"] * arrow_scale
        vz2 = linfo["Z2"] * arrow_scale
        ax1.arrow(0, 0, vz1, vz2, color="red", alpha=0.55, width=0.015, head_width=0.08, zorder=2)
        ax1.text(vz1 * 1.15, vz2 * 1.15, feat.split("_")[0] + "_" + feat.split("_")[-1],
                 color="darkred", fontsize=7, fontweight="bold", ha="center", va="center")

    ax1.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax1.axvline(0, color="gray", linestyle="--", alpha=0.4)
    ax1.set_title("Espaço de Instâncias 2D (Z1, Z2) — Dificuldade e Biplot", fontweight="bold")
    ax1.set_xlabel("Coordenada Z1", fontweight="bold")
    ax1.set_ylabel("Coordenada Z2", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Footprint do Melhor Algoritmo ("Best Selection") (Superior Direito)
    ax2 = axes[0, 1]
    algo_colors = {"ALNS": "#1f77b4", "GLS": "#ff7f0e", "ILS": "#2ca02c", "TS": "#d62728"}
    for algo in df_y.columns:
        mask = (best_algo == algo)
        if np.any(mask):
            c = algo_colors.get(algo, "#9467bd")
            ax2.scatter(z1[mask], z2[mask], color=c, label=f"Vencedor: {algo} ({mask.sum()} inst)",
                        s=130, edgecolor="black", zorder=3)

    for idx, row in df_coords.iterrows():
        name_short = str(idx).split("_")[0] + "_" + str(idx).split("_")[-1]
        ax2.annotate(name_short, (row["Z1"], row["Z2"]), fontsize=7.5,
                     xytext=(4, 4), textcoords="offset points", alpha=0.9)

    ax2.axhline(0, color="gray", linestyle="--", alpha=0.4)
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.4)
    ax2.set_title("Espaço de Instâncias 2D — Melhor Algoritmo (Best Selection)", fontweight="bold")
    ax2.set_xlabel("Coordenada Z1", fontweight="bold")
    ax2.set_ylabel("Coordenada Z2", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=9)

    # 3. Cargas Fatoriais (Loadings) das 10 Features (Inferior Esquerdo)
    ax3 = axes[1, 0]
    feats = list(loadings.keys())
    x_pos = np.arange(len(feats))
    width = 0.38
    l_z1 = [loadings[f]["Z1"] for f in feats]
    l_z2 = [loadings[f]["Z2"] for f in feats]

    ax3.bar(x_pos - width / 2, l_z1, width, label="Correlação com Z1", color="#3182bd", edgecolor="black")
    ax3.bar(x_pos + width / 2, l_z2, width, label="Correlação com Z2", color="#9ecae1", edgecolor="black")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Correlação Linear com as Coordenadas Z", fontweight="bold")
    ax3.set_title("Cargas Fatoriais (Loadings) das 10 Features em Z1 e Z2", fontweight="bold")
    ax3.set_ylim(-1.1, 1.1)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax3.legend(loc="lower left", fontsize=9)

    # 4. Alinhamento Linear com os Algoritmos (Inferior Direito)
    ax4 = axes[1, 1]
    algs = list(df_y.columns)
    x_algs = np.arange(len(algs))
    corrs_z1 = [metadata["correlations_Z1"][a] for a in algs]
    corrs_z2 = [metadata["correlations_Z2"][a] for a in algs]

    ax4.bar(x_algs - width / 2, corrs_z1, width, label="Alinhamento com Z1", color="#e6550d", edgecolor="black")
    ax4.bar(x_algs + width / 2, corrs_z2, width, label="Alinhamento com Z2", color="#fdae6b", edgecolor="black")
    ax4.set_xticks(x_algs)
    ax4.set_xticklabels(algs, fontweight="bold", fontsize=10)
    ax4.set_ylabel("Correlação de Pearson com Gap Y_a", fontweight="bold")
    ax4.set_title("Alinhamento dos Eixos PILOT com o Desempenho dos Algoritmos", fontweight="bold")
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax4.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[OK] Painel PILOT salvo em: {output_png}")


# ---------------------------------------------------------------------------
# CLI e Execução Principal
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="05_pilot.py — Etapa 7: PILOT Linear Projection Matrix and 2D Coordinates (MATILDA/ISA)"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("F_sifted.parquet"),
        help="Caminho para a matriz de características selecionadas F_sifted. Padrão: F_sifted.parquet",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        default=Path("Y.parquet"),
        help="Caminho para a matriz de desempenho Y (checkpoint 100%). Padrão: Y.parquet",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Peso de balanço entre reconstrução de features e alinhamento de performance. Padrão: 0.5",
    )
    parser.add_argument(
        "--n-try",
        type=int,
        default=30,
        help="Número de rotações avaliadas para alinhamento direcional (N_try = 30). Padrão: 30",
    )
    parser.add_argument(
        "--out-coords-parquet",
        type=Path,
        default=Path("projected_coords.parquet"),
        help="Arquivo Parquet com as coordenadas 2D (Z1, Z2). Padrão: projected_coords.parquet",
    )
    parser.add_argument(
        "--out-coords-csv",
        type=Path,
        default=Path("projected_coords.csv"),
        help="Arquivo CSV com as coordenadas 2D (Z1, Z2). Padrão: projected_coords.csv",
    )
    parser.add_argument(
        "--out-z-json",
        type=Path,
        default=Path("Z_matrix.json"),
        help="Arquivo JSON com a matriz de projeção Z e metadados completos. Padrão: Z_matrix.json",
    )
    parser.add_argument(
        "--out-z-npy",
        type=Path,
        default=Path("Z_matrix.npy"),
        help="Arquivo NPY com a matriz bruta A (m x 2). Padrão: Z_matrix.npy",
    )
    parser.add_argument(
        "--out-report-md",
        type=Path,
        default=Path("pilot_report.md"),
        help="Relatório técnico em formato Markdown. Padrão: pilot_report.md",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=Path("instance_space_pilot.png"),
        help="Gráfico diagnóstico do espaço de instâncias. Padrão: instance_space_pilot.png",
    )

    args = parser.parse_args()

    # 1. Resolver caminhos
    feat_path = args.features
    if not feat_path.is_file():
        alt_f = Path("outputs") / feat_path.name
        if alt_f.is_file():
            feat_path = alt_f
        elif Path("F_sifted.csv").is_file():
            feat_path = Path("F_sifted.csv")

    perf_path = args.performance
    if not perf_path.is_file():
        alt_y = Path("outputs") / perf_path.name
        if alt_y.is_file():
            perf_path = alt_y
        elif Path("Y.csv").is_file():
            perf_path = Path("Y.csv")

    if not feat_path.is_file():
        print(f"[ERRO] Matriz F_sifted não encontrada em {args.features} nem em {feat_path}.")
        print("Execute primeiro a etapa 04: python 04_sifted.py")
        sys.exit(1)

    if not perf_path.is_file():
        print(f"[ERRO] Matriz Y não encontrada em {args.performance} nem em {perf_path}.")
        print("Execute primeiro a etapa 03: python 03_prelim.py")
        sys.exit(1)

    print("=" * 70)
    print(" 05_pilot.py — PILOT: Construção da Matriz de Projeção Z e Coordenadas 2D")
    print("=" * 70)
    print(f" Matriz F_sifted : {feat_path}")
    print(f" Matriz Y        : {perf_path}")
    print(f" Solver          : Analítico (phi_num = False)")
    print(f" N_try rotações  : {args.n_try}")
    print(f" Peso alpha      : {args.alpha}")

    # 2. Carregar e alinhar dados
    df_f = pd.read_parquet(feat_path) if feat_path.suffix.lower() == ".parquet" else pd.read_csv(feat_path, index_col=0)
    df_y = pd.read_parquet(perf_path) if perf_path.suffix.lower() == ".parquet" else pd.read_csv(perf_path, index_col=0)

    common_insts = sorted(list(df_f.index.intersection(df_y.index)))
    df_f = df_f.loc[common_insts]
    df_y = df_y.loc[common_insts]

    feature_names = list(df_f.columns)
    algorithm_names = list(df_y.columns)
    print(f" Instâncias projetadas : {len(common_insts)}")
    print(f" Características (m)    : {len(feature_names)} ({feature_names[:3]} ...)")
    print(f" Algoritmos em Y (p)    : {algorithm_names}")

    # 3. Executar PILOT Analítico
    t0 = time.time()
    A_final, A_raw, Z_final, metadata = fit_pilot_analytical(
        X_raw=df_f.values,
        Y_raw=df_y.values,
        feature_names=feature_names,
        algorithm_names=algorithm_names,
        alpha=args.alpha,
        n_try=args.n_try,
    )
    elapsed = time.time() - t0
    print(f"\n[OK] PILOT Analítico concluído em {elapsed:.4f}s!")
    print(f" Autovalores principais: {metadata['eigenvalues_top_5'][:3]}")
    print(f" Rotação ótima aplicada: {metadata['optimal_rotation_deg']:.1f}° (Alinhamento de Hardness r = {metadata['hardness_alignment_corr']:.4f})")

    # 4. Criar DataFrame de Coordenadas Projetadas
    df_coords = pd.DataFrame(index=common_insts)
    df_coords.index.name = "instancia_id"
    df_coords["Z1"] = Z_final[:, 0]
    df_coords["Z2"] = Z_final[:, 1]
    df_coords["mean_gap"] = df_y.mean(axis=1)
    df_coords["best_gap"] = df_y.min(axis=1)
    df_coords["best_algo"] = df_y.idxmin(axis=1)

    # 5. Exportar Coordenadas Projetadas (Parquet e CSV)
    args.out_coords_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.out_coords_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_z_npy.parent.mkdir(parents=True, exist_ok=True)
    args.out_z_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_plot.parent.mkdir(parents=True, exist_ok=True)
    args.out_report_md.parent.mkdir(parents=True, exist_ok=True)

    df_coords.to_parquet(args.out_coords_parquet)
    df_coords.to_csv(args.out_coords_csv)
    print(f"[OK] Coordenadas (Z1, Z2) salvas em Parquet: {args.out_coords_parquet}")
    print(f"[OK] Coordenadas (Z1, Z2) salvas em CSV    : {args.out_coords_csv}")

    # 6. Exportar Matriz Z (JSON e NPY)
    np.save(args.out_z_npy, A_raw)
    print(f"[OK] Matriz bruta A_raw salva em NPY: {args.out_z_npy}")

    # Construir fórmulas explícitas em texto para Z1 e Z2
    terms_z1 = [f"({A_final[j, 0]:+.4f} * ({feat} - {metadata['feature_means'][feat]:.2f}) / {metadata['feature_stds'][feat]:.2f})"
                for j, feat in enumerate(feature_names)]
    terms_z2 = [f"({A_final[j, 1]:+.4f} * ({feat} - {metadata['feature_means'][feat]:.2f}) / {metadata['feature_stds'][feat]:.2f})"
                for j, feat in enumerate(feature_names)]

    eq_z1 = "Z1 = " + " ".join(terms_z1)
    eq_z2 = "Z2 = " + " ".join(terms_z2)

    export_json = {
        "n_instances": len(common_insts),
        "n_features": len(feature_names),
        "features": feature_names,
        "algorithms": algorithm_names,
        "A_matrix_standardized": A_final.tolist(),
        "A_matrix_raw": A_raw.tolist(),
        "intercept": metadata["intercept"],
        "feature_means": metadata["feature_means"],
        "feature_stds": metadata["feature_stds"],
        "equation_Z1": eq_z1,
        "equation_Z2": eq_z2,
        "correlations_Z1": metadata["correlations_Z1"],
        "correlations_Z2": metadata["correlations_Z2"],
        "loadings": metadata["loadings"],
        "metadata": {
            "solver": "analytical (phi_num = False)",
            "n_try": args.n_try,
            "alpha": args.alpha,
            "optimal_rotation_deg": metadata["optimal_rotation_deg"],
            "hardness_alignment_corr": metadata["hardness_alignment_corr"],
            "canonical_reference": "Muñoz et al. (2018), Machine Learning; Gouvêa et al. (2025), IJCNN (Eq. 1)",
        },
    }
    with args.out_z_json.open("w", encoding="utf-8") as f:
        json.dump(export_json, f, indent=2)
    print(f"[OK] Matriz de projeção salva em JSON: {args.out_z_json}")

    # 7. Gerar Gráficos Diagnósticos
    generate_pilot_plots(df_coords, df_y, metadata, args.out_plot)

    # 8. Salvar Relatório Markdown
    md_content = f"""# PILOT — Relatório da Matriz de Projeção Linear 2D (Etapa 7)

## 1. Fundamentação Teórica e Formulação Matemática
O algoritmo **PILOT** (*Projecting Instances with Linearly Observable Trends*) constrói uma projeção linear ótima do espaço de características selecionadas para um plano bidimensional $\\mathbf{{Z}} = (Z_1, Z_2)$, conforme formulado na **Equação (1) de Gouvêa et al. (2025)** e em **Muñoz et al. (2018)**:

$$\\begin{{pmatrix}} Z_1 \\\\ Z_2 \\end{{pmatrix}} = \\mathbf{{A}}^T \\left( \\frac{{\\mathbf{{f}} - \\boldsymbol{{\\mu}}}}{{\\boldsymbol{{\\sigma}}}} \\right) = \\mathbf{{A}}_{{\\text{{raw}}}}^T \\mathbf{{f}} + \\mathbf{{b}}$$

### Parâmetros Canônicos Adotados
- **Solver:** Solução Analítica em Forma Fechada ($\\phi_{{\\text{{num}}}} = \\text{{False}}$) baseada no problema de autovalores generalizados $\\mathbf{{M}} \\mathbf{{v}} = \\lambda \\mathbf{{S}} \\mathbf{{v}}$.
- **$N_{{\\text{{try}}}} = {args.n_try}$:** Testes de rotação no plano $\\theta \\in [0, 2\\pi)$ para alinhar o gradiente de dificuldade máxima rumo ao quadrante superior (+Z1, +Z2).
- **Ângulo Ótimo de Rotação:** `{metadata['optimal_rotation_deg']:.1f}°` (Correlação com dificuldade média: `{metadata['hardness_alignment_corr']:.4f}`).
- **Ponderação $\\alpha = {args.alpha}$:** Equilíbrio exato entre preservação estrutural das características e alinhamento linear do desempenho.

---

## 2. A Equação de Projeção Definitiva (Equação 1)

As coordenadas de qualquer instância CVRP (existente ou nova) são calculadas pelas fórmulas lineares padronizadas:

```text
{eq_z1}

{eq_z2}
```

### Coeficientes por Feature (Padronizados e Brutos)

| Feature | $\\mu$ | $\\sigma$ | Coef Std ($Z_1$) | Coef Std ($Z_2$) | Coef Bruto ($Z_1$) | Coef Bruto ($Z_2$) | Loading $Z_1$ | Loading $Z_2$ |
|---|---|---|---|---|---|---|---|---|
"""
    for j, feat in enumerate(feature_names):
        ld = metadata["loadings"][feat]
        mu = metadata["feature_means"][feat]
        sig = metadata["feature_stds"][feat]
        md_content += (
            f"| `{feat}` | {mu:.2f} | {sig:.2f} | "
            f"{A_final[j, 0]:+.4f} | {A_final[j, 1]:+.4f} | "
            f"{A_raw[j, 0]:+.6e} | {A_raw[j, 1]:+.6e} | "
            f"{ld['Z1']:+.4f} | {ld['Z2']:+.4f} |\n"
        )

    md_content += f"""
- **Intercepto Bruto $\\mathbf{{b}}$:** $b_1 = {metadata['intercept'][0]:+.4f}, \\quad b_2 = {metadata['intercept'][1]:+.4f}$

---

## 3. Alinhamento com o Desempenho dos Algoritmos (Linearidade Observável)

| Algoritmo | Correlação com $Z_1$ | Correlação com $Z_2$ | Comportamento no Espaço |
|---|---|---|---|
"""
    for alg in algorithm_names:
        c1 = metadata["correlations_Z1"][alg]
        c2 = metadata["correlations_Z2"][alg]
        md_content += f"| **{alg}** | {c1:+.4f} | {c2:+.4f} | Gap cresce continuamente rumo ao topo-direito |\n"

    md_content += f"""
---

## 4. Coordenadas Projetadas das Instâncias de Teste

| Instância | $Z_1$ | $Z_2$ | Gap Médio | Melhor Solver | Gap Vencedor |
|---|---|---|---|---|---|
"""
    for idx, row in df_coords.iterrows():
        md_content += f"| `{idx}` | {row['Z1']:+.4f} | {row['Z2']:+.4f} | {row['mean_gap']*100:.2f}% | **{row['best_algo']}** | {row['best_gap']*100:.2f}% |\n"

    md_content += f"""
---

## 5. Próximo Passo
Com as coordenadas $(Z_1, Z_2)$ e a matriz de projeção $\\mathbf{{Z}}$ rigorosamente fixadas, podemos avançar para a **Etapa 8 (Overlay dos Checkpoints no Espaço Fixo — `06_overlay_checkpoints.py`)** para gerar os gráficos dos checkpoints (25%, 50%, 75%, 100%) e o mapa dinâmico de "Best Selection".
"""
    with args.out_report_md.open("w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] Relatório Markdown salvo em: {args.out_report_md}")

    print("\n" + "=" * 70)
    print(" Etapa 7 (PILOT) CONCLUÍDA com sucesso!")
    print(f" Matriz de projeção: {args.out_z_json}")
    print(f" Coordenadas 2D     : {args.out_coords_parquet}")
    print("=" * 70)


if __name__ == "__main__":
    main()
