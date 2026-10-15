"""
07_project_new_instances.py
===========================
Etapa de Extensibilidade: Projeção de Novas Instâncias no Espaço PILOT Fixado (Z1, Z2)
Conforme especificação da Seção 10 do instructions.md.

Objetivo:
  Reaplicar o espaço de instâncias já construído a instâncias externas (ex.: Set X de Uchoa,
  benchmark público CVRPLib, ou novas instâncias sintéticas) SEM refazer SIFTED nem PILOT,
  e opcionalmente sem rodar nenhum algoritmo nelas (imputação neutra de probing features),
  permitindo a comparação visual de cobertura e diversidade topológica.

Entradas:
  - Diretório de instâncias .vrp externas (--instances-dir)
  - Z_matrix.json gerado na Etapa 7 (contendo features selecionadas, medias, desvios e coeficientes PILOT)
  - Opcional: projected_coords.parquet (coordenadas do dataset base para overlay)
  - Opcional: arquivo de probing features caso o usuário tenha executado traço do ILS

Saídas:
  - projected_new_instances.parquet / .csv: coordenadas (Z1, Z2) das novas instâncias.
  - projected_new_instances_overlay.png: mapa visual sobrepondo instâncias de referência e novas.
  - projected_new_instances_report.md: relatório técnico de cobertura e diversidade.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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

from cvrp_features import extract_instance_features


def load_pilot_model(z_matrix_path: Path) -> dict:
    """Carrega o modelo PILOT consolidado em Z_matrix.json."""
    if not z_matrix_path.exists():
        raise FileNotFoundError(f"Arquivo de projeção {z_matrix_path} não encontrado.")
    with open(z_matrix_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def project_instances(
    df_features: pd.DataFrame,
    pilot_model: dict,
    probing_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Aplica a normalização e a multiplicação da matriz Z para projetar novas instâncias em (Z1, Z2).
    """
    selected_features = pilot_model["features"]
    feature_means = pilot_model["feature_means"]
    feature_stds = pilot_model["feature_stds"]
    A_std = np.array(pilot_model["A_matrix_standardized"], dtype=float)

    # Montar matriz das 10 features para cada instância
    feature_values = []
    inst_ids = list(df_features.index)

    for inst_id in inst_ids:
        row_vals = []
        for feat in selected_features:
            if feat in df_features.columns and not pd.isna(df_features.loc[inst_id, feat]):
                row_vals.append(float(df_features.loc[inst_id, feat]))
            elif probing_df is not None and feat in probing_df.columns and inst_id in probing_df.index:
                row_vals.append(float(probing_df.loc[inst_id, feat]))
            else:
                # Se for feature de probing ausente, imputar com a média de referência do Z_matrix
                mean_val = feature_means.get(feat, 0.0)
                row_vals.append(mean_val)
        feature_values.append(row_vals)

    X_mat = np.array(feature_values, dtype=float)  # (N, 10)

    # Padronização z-score com parâmetros da população de treino
    means_arr = np.array([feature_means[f] for f in selected_features], dtype=float)
    stds_arr = np.array([feature_stds[f] for f in selected_features], dtype=float)
    stds_arr[stds_arr < 1e-12] = 1.0

    X_std = (X_mat - means_arr) / stds_arr

    # Projeção linear PILOT: Z = X_std @ A_std
    Z_coords = X_std @ A_std  # (N, 2)

    df_proj = pd.DataFrame(
        Z_coords,
        index=inst_ids,
        columns=["Z1", "Z2"]
    )
    df_proj.index.name = "instancia_id"
    return df_proj


def plot_projection_overlay(
    df_ref: pd.DataFrame | None,
    df_new: pd.DataFrame,
    pilot_model: dict,
    output_png: Path,
    title_suffix: str = "Extensão de Instâncias"
):
    """
    Plota as novas instâncias sobre o mapa de referência 2D (Z1, Z2).
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # 1. Plotar pontos de referência em segundo plano
    if df_ref is not None and "Z1" in df_ref.columns and "Z2" in df_ref.columns:
        ax.scatter(
            df_ref["Z1"], df_ref["Z2"],
            c="#94A3B8", s=60, alpha=0.55, edgecolors="#475569", linewidth=0.8,
            label=f"Dataset Base (N={len(df_ref)})", zorder=2
        )

    # 2. Plotar novas instâncias em primeiro plano com destaque
    ax.scatter(
        df_new["Z1"], df_new["Z2"],
        c="#DC2626", s=130, marker="*", alpha=0.95, edgecolors="#7F1D1D", linewidth=1.2,
        label=f"Novas Instâncias Projetadas (N={len(df_new)})", zorder=5
    )

    # Rótulos para novas instâncias apenas se N <= 12 para evitar poluição visual
    if len(df_new) <= 12:
        for inst_id, row in df_new.iterrows():
            ax.annotate(
                str(inst_id),
                (row["Z1"], row["Z2"]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=8, fontweight="bold", color="#991B1B",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#DC2626", alpha=0.8, lw=0.5),
                zorder=6
            )

    # 3. Adicionar vetores direcionais (biplot loadings)
    A_std = np.array(pilot_model["A_matrix_standardized"], dtype=float)
    features = pilot_model["features"]
    
    # Escala visual dos vetores
    all_z1 = np.concatenate([df_new["Z1"].values, (df_ref["Z1"].values if df_ref is not None else [])])
    all_z2 = np.concatenate([df_new["Z2"].values, (df_ref["Z2"].values if df_ref is not None else [])])
    scale_factor = 0.4 * max(np.ptp(all_z1), np.ptp(all_z2))

    for idx, feat in enumerate(features):
        vx = A_std[idx, 0] * scale_factor
        vy = A_std[idx, 1] * scale_factor
        ax.annotate(
            "", xy=(vx, vy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#0284C7", lw=1.2, alpha=0.7),
            zorder=3
        )
        ax.text(
            vx * 1.12, vy * 1.12, feat,
            fontsize=7.5, color="#0369A1", fontweight="semibold",
            ha="center", va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#BAE6FD", alpha=0.75, lw=0.5)
        )

    # Linhas de centro
    ax.axhline(0, color="#CBD5E1", linestyle="--", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#CBD5E1", linestyle="--", linewidth=0.8, zorder=1)

    ax.set_xlabel("Coordenada Z₁ (Eixo PILOT 1)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Coordenada Z₂ (Eixo PILOT 2 — Alinhamento de Hardness)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        f"Espaço de Instâncias CVRP — Projeção Externa ({title_suffix})\n"
        f"Reaplicação da Projeção Canônica PILOT (Seção 10 do instructions.md)",
        fontsize=12, fontweight="bold", pad=12
    )

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.9, facecolor="white", edgecolor="#CBD5E1")
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"Gráfico de projeção externa salvo: {output_png}")


def generate_projection_report(
    df_new: pd.DataFrame,
    df_ref: pd.DataFrame | None,
    pilot_model: dict,
    output_md: Path,
    probing_imputed: bool
):
    """Gera relatório markdown documentando as instâncias projetadas."""
    n_new = len(df_new)
    n_ref = len(df_ref) if df_ref is not None else 0

    lines = [
        "# Relatório de Projeção de Novas Instâncias (PILOT — Seção 10)",
        "",
        "## 1. Visão Geral",
        f"- **Data da Execução:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Novas Instâncias Projetadas:** {n_new}",
        f"- **Instâncias de Referência (Dataset Base):** {n_ref}",
        f"- **Modo de Features de Probing:** {'Imputação pela média de referência (100% livre de execução de algoritmos)' if probing_imputed else 'Valores reais fornecidos via probing'}",
        "",
        "## 2. Estatísticas de Coordenadas Projetadas",
        "",
        "| Coordenada | Mínimo | Média | Mediana | Máximo | Desvio-Padrão |",
        "|---|---:|---:|---:|---:|---:|",
        f"| $Z_1$ (Novas) | {df_new['Z1'].min():.4f} | {df_new['Z1'].mean():.4f} | {df_new['Z1'].median():.4f} | {df_new['Z1'].max():.4f} | {df_new['Z1'].std():.4f} |",
        f"| $Z_2$ (Novas) | {df_new['Z2'].min():.4f} | {df_new['Z2'].mean():.4f} | {df_new['Z2'].median():.4f} | {df_new['Z2'].max():.4f} | {df_new['Z2'].std():.4f} |",
    ]

    if df_ref is not None:
        lines.extend([
            f"| $Z_1$ (Base) | {df_ref['Z1'].min():.4f} | {df_ref['Z1'].mean():.4f} | {df_ref['Z1'].median():.4f} | {df_ref['Z1'].max():.4f} | {df_ref['Z1'].std():.4f} |",
            f"| $Z_2$ (Base) | {df_ref['Z2'].min():.4f} | {df_ref['Z2'].mean():.4f} | {df_ref['Z2'].median():.4f} | {df_ref['Z2'].max():.4f} | {df_ref['Z2'].std():.4f} |",
        ])

    lines.extend([
        "",
        "## 3. Tabela de Coordenadas Individuais",
        "",
        "| Instância | $Z_1$ | $Z_2$ |",
        "|---|---:|---:|",
    ])
    for inst_id, row in df_new.iterrows():
        lines.append(f"| `{inst_id}` | {row['Z1']:.4f} | {row['Z2']:.4f} |")

    lines.extend([
        "",
        "## 4. Alinhamento com a Seção 10 do instructions.md",
        "",
        "> Conforme o documento de especificação, as instâncias externas são transformadas utilizando exclusivamente os parâmetros pré-computados $(\\boldsymbol{\\mu}, \\boldsymbol{\\sigma})$ e a matriz de projeção linear $\\mathbf{A}_{\\text{std}}$ da Etapa 7 (PILOT). Nenhuma reotimização de SIFTED ou PILOT foi realizada, assegurando a comparabilidade geométrica direta no mesmo espaço bidimensional.",
        ""
    ])

    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Relatório de projeção externa salvo: {output_md}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="07_project_new_instances.py — Projeta novas instâncias no espaço PILOT fixado (Seção 10)"
    )
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="diretório com novos arquivos .vrp a projetar")
    parser.add_argument("--z-matrix", type=Path, default=Path("Z_matrix.json"),
                        help="arquivo Z_matrix.json da projeção PILOT (default: Z_matrix.json)")
    parser.add_argument("--ref-coords", type=Path, default=Path("projected_coords.parquet"),
                        help="arquivo com coordenadas do dataset de referência para overlay (default: projected_coords.parquet)")
    parser.add_argument("--probing-file", type=Path, default=None,
                        help="arquivo opcional com probing features para as novas instâncias")
    parser.add_argument("--output-coords", type=Path, default=Path("projected_new_instances.parquet"),
                        help="arquivo de saída parquet para as coordenadas (default: projected_new_instances.parquet)")
    parser.add_argument("--csv-output", type=Path, default=Path("projected_new_instances.csv"),
                        help="arquivo de saída CSV para as coordenadas (default: projected_new_instances.csv)")
    parser.add_argument("--output-plot", type=Path, default=Path("projected_new_instances_overlay.png"),
                        help="arquivo de saída PNG para o gráfico de sobreposição (default: projected_new_instances_overlay.png)")
    parser.add_argument("--output-report", type=Path, default=Path("projected_new_instances_report.md"),
                        help="relatório markdown de saída (default: projected_new_instances_report.md)")
    parser.add_argument("--workers", type=int, default=4,
                        help="número de workers para extração de features em paralelo (default: 4)")
    args = parser.parse_args()

    t0 = time.perf_counter()

    # 1. Carregar modelo PILOT
    if not args.z_matrix.exists():
        fallback = Path("outputs") / args.z_matrix.name
        if fallback.exists():
            args.z_matrix = fallback
        else:
            print(f"ERRO: Modelo PILOT {args.z_matrix} não encontrado. Execute 05_pilot.py primeiro.")
            return 1

    pilot_model = load_pilot_model(args.z_matrix)
    print(f"Modelo PILOT carregado de {args.z_matrix}: {len(pilot_model['features'])} features.")

    # 2. Localizar instâncias .vrp
    vrp_files = sorted(args.instances_dir.glob("*.vrp"))
    if not vrp_files:
        print(f"ERRO: Nenhuma instância .vrp encontrada em {args.instances_dir}")
        return 1
    print(f"Encontradas {len(vrp_files)} instâncias para projeção externa.")

    # 3. Extrair características estáticas
    print(f"Extraindo características estáticas com {args.workers} workers...")
    results = []
    if args.workers == 1:
        for vf in vrp_files:
            results.append(extract_instance_features(vf))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for feat_dict in executor.map(extract_instance_features, vrp_files):
                results.append(feat_dict)

    df_feats = pd.DataFrame(results).set_index("instancia_id")

    # 4. Carregar probing se disponível
    probing_df = None
    probing_imputed = True
    if args.probing_file and args.probing_file.exists():
        probing_df = pd.read_csv(args.probing_file).set_index("instancia_id")
        probing_imputed = False
        print(f"Probing features carregadas de {args.probing_file}.")
    else:
        print("NOTA: Probing features não fornecidas. Aplicando imputação neutra baseada na média da população de referência (Seção 10).")

    # 5. Projetar coordenadas (Z1, Z2)
    df_proj_new = project_instances(df_feats, pilot_model, probing_df)
    print(f"Projeção concluída para {len(df_proj_new)} instâncias.")

    # Gravar coordenadas
    args.output_coords.parent.mkdir(parents=True, exist_ok=True)
    df_proj_new.to_parquet(args.output_coords)
    print(f"Coordenadas gravadas em Parquet: {args.output_coords}")

    if args.csv_output:
        df_proj_new.to_csv(args.csv_output)
        print(f"Coordenadas gravadas em CSV: {args.csv_output}")

    # 6. Carregar coordenadas de referência se existirem
    df_ref = None
    if args.ref_coords.exists():
        df_ref = pd.read_parquet(args.ref_coords)
    else:
        fallback_ref = Path("outputs") / args.ref_coords.name
        if fallback_ref.exists():
            df_ref = pd.read_parquet(fallback_ref)
        elif Path("projected_coords.csv").exists():
            df_ref = pd.read_csv("projected_coords.csv").set_index("instancia_id")

    # 7. Gerar gráfico de overlay
    plot_projection_overlay(
        df_ref, df_proj_new, pilot_model, args.output_plot,
        title_suffix=args.instances_dir.name
    )

    # 8. Gerar relatório
    generate_projection_report(df_proj_new, df_ref, pilot_model, args.output_report, probing_imputed)

    elapsed = time.perf_counter() - t0
    print(f"Etapa 10 concluída com sucesso em {elapsed:.2f}s!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
