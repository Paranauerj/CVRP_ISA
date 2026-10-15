"""
04_sifted.py
============
Etapa 6: SIFTED — Seleção Inteligente de Características para Explicar a Dificuldade
Conforme especificação da Seção 6 do instructions.md e metodologia canônica do MATILDA (Smith-Miles & Muñoz, 2023).

Pipeline Metodológico:
  1. Filtro de Correlação de Pearson:
     - Calcula a correlação de Pearson |r_{f, Y_a}| entre cada feature f e o gap Y_a de cada algoritmo.
     - Mantém apenas features com correlação máxima |r| >= threshold (padrão: 0.5) com pelo menos um algoritmo.
  2. Agrupamento das Features em K Clusters Disjuntos:
     - Calcula a matriz de correlação cruzada R_{i, j} e a matriz de dissimilaridade D_{i, j} = 1 - |R_{i, j}|.
     - Aplica K-means clustering com K = 10 fixo (valor default deliberado do toolkit MATILDA de Smith-Miles & Muñoz).
  3. Busca da Combinação Ótima (1 Feature por Cluster):
     - Gera combinações selecionando 1 feature de cada um dos K clusters.
     - Para cada combinação:
       a) Padroniza e projeta temporariamente as K features em 2D via PCA.
       b) Treina um Random Forest Regressor para cada algoritmo a avaliando a predição de Y_a.
       c) Calcula o erro Out-Of-Bag (OOB MSE) de cada modelo.
     - Seleciona a combinação que minimiza o erro OOB médio entre os algoritmos do portfólio.
  4. Exportação dos Resultados:
     - sifted_features.json: metadados, clusters, correlações e métricas OOB.
     - sifted_features.csv: tabela detalhada das features candidatas e selecionadas.
     - F_sifted.parquet / F_sifted.csv: submatriz de F contendo estritamente as ~10 features selecionadas.
     - sifted_clusters.png: painel diagnóstico com correlações, clusters e avaliação OOB.
     - sifted_report.md: relatório técnico completo.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Funções Centrais do SIFTED
# ---------------------------------------------------------------------------

def filter_features_by_correlation(
    df_f: pd.DataFrame,
    df_y: pd.DataFrame,
    corr_threshold: float = 0.5,
    min_features: int = 10,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """
    Filtra features com correlação de Pearson |r| >= corr_threshold com pelo menos um algoritmo.
    Garante que pelo menos min_features sejam mantidas (relaxando o limiar se necessário).
    """
    corr_table: dict[str, dict[str, float]] = {}

    for col in df_f.columns:
        s_f = df_f[col]
        # Pular features constantes ou com desvio zero/NaN
        if s_f.std() == 0 or np.isnan(s_f.std()):
            continue

        c_dict: dict[str, float] = {}
        for alg in df_y.columns:
            s_y = df_y[alg]
            if s_y.std() > 0:
                r_val = float(s_f.corr(s_y))
                if not np.isnan(r_val):
                    c_dict[alg] = r_val

        if c_dict:
            max_abs_r = max(abs(v) for v in c_dict.values())
            c_dict["max_abs"] = max_abs_r
            corr_table[col] = c_dict

    # Ordenar por maior correlação absoluta
    sorted_features = sorted(corr_table.keys(), key=lambda k: corr_table[k]["max_abs"], reverse=True)

    # Filtrar pelo limiar
    filtered = [f for f in sorted_features if corr_table[f]["max_abs"] >= corr_threshold]

    # Se menos que min_features passarem pelo filtro, manter as top min_features
    if len(filtered) < min_features:
        print(f"[AVISO] Apenas {len(filtered)} features atingiram |r| >= {corr_threshold}. "
              f"Expandindo para as top {min_features} features mais correlacionadas.")
        filtered = sorted_features[:min_features]

    # Salvaguarda se nenhuma feature tiver correlação válida (ex.: amostra pontual)
    if not filtered:
        available = [c for c in df_f.columns if c != "instancia_id"]
        filtered = available[:min_features]
        for f in filtered:
            corr_table.setdefault(f, {"max_abs": 0.0})

    return filtered, corr_table


def cluster_features(
    df_f: pd.DataFrame,
    feature_list: list[str],
    k_clusters: int = 10,
    random_state: int = 42,
) -> tuple[dict[int, list[str]], np.ndarray]:
    """
    Agrupa as features em K clusters utilizando K-Means sobre a matriz de dissimilaridade D = 1 - |R|.
    """
    sub_f = df_f[feature_list]
    r_matrix = sub_f.corr().abs().values
    d_matrix = 1.0 - r_matrix
    np.fill_diagonal(d_matrix, 0.0)

    # Tratar eventuais NaNs em correlações pontuais
    d_matrix = np.nan_to_num(d_matrix, nan=1.0)

    k_eff = max(1, min(k_clusters, len(feature_list)))
    kmeans = KMeans(n_clusters=k_eff, n_init=30, max_iter=300, random_state=random_state)
    labels = kmeans.fit_predict(d_matrix)

    clusters: dict[int, list[str]] = {}
    for feat, lbl in zip(feature_list, labels):
        clusters.setdefault(int(lbl), []).append(feat)

    # Reindexar clusters de 0 a K-1 ordenando por tamanho decrescente
    sorted_cluster_ids = sorted(clusters.keys(), key=lambda cid: len(clusters[cid]), reverse=True)
    reindexed_clusters: dict[int, list[str]] = {
        new_id: clusters[old_id] for new_id, old_id in enumerate(sorted_cluster_ids)
    }

    return reindexed_clusters, d_matrix


def evaluate_feature_combinations(
    df_f: pd.DataFrame,
    df_y: pd.DataFrame,
    clusters: dict[int, list[str]],
    corr_table: dict[str, dict[str, float]],
    max_combinations: int = 1000,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[list[str], float, dict[str, float], dict[str, float], list[dict]]:
    """
    Avalia combinações (1 feature por cluster) projetando em 2D via PCA e medindo o erro OOB via Random Forest.
    Retorna a melhor combinação, erro OOB médio, métricas por algoritmo e histórico das avaliações.
    """
    cluster_ids = sorted(clusters.keys())

    # Ordenar features dentro de cada cluster pela correlação máxima com Y
    cluster_options = [
        sorted(clusters[cid], key=lambda f: corr_table[f]["max_abs"], reverse=True)
        for cid in cluster_ids
    ]

    total_possible = math.prod(len(opts) for opts in cluster_options)
    print(f" Espaço de combinações possíveis: {total_possible}")

    if total_possible <= max_combinations:
        combinations = list(itertools.product(*cluster_options))
    else:
        print(f"[INFO] Espaço de combinações ({total_possible}) excede o limite ({max_combinations}).")
        print(f" Priorizando as features mais correlacionadas de cada cluster...")
        # Restringir opções por cluster até caber no limite
        reduced_options = []
        for opts in cluster_options:
            reduced_options.append(opts[:min(2, len(opts))])
        combinations = list(itertools.product(*reduced_options))
        if len(combinations) > max_combinations:
            rng = np.random.default_rng(random_state)
            indices = rng.choice(len(combinations), size=max_combinations, replace=False)
            combinations = [combinations[i] for i in indices]

    print(f" Avaliando {len(combinations)} combinações via PCA (2D) + Random Forest (OOB)...")

    scaler = StandardScaler()
    best_error = float("inf")
    best_combo: list[str] = []
    best_oob_mse: dict[str, float] = {}
    best_oob_r2: dict[str, float] = {}
    eval_history: list[dict] = []

    algs = list(df_y.columns)

    t0 = time.time()
    for idx, combo in enumerate(combinations):
        combo_feats = list(combo)
        X_sub = df_f[combo_feats].values

        # Tratar NaNs caso existam
        X_sub = np.nan_to_num(X_sub, nan=0.0)

        # Padronização e Projeção temporária em 2D via PCA
        X_scaled = scaler.fit_transform(X_sub)
        X_2d = PCA(n_components=2, random_state=random_state).fit_transform(X_scaled)

        mse_by_alg: dict[str, float] = {}
        r2_by_alg: dict[str, float] = {}

        for alg in algs:
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                oob_score=True,
                random_state=random_state,
                n_jobs=-1,
            )
            rf.fit(X_2d, df_y[alg].values)

            # Extração de previsões OOB
            oob_pred = rf.oob_prediction_
            valid = ~np.isnan(oob_pred)
            if np.any(valid):
                mse = float(np.mean((df_y[alg].values[valid] - oob_pred[valid]) ** 2))
                r2 = float(rf.oob_score_)
            else:
                mse = float("inf")
                r2 = -1.0

            mse_by_alg[alg] = mse
            r2_by_alg[alg] = r2

        mean_mse = float(np.mean(list(mse_by_alg.values())))
        eval_history.append({
            "combo_index": idx,
            "mean_oob_mse": mean_mse,
            "mean_oob_r2": float(np.mean(list(r2_by_alg.values()))),
            "features": combo_feats,
        })

        if mean_mse < best_error:
            best_error = mean_mse
            best_combo = combo_feats
            best_oob_mse = mse_by_alg
            best_oob_r2 = r2_by_alg

    elapsed = time.time() - t0
    print(f" Concluído em {elapsed:.2f}s! Melhor erro OOB MSE: {best_error:.6f}")

    return best_combo, best_error, best_oob_mse, best_oob_r2, eval_history


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def generate_sifted_plots(
    df_f: pd.DataFrame,
    df_y: pd.DataFrame,
    feature_list: list[str],
    clusters: dict[int, list[str]],
    selected_features: list[str],
    eval_history: list[dict],
    best_oob_mse: dict[str, float],
    best_oob_r2: dict[str, float],
    output_png: Path,
) -> None:
    """Gera painel visual de 4 quadrantes para a etapa SIFTED."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("SIFTED — Seleção Inteligente de Features via Correlação, K-Means e OOB", fontsize=15, fontweight="bold")

    # 1. Matriz de Correlação das Features Filtradas (Superior Esquerdo)
    ax1 = axes[0, 0]
    sub_corr = df_f[feature_list].corr().abs().values
    im = ax1.imshow(sub_corr, cmap="YlGnBu", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="|Pearson r|")
    ax1.set_xticks(range(len(feature_list)))
    ax1.set_yticks(range(len(feature_list)))
    ax1.set_xticklabels(feature_list, rotation=90, fontsize=7)
    ax1.set_yticklabels(feature_list, fontsize=7)
    ax1.set_title(f"Matriz de Correlação Cruzada (|r|) ({len(feature_list)} features filtradas)", fontweight="bold")

    # 2. Distribuição das Features por Cluster e Selecionadas (Superior Direito)
    ax2 = axes[0, 1]
    cluster_labels = []
    cluster_sizes = []
    selected_markers = []
    for cid, feats in clusters.items():
        cluster_labels.append(f"C{cid}")
        cluster_sizes.append(len(feats))
        # Encontrar qual foi a selecionada
        sel = [f for f in feats if f in selected_features]
        selected_markers.append(sel[0] if sel else "None")

    x = np.arange(len(clusters))
    bars = ax2.bar(x, cluster_sizes, color="#4a90e2", edgecolor="black", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Cluster {i}\n({selected_markers[i]})" for i in x], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Quantidade de Features no Cluster")
    ax2.set_title("Composição dos Clusters (K=10) e Features Selecionadas", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Destacar no topo da barra a feature selecionada
    for i, bar in enumerate(bars):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"★ {selected_markers[i]}", ha="center", va="bottom", fontsize=7, fontweight="bold", rotation=30)
    ax2.set_ylim(0, max(cluster_sizes) + 2.5)

    # 3. Distribuição / Comparação dos Erros OOB das Combinações Avaliadas (Inferior Esquerdo)
    ax3 = axes[1, 0]
    all_oob_errors = [entry["mean_oob_mse"] for entry in eval_history]
    min_err = min(all_oob_errors)
    n_combs = len(eval_history)

    if n_combs <= 6:
        comb_labels = [f"Comb. {i+1}" for i in range(n_combs)]
        bar_colors = ["#2ca02c" if abs(err - min_err) < 1e-7 else "#aec7e8" for err in all_oob_errors]
        b_plots = ax3.bar(comb_labels, all_oob_errors, color=bar_colors, edgecolor="black", width=0.45)
        for b, err in zip(b_plots, all_oob_errors):
            ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001, f"{err:.5f}",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax3.set_ylim(0, max(all_oob_errors) * 1.25)
        ax3.set_title(f"Comparativo de Erro OOB ({n_combs} combinações testadas)", fontweight="bold")
        ax3.set_ylabel("Erro OOB Médio (MSE)")
        ax3.axhline(min_err, color="red", linestyle="--", linewidth=1.5, label=f"Vencedora: MSE = {min_err:.5f}")
        ax3.legend(loc="upper right", fontsize=8.5)
    else:
        ax3.hist(all_oob_errors, bins=min(25, max(8, n_combs // 4)), color="#7fcdbb", edgecolor="black", alpha=0.8)
        ax3.axvline(min_err, color="red", linestyle="--", linewidth=2.5, label=f"Vencedora: OOB MSE = {min_err:.5f}")
        ax3.set_title(f"Distribuição do Erro OOB Médio ({n_combs} combinações testadas)", fontweight="bold")
        ax3.set_xlabel("Erro OOB Médio (MSE)")
        ax3.set_ylabel("Frequência de Combinações")
        ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")

    # 4. Desempenho Preditivo da Combinação Vencedora por Algoritmo (Inferior Direito)
    ax4 = axes[1, 1]
    algs = list(df_y.columns)
    x_algs = np.arange(len(algs))
    width = 0.35

    mse_vals = [best_oob_mse[a] for a in algs]
    r2_vals = [best_oob_r2[a] for a in algs]

    ax4_twin = ax4.twinx()
    bars1 = ax4.bar(x_algs - width / 2, mse_vals, width, label="OOB MSE (menor é melhor)", color="#fc8d59", edgecolor="black")
    bars2 = ax4_twin.bar(x_algs + width / 2, r2_vals, width, label="OOB R² (maior é melhor)", color="#91cf60", edgecolor="black")

    ax4.set_xticks(x_algs)
    ax4.set_xticklabels(algs, fontweight="bold")
    ax4.set_ylabel("OOB MSE", color="#d73027", fontweight="bold")
    ax4_twin.set_ylabel("OOB R²", color="#1a9850", fontweight="bold")
    ax4.set_title("Acurácia Preditiva da Combinação Selecionada por Algoritmo", fontweight="bold")
    # Juntar legendas com posicionamento superior limpo sem colisão com as barras
    max_mse = max(mse_vals) if mse_vals else 0.1
    ax4.set_ylim(0, max_mse * 1.4)
    min_r2 = min(r2_vals) if r2_vals else -0.1
    max_r2 = max(r2_vals) if r2_vals else 0.1
    span_r2 = max(0.1, max_r2 - min_r2)
    ax4_twin.set_ylim(min_r2 - 0.1 * span_r2, max_r2 + 0.35 * span_r2)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        fontsize=8.5,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#cccccc",
    )

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[OK] Painel SIFTED salvo em: {output_png}")


# ---------------------------------------------------------------------------
# Orquestração CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="04_sifted.py — Etapa 6: SIFTED Feature Selection (MATILDA / Instance Space Analysis)"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("F.parquet"),
        help="Caminho para a matriz de características F bruta. Padrão: F.parquet",
    )
    parser.add_argument(
        "--performance",
        type=Path,
        default=Path("Y.parquet"),
        help="Caminho para a matriz de desempenho Y (checkpoint 100%%). Padrão: Y.parquet",
    )
    parser.add_argument(
        "--k-clusters",
        type=int,
        default=10,
        help="Número fixo K de clusters para o K-means (padrão MATILDA: 10).",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.3,
        help="Limiar mínimo de correlação de Pearson |r| com pelo menos um algoritmo (padrão MATILDA: 0.3).",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=1000,
        help="Número máximo de combinações a avaliar exaustivamente. Padrão: 1000",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Número de árvores no Random Forest para cálculo de OOB. Padrão: 100",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semente pseudoaleatória para reprodutibilidade estrita. Padrão: 42",
    )
    parser.add_argument(
        "--out-f-sifted-parquet",
        type=Path,
        default=Path("F_sifted.parquet"),
        help="Arquivo Parquet com as ~10 features selecionadas. Padrão: F_sifted.parquet",
    )
    parser.add_argument(
        "--out-f-sifted-csv",
        type=Path,
        default=Path("F_sifted.csv"),
        help="Arquivo CSV com as ~10 features selecionadas. Padrão: F_sifted.csv",
    )
    parser.add_argument(
        "--out-selected-json",
        type=Path,
        default=Path("sifted_features.json"),
        help="Arquivo JSON com a lista de features e metadados. Padrão: sifted_features.json",
    )
    parser.add_argument(
        "--out-selected-csv",
        type=Path,
        default=Path("sifted_features.csv"),
        help="Arquivo CSV detalhando clusters e correlações. Padrão: sifted_features.csv",
    )
    parser.add_argument(
        "--out-report-md",
        type=Path,
        default=Path("sifted_report.md"),
        help="Relatório técnico em formato Markdown. Padrão: sifted_report.md",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=Path("sifted_clusters.png"),
        help="Painel gráfico da etapa SIFTED. Padrão: sifted_clusters.png",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Restringe o SIFTED apenas a características geométricas e estruturais estáticas, excluindo probing (P1-P11).",
    )

    args = parser.parse_args()

    # 1. Resolver caminhos de F e Y com fallback automático
    features_path = args.features
    if not features_path.is_file():
        alt_f = Path("outputs") / features_path.name
        if alt_f.is_file():
            features_path = alt_f
        elif Path("F.csv").is_file():
            features_path = Path("F.csv")

    perf_path = args.performance
    if not perf_path.is_file():
        alt_y = Path("outputs") / perf_path.name
        if alt_y.is_file():
            perf_path = alt_y
        elif Path("Y.csv").is_file():
            perf_path = Path("Y.csv")

    if not features_path.is_file():
        print(f"[ERRO] Matriz F não encontrada em {args.features} nem em {features_path}.")
        sys.exit(1)

    if not perf_path.is_file():
        print(f"[ERRO] Matriz Y não encontrada em {args.performance} nem em {perf_path}.")
        print("Execute primeiro a etapa 03: python 03_prelim.py")
        sys.exit(1)

    print("=" * 70)
    print(" 04_sifted.py — SIFTED: Seleção de Características Relevantes e Não-Redundantes")
    print("=" * 70)
    print(f" Matriz F : {features_path}")
    print(f" Matriz Y : {perf_path}")
    print(f" Limiar r : |r| >= {args.corr_threshold}")
    print(f" K fixo   : {args.k_clusters} clusters (padrão canônico MATILDA)")

    # 2. Carregar dados e alinhar instâncias
    df_f = pd.read_parquet(features_path) if features_path.suffix.lower() == ".parquet" else pd.read_csv(features_path, index_col=0)
    df_y = pd.read_parquet(perf_path) if perf_path.suffix.lower() == ".parquet" else pd.read_csv(perf_path, index_col=0)

    common_insts = sorted(list(df_f.index.intersection(df_y.index)))
    if len(common_insts) == 0:
        print("[ERRO] Nenhuma instância comum entre F e Y!")
        sys.exit(1)

    df_f = df_f.loc[common_insts]
    df_y = df_y.loc[common_insts]
    print(f" Instâncias analisadas: {len(common_insts)}")
    print(f" Algoritmos em Y     : {list(df_y.columns)}")
    if args.static_only:
        p_cols = [c for c in df_f.columns if c.startswith("P")]
        if p_cols:
            df_f = df_f.drop(columns=p_cols)
            print(f" [INFO] Modo --static-only ativado: {len(p_cols)} probing features removidas de F.")
    print(f" Features brutas em F: {df_f.shape[1]}")

    # 3. Passo 1: Filtro de Correlação de Pearson com Y (Canônico MATILDA |r| >= tau)
    filtered_features, corr_table = filter_features_by_correlation(
        df_f, df_y, corr_threshold=args.corr_threshold, min_features=args.k_clusters
    )
    print(f"\n[Passo 1] Features retidas com |r| >= {args.corr_threshold}: {len(filtered_features)} de {df_f.shape[1]}")
    for idx, feat in enumerate(filtered_features[:10], 1):
        c_info = corr_table[feat]
        print(f"   {idx:>2}. {feat:<28} : max |r| = {c_info['max_abs']:.4f}")
    if len(filtered_features) > 10:
        print(f"   ... e mais {len(filtered_features) - 10} features.")

    # 4. Passo 2: Agrupamento em K Clusters via K-Means
    clusters, d_matrix = cluster_features(
        df_f, filtered_features, k_clusters=args.k_clusters, random_state=args.random_state
    )
    print(f"\n[Passo 2] Agrupamento K-Means (K={len(clusters)}) sobre dissimilaridade 1 - |r|:")
    for cid, feats in clusters.items():
        print(f"   Cluster {cid:>2} ({len(feats)} features): {', '.join(feats)}")

    # 5. Passo 3: Avaliação de Combinações via PCA (2D) + Random Forest OOB
    best_combo, best_error, best_oob_mse, best_oob_r2, eval_history = evaluate_feature_combinations(
        df_f=df_f,
        df_y=df_y,
        clusters=clusters,
        corr_table=corr_table,
        max_combinations=args.max_combinations,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

    print("\n" + "=" * 70)
    print(f" COMBINAÇÃO VENCEDORA SELECIONADA ({len(best_combo)} FEATURES):")
    print("=" * 70)
    for idx, feat in enumerate(best_combo, 1):
        c_info = corr_table[feat]
        # Identificar cluster pertencente
        cid = [k for k, v in clusters.items() if feat in v][0]
        print(f" {idx:>2}. [Cluster {cid:>2}] {feat:<28} | max |r| = {c_info['max_abs']:.4f}")

    print("\n Desempenho Preditivo da Projeção 2D (OOB):")
    print(f" - Erro OOB Médio (MSE): {best_error:.6f}")
    for alg in df_y.columns:
        print(f"   * {alg:<6}: OOB MSE = {best_oob_mse[alg]:.6f} | OOB R² = {best_oob_r2[alg]:.4f}")

    # 6. Exportar submatriz F_sifted
    args.out_f_sifted_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.out_f_sifted_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_selected_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_selected_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_report_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_plot.parent.mkdir(parents=True, exist_ok=True)

    df_f_sifted = df_f[best_combo].copy()
    df_f_sifted.to_parquet(args.out_f_sifted_parquet)
    df_f_sifted.to_csv(args.out_f_sifted_csv)
    print(f"\n[OK] Matriz F_sifted salva em Parquet: {args.out_f_sifted_parquet} ({df_f_sifted.shape[0]}x{df_f_sifted.shape[1]})")
    print(f"[OK] Matriz F_sifted salva em CSV    : {args.out_f_sifted_csv}")

    # 7. Exportar Tabela CSV detalhada de Features
    table_rows = []
    for cid, feats in clusters.items():
        for feat in feats:
            c_info = corr_table[feat]
            row = {
                "cluster_id": cid,
                "feature": feat,
                "is_selected": feat in best_combo,
                "max_abs_corr": c_info["max_abs"],
            }
            for alg in df_y.columns:
                row[f"corr_{alg}"] = c_info.get(alg, np.nan)
            table_rows.append(row)
    df_table = pd.DataFrame(table_rows).sort_values(["cluster_id", "is_selected", "max_abs_corr"], ascending=[True, False, False])
    df_table.to_csv(args.out_selected_csv, index=False)
    print(f"[OK] Tabela detalhada de features salva em CSV: {args.out_selected_csv}")

    # 8. Exportar JSON de Metadados
    summary_json = {
        "n_instances": len(common_insts),
        "k_clusters": len(clusters),
        "corr_threshold": args.corr_threshold,
        "selected_features": best_combo,
        "best_mean_oob_mse": best_error,
        "per_algorithm_oob_mse": best_oob_mse,
        "per_algorithm_oob_r2": best_oob_r2,
        "clusters": {str(k): v for k, v in clusters.items()},
        "total_combinations_evaluated": len(eval_history),
        "methodology": {
            "canonical_reference": "Smith-Miles & Muñoz (2023), ACM Computing Surveys",
            "k_selection": "Fixed K=10 per canonical MATILDA standard (no Silhouette/DB/CH optimization)",
            "projection_during_sifted": "Temporary 2D PCA for Out-of-Bag validation",
            "model": "Random Forest Regressor (n_estimators=100) per algorithm predicting relative gap Y_a",
        },
    }
    with args.out_selected_json.open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"[OK] Metadados SIFTED salvos em JSON: {args.out_selected_json}")

    # 9. Gerar Painel Gráfico
    generate_sifted_plots(
        df_f=df_f,
        df_y=df_y,
        feature_list=filtered_features,
        clusters=clusters,
        selected_features=best_combo,
        eval_history=eval_history,
        best_oob_mse=best_oob_mse,
        best_oob_r2=best_oob_r2,
        output_png=args.out_plot,
    )

    # 10. Salvar Relatório Markdown
    md_content = f"""# SIFTED — Relatório de Seleção de Características (Etapa 6)

## 1. Metodologia e Fundamentação Teórica
O estágio **SIFTED** (*Selection of Instance Features to Explain Difficulty*) do framework **MATILDA** (Smith-Miles & Muñoz, 2023) reduz o espaço dimensional original eliminando redundâncias e selecionando um subconjunto ortogonal de características que melhor explicam as variações de desempenho dos algoritmos.

- **Filtro de Relevância:** Correlação linear de Pearson $|r_{{f, Y_a}}| \\ge {args.corr_threshold:.2f}$ com pelo menos um algoritmo do portfólio.
- **Controle de Redundância:** Agrupamento via $K$-Means sobre a matriz de dissimilaridade entre features ($D_{{i, j}} = 1 - |r_{{i, j}}|$).
- **Parâmetro $K$:** **$K = {len(clusters)}$ fixo**, seguindo rigorosamente o padrão deliberado do toolkit canônico (evitando sobreajuste via métricas internas como Silhouette, Davies-Bouldin ou Calinski-Harabasz).
- **Critério de Seleção Final:** Geração de combinações candidatas ($1$ feature por cluster), projeção temporária em 2D via PCA e treinamento de regressões Random Forest para prever $Y_a$. A combinação vencedora minimiza o erro Out-of-Bag médio ($\text{{OOB MSE}}$).

## 2. Resumo da Execução
- **Total de instâncias:** {len(common_insts)}
- **Features brutas candidatas:** {df_f.shape[1]}
- **Features aprovadas no filtro $|r| \\ge {args.corr_threshold:.2f}$:** {len(filtered_features)}
- **Número de clusters formados:** {len(clusters)}
- **Combinações avaliadas via OOB:** {len(eval_history)}
- **Erro OOB Médio da Vencedora (MSE):** `{best_error:.6f}`

## 3. Características Finais Selecionadas ($K={len(best_combo)}$)

| # | Cluster | Feature Selecionada | Família | Max $\|r\|$ com $Y$ |
|---|---|---|---|---|
"""
    for idx, feat in enumerate(best_combo, 1):
        cid = [k for k, v in clusters.items() if feat in v][0]
        c_info = corr_table[feat]
        family = feat.split("_")[0]
        md_content += f"| {idx} | Cluster {cid} | `{feat}` | {family} | {c_info['max_abs']:.4f} |\n"

    md_content += f"""
## 4. Acurácia Preditiva dos Modelos Random Forest por Algoritmo

| Algoritmo | OOB MSE (Menor é Melhor) | OOB $R^2$ (Maior é Melhor) |
|---|---|---|
"""
    for alg in df_y.columns:
        md_content += f"| **{alg}** | {best_oob_mse[alg]:.6f} | {best_oob_r2[alg]:.4f} |\n"

    md_content += f"""
## 5. Artefatos Produzidos
- `F_sifted.parquet` e `F_sifted.csv`: Submatriz $F$ pronta com as {len(best_combo)} features selecionadas.
- `sifted_features.json`: Metadados completos da seleção.
- `sifted_features.csv`: Tabela com todas as features candidatas, seus clusters e correlações.
- `sifted_clusters.png`: Painel gráfico de 4 quadrantes.

## 6. Próximo Passo
Avançar para a **Etapa 7 (PILOT — `05_pilot.py`)**, que receberá `F_sifted.parquet` e `Y.parquet` para calcular analiticamente a matriz de projeção $Z$ e as coordenadas definitivas $(Z_1, Z_2)$ do espaço de instâncias.
"""
    with args.out_report_md.open("w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] Relatório Markdown salvo em: {args.out_report_md}")

    print("\n" + "=" * 70)
    print(" Etapa 6 (SIFTED) CONCLUÍDA com sucesso!")
    print(f" Arquivo gerado para a Etapa 7: {args.out_f_sifted_parquet}")
    print("=" * 70)


if __name__ == "__main__":
    main()
