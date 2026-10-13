"""
run_pipeline.py
===============
Orquestrador Mestre de Execução do Pipeline CVRP ISA (Instance Space Analysis).
Executa de ponta a ponta as etapas 0 a 6 (e opcionalmente 7) de instructions.md,
centralizando todos os artefatos de saída em um único diretório por execução (run).

Estrutura de Saída por Execução (ex: outputs/run_YYYYMMDD_HHMMSS/):
  - 00: input_validation.csv
  - 01: checkpoint_costs_raw.csv, checkpoint_costs_median.csv, ils_trace.csv, ils_probing_median.csv
  - 02: F.parquet, F.csv
  - 03: Y.parquet, Y.csv, prelim_summary.json, prelim_report.md, prelim_distribution.png
  - 04: F_sifted.parquet, F_sifted.csv, sifted_features.json, sifted_features.csv, sifted_clusters.png, sifted_report.md
  - 05: Z_matrix.json, Z_matrix.npy, projected_coords.parquet, projected_coords.csv, instance_space_pilot.png, pilot_report.md
  - 06: Y_checkpoints.parquet, Y_checkpoints.csv, checkpoint_overlays_algorithms.png,
        checkpoint_overlays_best_selection.png, checkpoint_evolution_summary.png, checkpoint_overlay_report.md, params_log.md
  - 07 (opcional): projected_external_instances.parquet, external_instances_overlay.png, external_instances_report.md
  - ML: ml_models_accuracy.json, pipeline_summary.md
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
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

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.svm import SVC


def run_command(cmd: list[str], step_name: str, log_file: Path | None = None) -> float:
    """Executa um comando de subprocesso com medição de tempo e tratamento de erro."""
    print("\n" + "=" * 75)
    print(f" ▶ INICIANDO: {step_name}")
    print(f"   Comando: {' '.join(cmd)}")
    print("=" * 75)
    t0 = time.perf_counter()

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")

    print(res.stdout)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- {step_name} ---\n")
            f.write(res.stdout)

    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"\n❌ ERRO na etapa '{step_name}' (Código {res.returncode}) após {elapsed:.2f}s!")
        sys.exit(res.returncode)

    print(f" ✔ CONCLUÍDO: {step_name} em {elapsed:.2f}s")
    return elapsed


def evaluate_machine_learning_models(output_dir: Path) -> dict:
    """
    Avalia a acurácia dos modelos de Machine Learning (SVM e Random Forest):
      1. Footprint individual por algoritmo (Y_a <= epsilon): SVM (RBF) vs RF no espaço (Z1, Z2) e em F_sifted.
      2. Algoritmo Vencedor / Best Selection (AAS): SVM (RBF) vs RF prevendo argmin(Y).
    """
    coords_path = output_dir / "projected_coords.parquet"
    y_path = output_dir / "Y.parquet"
    f_sifted_path = output_dir / "F_sifted.parquet"
    prelim_path = output_dir / "prelim_summary.json"

    if not (coords_path.exists() and y_path.exists() and f_sifted_path.exists()):
        print("Arquivos necessários para validação de ML não encontrados.")
        return {}

    df_coords = pd.read_parquet(coords_path)
    df_y = pd.read_parquet(y_path)
    df_f_sifted = pd.read_parquet(f_sifted_path)

    # Identificar instâncias comuns
    common_idx = df_coords.index.intersection(df_y.index).intersection(df_f_sifted.index)
    df_coords = df_coords.loc[common_idx]
    df_y = df_y.loc[common_idx]
    df_f_sifted = df_f_sifted.loc[common_idx]

    eps = 0.1717
    if prelim_path.exists():
        try:
            with open(prelim_path, "r", encoding="utf-8") as f:
                pj = json.load(f)
                eps = float(pj.get("chosen_epsilon", 0.1717))
        except Exception:
            pass

    n_samples = len(common_idx)
    cv = LeaveOneOut() if n_samples <= 30 else KFold(n_splits=5, shuffle=True, random_state=42)
    cv_desc = "Leave-One-Out (LOO)" if n_samples <= 30 else "5-Fold CV"

    ml_results = {
        "n_samples": n_samples,
        "cv_method": cv_desc,
        "epsilon": eps,
        "footprint_models": {},
        "best_selection_model": {}
    }

    print("\n" + "=" * 75)
    print(f" 🧠 AVALIAÇÃO DE MODELOS DE MACHINE LEARNING (ISA / AAS)")
    print(f"    Método de Validação Cruzada: {cv_desc} | Amostras: {n_samples} | ε = {eps:.4f}")
    print("=" * 75)

    print("\n--- 1. Modelos de Footprint por Algoritmo (Y <= ε) ---")
    for alg in df_y.columns:
        binary_label = (df_y[alg] <= eps).astype(int)
        n_pos = int(binary_label.sum())
        n_neg = n_samples - n_pos

        svm = SVC(kernel="rbf", C=1.0)
        svm_acc = float(cross_val_score(svm, df_coords[["Z1", "Z2"]], binary_label, cv=cv).mean())

        rf_2d = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_2d_acc = float(cross_val_score(rf_2d, df_coords[["Z1", "Z2"]], binary_label, cv=cv).mean())

        rf_fs = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_fs_acc = float(cross_val_score(rf_fs, df_f_sifted, binary_label, cv=cv).mean())

        ml_results["footprint_models"][alg] = {
            "good_count": n_pos,
            "bad_count": n_neg,
            "svm_2d_accuracy": svm_acc,
            "rf_2d_accuracy": rf_2d_acc,
            "rf_sifted_accuracy": rf_fs_acc,
        }
        print(f" Algoritmo {alg:<6} ({n_pos} boas / {n_neg} ruins):")
        print(f"   * SVM RBF em (Z1, Z2)    : {svm_acc*100:.1f}% acurácia")
        print(f"   * Random Forest em (Z1, Z2) : {rf_2d_acc*100:.1f}% acurácia")
        print(f"   * Random Forest em F_sifted : {rf_fs_acc*100:.1f}% acurácia")

    print("\n--- 2. Seleção do Melhor Algoritmo (Best Solver Selection / AAS) ---")
    best_algo = df_y.idxmin(axis=1)
    solver_counts = best_algo.value_counts().to_dict()

    svm_best = SVC(kernel="rbf", C=1.0)
    svm_best_acc = float(cross_val_score(svm_best, df_coords[["Z1", "Z2"]], best_algo, cv=cv).mean())

    rf_best = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_best_acc = float(cross_val_score(rf_best, df_coords[["Z1", "Z2"]], best_algo, cv=cv).mean())

    rf_best_fs = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_best_fs_acc = float(cross_val_score(rf_best_fs, df_f_sifted, best_algo, cv=cv).mean())

    ml_results["best_selection_model"] = {
        "class_distribution": solver_counts,
        "svm_2d_accuracy": svm_best_acc,
        "rf_2d_accuracy": rf_best_acc,
        "rf_sifted_accuracy": rf_best_fs_acc,
    }
    print(f" Distribuição de Vencedores: {solver_counts}")
    print(f"   * SVM RBF em (Z1, Z2)       : {svm_best_acc*100:.1f}% acurácia")
    print(f"   * Random Forest em (Z1, Z2) : {rf_best_acc*100:.1f}% acurácia")
    print(f"   * Random Forest em F_sifted : {rf_best_fs_acc*100:.1f}% acurácia")

    # Salvar em JSON
    out_json = output_dir / "ml_models_accuracy.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ml_results, f, indent=2)
    print(f"\n[OK] Resultados de acurácia de ML salvos em: {out_json}")

    return ml_results


def generate_pipeline_summary(
    output_dir: Path,
    timings: dict[str, float],
    args: argparse.Namespace,
    ml_results: dict
) -> None:
    """Gera um sumário executivo completo do pipeline em Markdown."""
    total_time = sum(timings.values())
    summary_path = output_dir / "pipeline_summary.md"

    md_lines = [
        "# Sumário Executivo — CVRP Instance Space Analysis (ISA)",
        "",
        f"- **Diretório da Execução (Run Repo):** `{output_dir.resolve()}`",
        f"- **Data da Execução:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Tempo Total de Execução do Pipeline:** `{total_time:.2f}s` ({total_time/60:.2f} min)",
        "",
        "## 1. Tempo de Execução por Etapa",
        "",
        "| Etapa | Script | Tempo (s) | Fração (%) |",
        "|---|---|---:|---:|",
    ]
    for step, t in timings.items():
        frac = (t / max(1e-6, total_time)) * 100
        md_lines.append(f"| {step} | `{timings_scripts.get(step, '')}` | {t:.2f}s | {frac:.1f}% |")
    md_lines.append(f"| **TOTAL** | — | **{total_time:.2f}s** | **100.0%** |")

    md_lines.extend([
        "",
        "## 2. Acurácia dos Modelos de Machine Learning (SVM e Random Forest)",
        "",
        "### Fundamentação Teórica e Significado Científico no ISA",
        "Na metodologia canônica de Instance Space Analysis (*Smith-Miles & Muñoz, 2023*; *Notice et al., 2025*; *Gouvêa et al., 2025*), avaliam-se duas tarefas essenciais de Aprendizado de Máquina:",
        "1. **Previsão de Footprint (Classificação Binária):** Para cada algoritmo individual, prevê se ele apresentará bom desempenho ($Y \\le \\varepsilon$) com base nas características da instância.",
        "2. **Seleção do Melhor Algoritmo (*Best Algorithm Selection* / AAS — Multi-classe):** Prevê diretamente qual meta-heurística (ALNS, ILS, GLS, TS) produzirá o menor custo para a instância.",
        "",
        "Quando os classificadores (especialmente o SVM RBF) atingem alta acurácia usando **exclusivamente as duas coordenadas projetadas $(Z_1, Z_2)$**, isso fornece três comprovações científicas centrais:",
        "- **Preservação de Informação:** O espaço 2D construído analiticamente pelo PILOT não perdeu a informação discriminatória contida nas características originais.",
        "- **Separabilidade Territorial:** As coordenadas $(Z_1, Z_2)$ conseguiram delimitar com nitidez as fronteiras e nichos de dominância de cada meta-heurística.",
        "- **Validação de AAS:** Permite que instâncias novas tenham o seu algoritmo ideal selecionado a custo computacional de inferência quase nulo ($O(1)$), dispensando a execução redundante de todo o portfólio.",
        "",
        f"- **Método de Validação Cruzada:** {ml_results.get('cv_method', 'N/A')}",
        f"- **Tolerância de Desempenho (ε):** `{ml_results.get('epsilon', 0.1717):.4f}`",
        "",
        "### A. Previsão de Footprint (Classificação Y <= ε)",
        "",
        "| Algoritmo | Boas (Y <= ε) | Ruins (Y > ε) | SVM RBF em (Z1, Z2) | RF em (Z1, Z2) | RF em F_sifted |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for alg, data in ml_results.get("footprint_models", {}).items():
        md_lines.append(
            f"| **{alg}** | {data['good_count']} | {data['bad_count']} | "
            f"**{data['svm_2d_accuracy']*100:.1f}%** | {data['rf_2d_accuracy']*100:.1f}% | {data['rf_sifted_accuracy']*100:.1f}% |"
        )

    best_data = ml_results.get("best_selection_model", {})
    md_lines.extend([
        "",
        "### B. Algoritmo Vencedor (Best Algorithm Selection / AAS)",
        "",
        f"- **Distribuição de Vencedores:** `{best_data.get('class_distribution', {})}`",
        f"- **Acurácia SVM RBF em (Z1, Z2):** **{best_data.get('svm_2d_accuracy', 0.0)*100:.1f}%**",
        f"- **Acurácia Random Forest em (Z1, Z2):** **{best_data.get('rf_2d_accuracy', 0.0)*100:.1f}%**",
        f"- **Acurácia Random Forest em F_sifted (10 features):** **{best_data.get('rf_sifted_accuracy', 0.0)*100:.1f}%**",
        "",
        "## 3. Painel de Visualizações e Artefatos Gráficos Gerados",
        "",
        "| Arquivo de Imagem | Conteúdo / Descrição | Finalidade no Estudo |",
        "|---|---|---|",
        "| `prelim_distribution.png` | Boxplots, histogramas e curvas ECDF da distribuição de Y | Calibração empírica da tolerância ε |",
        "| `sifted_clusters.png` | Agrupamento dos 10 clusters e matriz de correlação das features | Diagnóstico do estágio SIFTED |",
        "| `instance_space_pilot.png` | Espaço 2D PILOT (Z1, Z2) com biplot loadings e correlações | Mapa canônico de referência das instâncias |",
        "| `feature_distribution_maps.png` | Dispersão individual das top características no plano Z (Notice et al., 2025, Fig. 5) | Interpretabilidade física dos eixos PILOT |",
        "| `checkpoint_overlays_algorithms.png` | Matriz de dispersão de gaps por solver e checkpoint (25%-100%) | Rastreamento temporal de qualidade |",
        "| `checkpoint_overlays_best_selection.png` | Mapa discreto do solver vencedor por checkpoint com polígonos de footprint | Delimitação de envelopes temporais de domínio |",
        "| `checkpoint_overlays_best_selection_points.png` | Dispersão individual do solver vencedor sem polígonos | Visualização desobstruída de cada instância |",
        "| `checkpoint_evolution_summary.png` | Fatias agregadas de vitória e redução percentual de custo | Dinâmica temporal de dominância |",
        "",
        "## 4. Checklist de Artefatos Gerados no Repositório da Run",
        "",
        "Todos os arquivos abaixo foram centralizados neste diretório de execução:",
        "- `input_validation.csv` — Validação e pareamento das instâncias",
        "- `checkpoint_costs_raw.csv` / `checkpoint_costs_median.csv` — Custos observados nos 4 checkpoints",
        "- `ils_trace.csv` / `ils_probing_median.csv` — 11 features de probing dinâmico do ILS",
        "- `F.parquet` / `F.csv` — Matriz completa de características (63 colunas)",
        "- `Y.parquet` / `Y.csv` — Matriz de gaps relativos a 100% vs BKS",
        "- `prelim_distribution.png` / `prelim_report.md` — Diagnóstico de percentis de ε",
        "- `F_sifted.parquet` / `sifted_clusters.png` — 10 características selecionadas e clusters ortogonais",
        "- `Z_matrix.json` / `projected_coords.parquet` — Projeção PILOT e coordenadas 2D",
        "- `instance_space_pilot.png` — Mapa em 4 quadrantes com biplot loadings",
        "- `feature_distribution_maps.png` — Mapas de distribuição das características estruturais (Notice Fig. 5)",
        "- `checkpoint_overlays_algorithms.png` — Grade 4x3 de gap por algoritmo",
        "- `checkpoint_overlays_best_selection.png` — Evolução com polígonos de footprint (Convex Hull)",
        "- `checkpoint_overlays_best_selection_points.png` — Dispersão pura do melhor algoritmo sem polígonos",
        "- `checkpoint_evolution_summary.png` — Gráficos agregados de convergência",
        "- `footprint_metrics.csv` — Métricas formais de footprint (Area, Density, Purity) por checkpoint",
        "- `params_log.md` — Rascunho metodológico consolidado para o artigo científico",
        "- `ml_models_accuracy.json` — Métricas numéricas de validação cruzada dos modelos de ML",
    ])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\n[OK] Sumário executivo consolidado gravado em: {summary_path}")


timings_scripts = {
    "Etapa 0: Validação de Entradas": "00_validate_inputs.py",
    "Etapa 1: Benchmark e Probing": "01_run_algorithms.py",
    "Etapa 2: Extração de Features (F)": "02_extract_features.py",
    "Etapa 3: PRELIM (Gaps Y e ε)": "03_prelim.py",
    "Etapa 4: SIFTED (K=10 Features)": "04_sifted.py",
    "Etapa 5: PILOT (Projeção Z)": "05_pilot.py",
    "Etapa 6: Overlay de Checkpoints": "06_overlay_checkpoints.py",
    "Etapa 7: Projeção de Novas Instâncias": "07_project_new_instances.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="run_pipeline.py — Orquestrador Mestre de Execução do Pipeline CVRP ISA"
    )
    # Diretórios
    parser.add_argument("--instances-dir", type=Path, default=Path("test_instances"),
                        help="Diretório contendo instâncias .vrp e .sol (padrão: test_instances)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Diretório único de saída para todos os artefatos desta run (padrão: outputs/run_YYYYMMDD_HHMMSS)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Nome customizado para a run (ex: test10, benchmark_10k). Se fornecido, pasta será outputs/<run-name>")
    
    # Controle de etapas
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Pula as Etapas 0 e 1 (reutilizando custos e probing pré-computados)")
    parser.add_argument("--existing-costs-dir", type=Path, default=None,
                        help="Diretório contendo checkpoint_costs_median.csv e ils_probing_median.csv para reaproveitar quando --skip-benchmark for usado")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limite de instâncias para amostragem ou teste rápido")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="retoma etapas a partir de resultados intermediários existentes (default: True)")

    # Parâmetros de Benchmark (Etapa 1)
    parser.add_argument("--algorithms", nargs="+", default=["ALNS", "ILS", "GLS", "TS"],
                        help="Algoritmos a comparar (padrão: ALNS ILS GLS TS)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1001, 2001, 3001],
                        help="Sementes pseudoaleatórias (padrão: 1001 2001 3001)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Número de workers paralelos (padrão: 4)")
    parser.add_argument("--seconds-per-customer", type=float, default=0.5,
                        help="Taxa de tempo por cliente (padrão: 0.5s)")
    parser.add_argument("--time-cap", type=float, default=100.0,
                        help="Teto máximo de tempo por instância em segundos (padrão: 100.0s)")

    # Parâmetros de ISA (Etapas 3 a 5)
    parser.add_argument("--epsilon", type=str, default="auto",
                        help="Tolerância ε de desempenho no PRELIM ('auto' = mediana canônica MATILDA, ou valor float)")
    parser.add_argument("--k-clusters", type=int, default=10,
                        help="Número fixo K de clusters para o SIFTED (padrão canônico: 10)")
    parser.add_argument("--n-try", type=int, default=30,
                        help="Número de rotações avaliadas no PILOT (padrão: 30)")

    # Extensibilidade (Etapa 7 opcional)
    parser.add_argument("--external-instances-dir", type=Path, default=None,
                        help="Diretório opcional com novas instâncias para projetar no espaço fixado (ex: Set X de Uchoa)")

    args = parser.parse_args()

    # 1. Definir repositório único de saída da run
    if args.output_dir is not None:
        out_dir = args.output_dir
    elif args.run_name is not None:
        out_dir = Path("outputs") / args.run_name
    else:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"run_{timestamp_str}"

    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "pipeline_execution.log"

    print("=" * 75)
    print(" 🚀 ORQUESTRADOR MESTRE DO PIPELINE CVRP ISA")
    print(f"    Repositório Único da Run : {out_dir.resolve()}")
    print(f"    Instâncias Base          : {args.instances_dir.resolve()}")
    print(f"    Algoritmos Avaliados     : {args.algorithms}")
    print(f"    Pular Benchmark          : {args.skip_benchmark}")
    print("=" * 75)

    timings: dict[str, float] = {}
    py_exec = sys.executable

    # -----------------------------------------------------------------------
    # Etapa 0: Validação de Entradas
    # -----------------------------------------------------------------------
    report_file = out_dir / "input_validation.csv"
    if not args.skip_benchmark:
        cmd_0 = [
            py_exec, "00_validate_inputs.py",
            "--instances-dir", str(args.instances_dir),
            "--report", str(report_file),
        ]
        timings["Etapa 0: Validação de Entradas"] = run_command(cmd_0, "00_validate_inputs.py", log_file)
    else:
        print("\n⏩ Pulando Etapa 0 (Benchmark previamente executado)")

    # -----------------------------------------------------------------------
    # Etapa 1: Execução dos Algoritmos (Benchmark e Probing)
    # -----------------------------------------------------------------------
    raw_costs = out_dir / "checkpoint_costs_raw.csv"
    median_costs = out_dir / "checkpoint_costs_median.csv"
    ils_trace = out_dir / "ils_trace.csv"
    ils_median = out_dir / "ils_probing_median.csv"

    if not args.skip_benchmark:
        cmd_1 = [
            py_exec, "01_run_algorithms.py",
            "--instances-dir", str(args.instances_dir),
            "--output", str(raw_costs),
            "--median-output", str(median_costs),
            "--ils-trace-output", str(ils_trace),
            "--ils-median-output", str(ils_median),
            "--algorithms", *args.algorithms,
            "--seeds", *[str(s) for s in args.seeds],
            "--workers", str(args.workers),
            "--seconds-per-customer", str(args.seconds_per_customer),
            "--time-cap", str(args.time_cap),
        ]
        if args.limit:
            cmd_1.extend(["--limit", str(args.limit)])
        if args.resume:
            cmd_1.append("--resume")
        else:
            cmd_1.append("--no-resume")
        timings["Etapa 1: Benchmark e Probing"] = run_command(cmd_1, "01_run_algorithms.py", log_file)
    else:
        print("\n⏩ Pulando Etapa 1 (Reutilizando custos pré-computados)")
        # Se os arquivos não estiverem em out_dir, copiar do diretório fornecido ou da raiz
        source_dir = args.existing_costs_dir or Path(".")
        for fname in ["checkpoint_costs_raw.csv", "checkpoint_costs_median.csv", "ils_trace.csv", "ils_probing_median.csv"]:
            src = source_dir / fname
            dst = out_dir / fname
            if not dst.exists():
                if src.exists():
                    shutil.copy(src, dst)
                    print(f"   [Copiado] {src} -> {dst}")
                elif (Path("outputs") / fname).exists():
                    shutil.copy(Path("outputs") / fname, dst)
                    print(f"   [Copiado de outputs/] {Path('outputs') / fname} -> {dst}")

    # -----------------------------------------------------------------------
    # Etapa 2: Extração de Características (F)
    # -----------------------------------------------------------------------
    f_parquet = out_dir / "F.parquet"
    f_csv = out_dir / "F.csv"
    cmd_2 = [
        py_exec, "02_extract_features.py",
        "--instances-dir", str(args.instances_dir),
        "--probing-file", str(ils_median),
        "--output", str(f_parquet),
        "--csv-output", str(f_csv),
        "--workers", str(args.workers),
    ]
    if args.limit:
        cmd_2.extend(["--limit", str(args.limit)])
    timings["Etapa 2: Extração de Features (F)"] = run_command(cmd_2, "02_extract_features.py", log_file)

    # -----------------------------------------------------------------------
    # Etapa 3: PRELIM (Matriz de Gaps Y e Ponto de Decisão ε)
    # -----------------------------------------------------------------------
    y_parquet = out_dir / "Y.parquet"
    y_csv = out_dir / "Y.csv"
    prelim_json = out_dir / "prelim_summary.json"
    prelim_md = out_dir / "prelim_report.md"
    prelim_plot = out_dir / "prelim_distribution.png"

    cmd_3 = [
        py_exec, "03_prelim.py",
        "--features", str(f_parquet),
        "--costs", str(median_costs),
        "--instances-dir", str(args.instances_dir),
        "--epsilon", str(args.epsilon),
        "--out-y-parquet", str(y_parquet),
        "--out-y-csv", str(y_csv),
        "--out-report-json", str(prelim_json),
        "--out-report-md", str(prelim_md),
        "--out-plot", str(prelim_plot),
    ]
    if report_file.exists():
        cmd_3.extend(["--bks-file", str(report_file)])
    timings["Etapa 3: PRELIM (Gaps Y e ε)"] = run_command(cmd_3, "03_prelim.py", log_file)

    # -----------------------------------------------------------------------
    # Etapa 4: SIFTED (Seleção de K=10 Features)
    # -----------------------------------------------------------------------
    f_sifted_parquet = out_dir / "F_sifted.parquet"
    f_sifted_csv = out_dir / "F_sifted.csv"
    sifted_json = out_dir / "sifted_features.json"
    sifted_csv = out_dir / "sifted_features.csv"
    sifted_md = out_dir / "sifted_report.md"
    sifted_plot = out_dir / "sifted_clusters.png"

    cmd_4 = [
        py_exec, "04_sifted.py",
        "--features", str(f_parquet),
        "--performance", str(y_parquet),
        "--k-clusters", str(args.k_clusters),
        "--out-f-sifted-parquet", str(f_sifted_parquet),
        "--out-f-sifted-csv", str(f_sifted_csv),
        "--out-selected-json", str(sifted_json),
        "--out-selected-csv", str(sifted_csv),
        "--out-report-md", str(sifted_md),
        "--out-plot", str(sifted_plot),
    ]
    timings["Etapa 4: SIFTED (K=10 Features)"] = run_command(cmd_4, "04_sifted.py", log_file)

    # -----------------------------------------------------------------------
    # Etapa 5: PILOT (Projeção Linear Analítica Z1, Z2)
    # -----------------------------------------------------------------------
    coords_parquet = out_dir / "projected_coords.parquet"
    coords_csv = out_dir / "projected_coords.csv"
    z_json = out_dir / "Z_matrix.json"
    z_npy = out_dir / "Z_matrix.npy"
    pilot_md = out_dir / "pilot_report.md"
    pilot_plot = out_dir / "instance_space_pilot.png"
    feature_maps_plot = out_dir / "feature_distribution_maps.png"

    cmd_5 = [
        py_exec, "05_pilot.py",
        "--features", str(f_sifted_parquet),
        "--performance", str(y_parquet),
        "--n-try", str(args.n_try),
        "--out-coords-parquet", str(coords_parquet),
        "--out-coords-csv", str(coords_csv),
        "--out-z-json", str(z_json),
        "--out-z-npy", str(z_npy),
        "--out-report-md", str(pilot_md),
        "--out-plot", str(pilot_plot),
        "--out-feature-maps", str(feature_maps_plot),
    ]
    timings["Etapa 5: PILOT (Projeção Z)"] = run_command(cmd_5, "05_pilot.py", log_file)

    # -----------------------------------------------------------------------
    # Etapa 6: Overlay de Checkpoints e Evolução Temporal
    # -----------------------------------------------------------------------
    cp_parquet = out_dir / "Y_checkpoints.parquet"
    cp_csv = out_dir / "Y_checkpoints.csv"
    footprint_csv = out_dir / "footprint_metrics.csv"
    cp_plot_algs = out_dir / "checkpoint_overlays_algorithms.png"
    cp_plot_best = out_dir / "checkpoint_overlays_best_selection.png"
    cp_plot_best_points = out_dir / "checkpoint_overlays_best_selection_points.png"
    cp_plot_summary = out_dir / "checkpoint_evolution_summary.png"
    cp_report_md = out_dir / "checkpoint_overlay_report.md"
    params_log_md = out_dir / "params_log.md"

    cmd_6 = [
        py_exec, "06_overlay_checkpoints.py",
        "--coords", str(coords_parquet),
        "--costs", str(median_costs),
        "--instances-dir", str(args.instances_dir),
        "--out-checkpoints-parquet", str(cp_parquet),
        "--out-checkpoints-csv", str(cp_csv),
        "--out-footprint-metrics", str(footprint_csv),
        "--out-plot-algorithms", str(cp_plot_algs),
        "--out-plot-best", str(cp_plot_best),
        "--out-plot-best-points", str(cp_plot_best_points),
        "--out-plot-summary", str(cp_plot_summary),
        "--out-report-md", str(cp_report_md),
        "--out-params-log", str(params_log_md),
    ]
    if report_file.exists():
        cmd_6.extend(["--bks-file", str(report_file)])
    timings["Etapa 6: Overlay de Checkpoints"] = run_command(cmd_6, "06_overlay_checkpoints.py", log_file)

    # -----------------------------------------------------------------------
    # Etapa 7: (Opcional) Projeção de Novas Instâncias Externas
    # -----------------------------------------------------------------------
    if args.external_instances_dir and args.external_instances_dir.exists():
        ext_coords_parquet = out_dir / "projected_external_instances.parquet"
        ext_coords_csv = out_dir / "projected_external_instances.csv"
        ext_plot = out_dir / "external_instances_overlay.png"
        ext_report = out_dir / "external_instances_report.md"

        cmd_7 = [
            py_exec, "07_project_new_instances.py",
            "--instances-dir", str(args.external_instances_dir),
            "--z-matrix", str(z_json),
            "--ref-coords", str(coords_parquet),
            "--output-coords", str(ext_coords_parquet),
            "--csv-output", str(ext_coords_csv),
            "--output-plot", str(ext_plot),
            "--output-report", str(ext_report),
            "--workers", str(args.workers),
        ]
        timings["Etapa 7: Projeção de Novas Instâncias"] = run_command(cmd_7, "07_project_new_instances.py", log_file)

    # -----------------------------------------------------------------------
    # Avaliação de Modelos de Machine Learning & Sumário Consolidado
    # -----------------------------------------------------------------------
    ml_results = evaluate_machine_learning_models(out_dir)
    generate_pipeline_summary(out_dir, timings, args, ml_results)

    print("\n" + "=" * 75)
    print(" 🎉 PIPELINE CVRP ISA CONCLUÍDO COM SUCESSO!")
    print(f" 📂 Todos os artefatos foram gravados em: {out_dir.resolve()}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())
