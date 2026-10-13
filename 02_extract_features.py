"""
02_extract_features.py
======================
Extração da Matriz de Características (F) para o CVRP — Instance Space Analysis (ISA)
Conforme especificação da Seção 4 do instructions.md.

Categorias de Features Extraídas:
  1. ND  — Distribuição de nós (ND1 a ND9)
  2. MST — Árvore geradora mínima (MST1 a MST3)
  3. G   — Geométricas (G1 a G5)
  4. NN  — Vizinhança mais próxima (NN1 a NN8)
  5. VRP — Específicas do CVRP (VRP1 a VRP5)
  6. P   — Probing features do traço do ILS (P1 a P11, integradas de ils_probing_median.csv)

Output:
  - F.parquet (e opcionalmente F.csv), indexada por instancia_id.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse.csgraph as csgraph
from scipy.spatial import ConvexHull
from scipy.stats import kurtosis, skew
from sklearn.cluster import DBSCAN
import vrplib


from cvrp_features import extract_instance_features




def main() -> int:
    parser = argparse.ArgumentParser(description="Extrator da Matriz de Caracteristicas F (ISA - Seção 4)")
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="diretório com os arquivos .vrp")
    parser.add_argument("--probing-file", type=Path, default=Path("ils_probing_median.csv"),
                        help="arquivo com as 11 probing features (P1-P11) do ILS (default: ils_probing_median.csv)")
    parser.add_argument("--output", type=Path, default=Path("F.parquet"),
                        help="arquivo parquet de saída para a matriz F (default: F.parquet)")
    parser.add_argument("--csv-output", type=Path, default=Path("F.csv"),
                        help="arquivo csv de saída opcional (default: F.csv)")
    parser.add_argument("--limit", type=int, default=None,
                        help="limite de instâncias para amostragem/teste")
    parser.add_argument("--workers", type=int, default=4,
                        help="número de workers em paralelo (default: 4)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    paths = sorted(args.instances_dir.glob("*.vrp"))[:args.limit]
    if not paths:
        print(f"ERRO: Nenhuma instância .vrp encontrada em {args.instances_dir}")
        return 1

    print(f"Extraindo características de {len(paths)} instâncias com {args.workers} workers...")

    results: list[dict] = []
    if args.workers == 1:
        for p in paths:
            results.append(extract_instance_features(p))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for feat_dict in executor.map(extract_instance_features, paths):
                results.append(feat_dict)

    df_static = pd.DataFrame(results).set_index("instancia_id")
    print(f"Features estáticas extraídas: {df_static.shape[1]} colunas para {len(df_static)} instâncias.")

    # Fusão com as 11 Probing Features (P1 a P11)
    existing_p = [c for c in df_static.columns if c.startswith("P")]
    if existing_p:
        df_static = df_static.drop(columns=existing_p)

    if args.probing_file.exists():
        print(f"Carregando probing features de {args.probing_file}...")
        df_probing = pd.read_csv(args.probing_file).set_index("instancia_id")
        p_cols = [c for c in df_probing.columns if c.startswith("P")]
        print(f"Colunas de probing detectadas: {p_cols}")
        df_final = df_static.join(df_probing[p_cols], how="left")
    else:
        # Tenta procurar na pasta outputs/ caso não esteja na raiz
        fallback_prob = Path("outputs") / args.probing_file.name
        if fallback_prob.exists():
            print(f"Carregando probing features de {fallback_prob} (fallback)...")
            df_probing = pd.read_csv(fallback_prob).set_index("instancia_id")
            p_cols = [c for c in df_probing.columns if c.startswith("P")]
            print(f"Colunas de probing detectadas: {p_cols}")
            df_final = df_static.join(df_probing[p_cols], how="left")
        else:
            print(f"AVISO: Arquivo de probing {args.probing_file} não encontrado. F será gravada sem as features P.")
            df_final = df_static

    # Gravar outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(args.output)
    print(f"Matriz F gravada em formato Parquet: {args.output} ({df_final.shape[0]} linhas x {df_final.shape[1]} features)")

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(args.csv_output)
        print(f"Matriz F gravada em formato CSV: {args.csv_output}")

    elapsed = time.perf_counter() - t0
    print(f"Concluído com sucesso em {elapsed:.2f}s!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
