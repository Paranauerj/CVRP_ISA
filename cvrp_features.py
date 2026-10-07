"""
cvrp_features.py
================
Módulo independente e reutilizável para extração de características de instâncias CVRP.
Conforme especificação das Seções 4 e 10 do instructions.md.

Categorias de Features Extraídas:
  1. ND  — Distribuição de nós (ND1 a ND9)
  2. MST — Árvore geradora mínima (MST1 a MST3)
  3. G   — Geométricas (G1 a G5)
  4. NN  — Vizinhança mais próxima (NN1 a NN8)
  5. VRP — Específicas do CVRP (VRP1 a VRP5)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import scipy.sparse.csgraph as csgraph
from scipy.spatial import ConvexHull
from scipy.stats import kurtosis, skew
from sklearn.cluster import DBSCAN
import vrplib


def _pt_to_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Calcula a distância euclidiana mínima entre o ponto p e o segmento ab."""
    ab = b - a
    l2 = float(np.dot(ab, ab))
    if l2 < 1e-12:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / l2))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def extract_instance_features(vrp_path: Path) -> dict:
    """Extrai todas as features estáticas O(V^2) de uma instância CVRP."""
    inst_id = vrp_path.stem
    data = vrplib.read_instance(str(vrp_path))

    dist = np.asarray(data["edge_weight"], dtype=float)
    demand = np.asarray(data["demand"], dtype=float)
    capacity = float(data["capacity"])
    coords = np.asarray(data["node_coord"], dtype=float) if "node_coord" in data and data["node_coord"] is not None else None

    n_nodes = len(demand)          # Total de nós (depósito 0 + clientes 1..N)
    n_cust = n_nodes - 1           # Número de clientes N
    depot_idx = 0
    cust_indices = np.arange(1, n_nodes)

    features: dict[str, float] = {"instancia_id": inst_id, "N": float(n_cust)}

    # Triângulo superior das distâncias (pares não-direcionados sem auto-arestas)
    tri_i, tri_j = np.triu_indices(n_nodes, k=1)
    upper_dists = dist[tri_i, tri_j]

    # -----------------------------------------------------------------------
    # 1. ND — Distribuição de nós (ND1 a ND9)
    # -----------------------------------------------------------------------
    features["ND1_dist_mean"] = float(np.mean(upper_dists))
    features["ND1_dist_std"] = float(np.std(upper_dists))
    features["ND1_dist_median"] = float(np.median(upper_dists))
    features["ND1_dist_kurtosis"] = float(kurtosis(upper_dists)) if len(upper_dists) > 3 else 0.0

    # ND2: soma das arestas de menor custo para cada nó (excluindo a si mesmo)
    min_edges = []
    for i in range(n_nodes):
        row = dist[i].copy()
        row[i] = np.inf
        min_edges.append(np.min(row))
    features["ND2_min_edge_sum"] = float(np.sum(min_edges))

    # ND3: fração de distâncias distintas
    features["ND3_distinct_dist_ratio"] = float(len(np.unique(upper_dists)) / max(1, len(upper_dists)))

    # ND4 & ND5: centroide e dispersão dos clientes
    if coords is not None and len(coords) > 1:
        cust_coords = coords[cust_indices]
        c_x, c_y = float(np.mean(cust_coords[:, 0])), float(np.mean(cust_coords[:, 1]))
        features["ND4_centroid_x"] = c_x
        features["ND4_centroid_y"] = c_y

        dists_to_centroid = np.linalg.norm(cust_coords - np.array([c_x, c_y]), axis=1)
        features["ND5_dist_to_centroid_mean"] = float(np.mean(dists_to_centroid))
        features["ND5_dist_to_centroid_std"] = float(np.std(dists_to_centroid))

        # ND6 - ND9: Clustering adaptativo via DBSCAN
        eps_val = float(np.percentile(min_edges[1:], 75) * 1.5)
        eps_val = max(1.0, eps_val)
        db = DBSCAN(eps=eps_val, min_samples=3).fit(cust_coords)
        labels = db.labels_
        unique_clusters = [lbl for lbl in set(labels) if lbl != -1]
        n_clusters = len(unique_clusters)

        features["ND6_n_clusters"] = float(n_clusters)
        features["ND9_clusters_per_node"] = float(n_clusters / max(1, n_cust))

        if n_clusters > 0:
            c_sizes = [float(np.sum(labels == c_id)) for c_id in unique_clusters]
            features["ND7_cluster_size_mean"] = float(np.mean(c_sizes))
            features["ND7_cluster_size_std"] = float(np.std(c_sizes))
        else:
            features["ND7_cluster_size_mean"] = 0.0
            features["ND7_cluster_size_std"] = 0.0

        if n_clusters >= 2:
            centroids = np.array([np.mean(cust_coords[labels == c_id], axis=0) for c_id in unique_clusters])
            c_tri_i, c_tri_j = np.triu_indices(n_clusters, k=1)
            c_dists = np.linalg.norm(centroids[c_tri_i] - centroids[c_tri_j], axis=1)
            features["ND8_inter_cluster_dist_mean"] = float(np.mean(c_dists))
        else:
            features["ND8_inter_cluster_dist_mean"] = 0.0
    else:
        features["ND4_centroid_x"] = 0.0
        features["ND4_centroid_y"] = 0.0
        features["ND5_dist_to_centroid_mean"] = 0.0
        features["ND5_dist_to_centroid_std"] = 0.0
        features["ND6_n_clusters"] = 0.0
        features["ND7_cluster_size_mean"] = 0.0
        features["ND7_cluster_size_std"] = 0.0
        features["ND8_inter_cluster_dist_mean"] = 0.0
        features["ND9_clusters_per_node"] = 0.0

    # -----------------------------------------------------------------------
    # 2. MST — Árvore geradora mínima (MST1 a MST3)
    # -----------------------------------------------------------------------
    mst_csr = csgraph.minimum_spanning_tree(dist)
    mst_weights = mst_csr.data
    features["MST1_edge_cost_mean"] = float(np.mean(mst_weights)) if len(mst_weights) > 0 else 0.0
    features["MST1_edge_cost_std"] = float(np.std(mst_weights)) if len(mst_weights) > 0 else 0.0
    features["MST1_edge_cost_median"] = float(np.median(mst_weights)) if len(mst_weights) > 0 else 0.0
    features["MST1_edge_cost_sum"] = float(np.sum(mst_weights)) if len(mst_weights) > 0 else 0.0

    # MST2: graus dos nós na MST
    adj_undirected = (mst_csr + mst_csr.T > 0).astype(float)
    degrees = np.array(adj_undirected.sum(axis=1)).flatten()
    features["MST2_node_degree_mean"] = float(np.mean(degrees))
    features["MST2_node_degree_std"] = float(np.std(degrees))
    features["MST2_node_degree_max"] = float(np.max(degrees))

    # MST3: profundidade da MST a partir do depósito (hops)
    depths = csgraph.dijkstra(adj_undirected, indices=depot_idx, unweighted=True)
    features["MST3_tree_depth"] = float(np.max(depths[np.isfinite(depths)]))

    # -----------------------------------------------------------------------
    # 3. G — Geométricas (G1 a G5)
    # -----------------------------------------------------------------------
    if coords is not None and len(coords) >= 4:
        min_x, max_x = float(coords[:, 0].min()), float(coords[:, 0].max())
        min_y, max_y = float(coords[:, 1].min()), float(coords[:, 1].max())
        features["G1_bbox_area"] = float((max_x - min_x) * (max_y - min_y))

        hull = ConvexHull(coords)
        features["G2_hull_area"] = float(hull.volume)  # Em 2D, volume é a área do fecho
        features["G3_hull_points_ratio"] = float(len(hull.vertices) / max(1, len(coords)))

        # G4: distância dos pontos internos ao contorno do fecho convexo
        hull_set = set(hull.vertices)
        verts_ordered = hull.vertices
        hull_edges = [
            (coords[verts_ordered[i]], coords[verts_ordered[(i + 1) % len(verts_ordered)]])
            for i in range(len(verts_ordered))
        ]
        interior_points = [coords[i] for i in range(len(coords)) if i not in hull_set]
        if interior_points:
            int_dists = [min(_pt_to_seg_dist(p, a, b) for a, b in hull_edges) for p in interior_points]
            features["G4_interior_dist_mean"] = float(np.mean(int_dists))
            features["G4_interior_dist_std"] = float(np.std(int_dists))
        else:
            features["G4_interior_dist_mean"] = 0.0
            features["G4_interior_dist_std"] = 0.0

        # G5: comprimentos das arestas do fecho convexo
        edge_lens = [float(np.linalg.norm(a - b)) for a, b in hull_edges]
        features["G5_hull_edge_len_mean"] = float(np.mean(edge_lens))
        features["G5_hull_edge_len_std"] = float(np.std(edge_lens))
    else:
        features["G1_bbox_area"] = 0.0
        features["G2_hull_area"] = 0.0
        features["G3_hull_points_ratio"] = 0.0
        features["G4_interior_dist_mean"] = 0.0
        features["G4_interior_dist_std"] = 0.0
        features["G5_hull_edge_len_mean"] = 0.0
        features["G5_hull_edge_len_std"] = 0.0

    # -----------------------------------------------------------------------
    # 4. NN — Vizinhança mais próxima (NN1 a NN8)
    # -----------------------------------------------------------------------
    features["NN1_dist_1nn_mean"] = float(np.mean(min_edges))
    features["NN1_dist_1nn_std"] = float(np.std(min_edges))

    k_nn = min(3, n_nodes - 1)
    adj_knn = np.zeros((n_nodes, n_nodes), dtype=int)
    for i in range(n_nodes):
        row = dist[i].copy()
        row[i] = np.inf
        nn_indices = np.argsort(row)[:k_nn]
        adj_knn[i, nn_indices] = 1

    n_scc, scc_labels = csgraph.connected_components(adj_knn, connection="strong")
    n_wcc, wcc_labels = csgraph.connected_components(adj_knn, connection="weak")
    features["NN2_n_scc"] = float(n_scc)
    features["NN3_n_wcc"] = float(n_wcc)

    scc_sizes = np.bincount(scc_labels)
    wcc_sizes = np.bincount(wcc_labels)
    features["NN4_scc_size_mean"] = float(np.mean(scc_sizes))
    features["NN5_wcc_size_mean"] = float(np.mean(wcc_sizes))

    in_degrees = np.sum(adj_knn, axis=0)
    features["NN6_in_degree_mean"] = float(np.mean(in_degrees))
    features["NN6_in_degree_std"] = float(np.std(in_degrees))

    features["NN7_scc_to_wcc_ratio"] = float(n_scc / max(1, n_wcc))

    if coords is not None and len(coords) > 2:
        angles = []
        for i in range(n_nodes):
            row = dist[i].copy()
            row[i] = np.inf
            nn_2 = np.argsort(row)[:2]
            u = coords[nn_2[0]] - coords[i]
            v = coords[nn_2[1]] - coords[i]
            norm_u, norm_v = float(np.linalg.norm(u)), float(np.linalg.norm(v))
            if norm_u > 1e-9 and norm_v > 1e-9:
                cos_th = float(np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0))
                angles.append(np.arccos(cos_th))
        if angles:
            features["NN8_angle_mean"] = float(np.mean(angles))
            features["NN8_angle_std"] = float(np.std(angles))
        else:
            features["NN8_angle_mean"] = 0.0
            features["NN8_angle_std"] = 0.0
    else:
        features["NN8_angle_mean"] = 0.0
        features["NN8_angle_std"] = 0.0

    # -----------------------------------------------------------------------
    # 5. VRP — Específicas do problema (VRP1 a VRP5)
    # -----------------------------------------------------------------------
    if coords is not None:
        depot_coord = coords[depot_idx]
        cust_coords = coords[cust_indices]
        centroid = np.mean(cust_coords, axis=0)
        features["VRP1_centroid_to_depot"] = float(np.linalg.norm(centroid - depot_coord))
    else:
        features["VRP1_centroid_to_depot"] = 0.0

    dists_to_depot = dist[depot_idx, cust_indices]
    features["VRP2_cust_to_depot_mean"] = float(np.mean(dists_to_depot))
    features["VRP2_cust_to_depot_std"] = float(np.std(dists_to_depot))

    cust_demands = demand[cust_indices]
    features["VRP3_demand_mean"] = float(np.mean(cust_demands))
    dem_std = float(np.std(cust_demands))
    features["VRP3_demand_std"] = dem_std
    if dem_std < 1e-9 or len(cust_demands) <= 2:
        features["VRP3_demand_skew"] = 0.0
    else:
        features["VRP3_demand_skew"] = float(skew(cust_demands))

    tot_demand = float(np.sum(cust_demands))
    features["VRP4_demand_to_capacity_ratio"] = float(tot_demand / max(1.0, capacity))
    mean_dem = float(np.mean(cust_demands)) if len(cust_demands) > 0 else 1.0
    features["VRP5_avg_cust_per_vehicle"] = float(capacity / max(1e-6, mean_dem))

    return features
