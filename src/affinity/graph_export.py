"""Write the affinity graph as data for the simulator UI (ui/index.html).

affinity_graph.py draws the graph with matplotlib and saves a PNG. This writes
the same graph -- the strong-edge subgraph H, its Louvain clusters, the spring
layout positions and the cluster summary, exactly as that script computed them --
to affinity_graph.json, so the UI can draw it in its own style. Nothing is
recomputed except the modularity figure, which the script only printed.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


def graph_json(H: nx.Graph, clusters: list, pos: dict, summary, min_jaccard: float,
               resolution: float, seed: int) -> dict:
    """H: strong-edge subgraph; clusters: list of SKU sets, index = cluster id;
    pos: {sku: (x, y)}; summary: DataFrame indexed by cluster id with columns
    skus, pick_lines, top_sku, name."""
    cluster_of = {sku: i for i, c in enumerate(clusters) for sku in c}
    nodes = [{"sku": str(s), "cluster": int(cluster_of[s]), "x": float(pos[s][0]), "y": float(pos[s][1]),
              "orders": int(d.get("orders", 0)), "descr": str(d.get("descr", "")), "degree": int(H.degree[s])}
             for s, d in H.nodes(data=True)]
    edges = [{"a": str(u), "b": str(v), "jaccard": float(d["weight"]), "orders": int(d.get("orders", 0))}
             for u, v, d in H.edges(data=True)]
    rows = [{"id": int(i), "name": str(r["name"]), "skus": int(r["skus"]), "pick_lines": int(r["pick_lines"]),
             "top_sku": str(r["top_sku"])} for i, r in summary.iterrows()]
    return {"schema": "racksmart-affinity/1",
            "method": {"edges": f"order co-occurrence, Jaccard >= {min_jaccard}", "communities": "Louvain",
                       "resolution": resolution, "seed": seed, "layout": f"networkx spring_layout, seed {seed}",
                       "modularity": round(float(nx.community.modularity(H, clusters, weight="weight")), 3)},
            "clusters": sorted(rows, key=lambda r: r["id"]), "nodes": nodes, "edges": edges}


def write_graph_json(path, *args, **kw) -> Path:
    path = Path(path)
    path.write_text(json.dumps(graph_json(*args, **kw)), encoding="utf-8")
    return path
