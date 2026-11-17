"""Write the affinity graph as data for the simulator UI (ui/index.html).

affinity_graph.py draws the graph with matplotlib and saves a PNG. This writes
the same graph -- the strong-edge subgraph H, its Louvain clusters, the spring
layout positions and the cluster summary, exactly as that script computed them --
to affinity_graph.json, so the UI can draw it in its own style. Nothing is
recomputed except the modularity figure, which the script only printed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx

DEFAULT_CUTOFF = "2026-09-21"      # exclusive: picks before this day; = the simulator's run start


def file_sha16(path) -> str:
    """Same hash as reslot.utils.manifest.file_hash (the simulator checks it)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_meta(path, cutoff: str, **info) -> Path:
    """<output>_meta.json: the data cutoff (exclusive) and how the output was made."""
    path = Path(path)
    path.write_text(json.dumps({"cutoff_exclusive": str(cutoff), **info}, indent=2, default=str), encoding="utf-8")
    return path


def read_meta(path) -> dict:
    """The meta a previous step wrote; refuses an output with none (cutoff unknown)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found: rerun the previous affinity step (build_matrix.py first)")
    return json.loads(path.read_text(encoding="utf-8"))


def graph_json(H: nx.Graph, clusters: list, pos: dict, summary, min_jaccard: float,
               resolution: float, seed: int, cutoff: str | None = None) -> dict:
    """H: strong-edge subgraph; clusters: list of SKU sets, index = cluster id;
    pos: {sku: (x, y)}; summary: DataFrame indexed by cluster id with columns
    skus, pick_lines, top_sku, name. cutoff: exclusive end of the picks used."""
    cluster_of = {sku: i for i, c in enumerate(clusters) for sku in c}
    nodes = [{"sku": str(s), "cluster": int(cluster_of[s]), "x": float(pos[s][0]), "y": float(pos[s][1]),
              "orders": int(d.get("orders", 0)), "descr": str(d.get("descr", "")), "degree": int(H.degree[s])}
             for s, d in H.nodes(data=True)]
    edges = [{"a": str(u), "b": str(v), "jaccard": float(d["weight"]), "orders": int(d.get("orders", 0))}
             for u, v, d in H.edges(data=True)]
    rows = [{"id": int(i), "name": str(r["name"]), "skus": int(r["skus"]), "pick_lines": int(r["pick_lines"]),
             "top_sku": str(r["top_sku"])} for i, r in summary.iterrows()]
    return {"schema": "racksmart-affinity/1",
            "data": {"cutoff_exclusive": cutoff},
            "method": {"edges": f"order co-occurrence, Jaccard >= {min_jaccard}", "communities": "Louvain",
                       "resolution": resolution, "seed": seed, "layout": f"networkx spring_layout, seed {seed}",
                       "modularity": round(float(nx.community.modularity(H, clusters, weight="weight")), 3)},
            "clusters": sorted(rows, key=lambda r: r["id"]), "nodes": nodes, "edges": edges}


def write_graph_json(path, *args, **kw) -> Path:
    path = Path(path)
    path.write_text(json.dumps(graph_json(*args, **kw)), encoding="utf-8")
    return path
