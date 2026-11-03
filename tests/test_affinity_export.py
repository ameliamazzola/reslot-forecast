"""Tests for the affinity graph export (src/affinity/graph_export.py) and its
place in the UI bundle. Synthetic graph only."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "affinity"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import networkx as nx
import pandas as pd
import pytest

from graph_export import graph_json, write_graph_json
from reslot.sim.export import BundleError


def toy():
    """Two clusters of three SKUs, as affinity_graph.py hands them over."""
    H = nx.Graph()
    for sku, orders in [("1", 50), ("2", 40), ("3", 30), ("4", 20), ("5", 10), ("6", 5)]:
        H.add_node(sku, orders=orders, descr=f"PRODUCT {sku}", support=orders / 100)
    for u, v, w in [("1", "2", .5), ("2", "3", .4), ("1", "3", .3), ("4", "5", .6), ("5", "6", .2), ("3", "4", .11)]:
        H.add_edge(u, v, weight=w, orders=int(w * 100))
    clusters = [{"1", "2", "3"}, {"4", "5", "6"}]
    pos = nx.spring_layout(H, weight="weight", seed=42)
    summary = pd.DataFrame({"skus": [3, 3], "pick_lines": [900, 300], "top_sku": ["PRODUCT 1", "PRODUCT 4"],
                            "name": ["Alpha", "Beta"]}, index=pd.Index([0, 1], name="cluster"))
    return H, clusters, pos, summary


def test_graph_json_is_the_script_graph_unchanged():
    H, clusters, pos, summary = toy()
    g = graph_json(H, clusters, pos, summary, min_jaccard=0.10, resolution=1.0, seed=42)
    assert g["schema"] == "racksmart-affinity/1"
    assert {n["sku"]: n["cluster"] for n in g["nodes"]} == {"1": 0, "2": 0, "3": 0, "4": 1, "5": 1, "6": 1}
    n1 = next(n for n in g["nodes"] if n["sku"] == "1")
    assert (n1["x"], n1["y"]) == (float(pos["1"][0]), float(pos["1"][1])) and n1["orders"] == 50
    assert len(g["edges"]) == H.number_of_edges() and g["clusters"][0]["name"] == "Alpha"
    assert g["method"]["seed"] == 42 and "Jaccard >= 0.1" in g["method"]["edges"]
    assert g["method"]["modularity"] == round(nx.community.modularity(H, clusters, weight="weight"), 3)


def test_bundle_carries_the_graph_and_refuses_anything_else(tmp_path, two_runs):
    from reslot.sim.export import build_bundle
    from reslot.travel.artifacts import CROSSWALK_FILE, DISTANCE_FILE, load_travel
    from reslot.utils.manifest import file_hash
    H, clusters, pos, summary = toy()
    path = write_graph_json(tmp_path / "affinity_graph.json", H, clusters, pos, summary,
                            min_jaccard=0.10, resolution=1.0, seed=42)
    g = json.loads(path.read_text())
    _, travel, runs = two_runs
    stops = pd.read_csv(travel / "stops.csv", dtype={"stop_id": str, "label": str})
    hashes = {f: file_hash(travel / f) for f in (DISTANCE_FILE, CROSSWALK_FILE)}
    b = build_bundle([str(d) for d in runs.values()], load_travel(travel), stops, "t", hashes, g)
    assert b["affinity"]["clusters"] == g["clusters"] and len(b["affinity"]["nodes"]) == 6
    with pytest.raises(BundleError, match="affinity"):
        build_bundle([str(d) for d in runs.values()], load_travel(travel), stops, "t", hashes, {"schema": "other"})
    assert "affinity" not in build_bundle([str(d) for d in runs.values()], load_travel(travel), stops, "t", hashes)


from test_export_ui import two_runs  # noqa: E402,F401  (shared fixture)
