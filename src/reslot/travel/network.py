"""
Travel network built from the York DC layout drawing (DXF).

The drawing, not the WMS, is the source of truth for geometry. Paths and stop
points are drawn by hand on dedicated layers; this module turns them into a
graph and a stop-to-stop distance table in feet.

Layer conventions (names compared case-insensitively, leading apostrophe
ignored -- AutoCAD exported them as 'Path_Aisle etc.):

    path_aisle, path_cross   LINE     travel paths (graph edges)
    stop_rack                CIRCLE   where a truck stops at rack; 4 locations each
    stop_junction            CIRCLE   aisle / cross-aisle intersections; no locations
    pd_placeholder           CIRCLE   PLACEHOLDER trip start / drop-off
    -st_r_sd_col             INSERT   rack sections with a building column (checks only)

Two lines are connected ONLY where they share a point: an endpoint or a stop
circle. Lines that merely cross are not joined. That is deliberate -- every
place a truck can turn has to be drawn -- and it is why every junction carries
a circle. Guarded by test_crossing_without_shared_point_is_not_a_junction.

Units are drawing units: inches, AutoCAD World coordinates. Distances are
reported in feet.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

STOP_PREFIX = {"rack": "S", "junction": "J", "pd": "PD"}


def _norm(layer: str) -> str:
    return layer.strip().lstrip("'").lower()


def stop_id(kind: str, x: float, y: float) -> str:
    """Stable ID from coordinates, e.g. S_180855.6_5571.4. Shared by the
    distance table and the location crosswalk -- they join on this."""
    return f"{STOP_PREFIX[kind]}_{x:.1f}_{y:.1f}"


# --------------------------------------------------------------------------
# Read
# --------------------------------------------------------------------------

def read_layout(dxf_path: str | Path, layers: dict) -> dict:
    """Pull path lines, stop circles and column sections out of the DXF.

    Streams modelspace (the file is ~240 MB), so memory stays flat; expect a
    few minutes. Returns plain arrays so everything downstream is testable
    without a drawing.
    """
    from ezdxf.addons import iterdxf

    path_layers = {_norm(l) for l in layers["paths"]}
    stop_layers = {_norm(k): v for k, v in layers["stops"].items()}
    col_layer = _norm(layers.get("columns", ""))

    lines, circles, columns = [], [], []
    for ent in iterdxf.modelspace(str(dxf_path)):
        lay, kind = _norm(ent.dxf.get("layer", "")), ent.dxftype()
        if kind == "LINE" and lay in path_layers:
            lines.append((tuple(ent.dxf.start)[:2], tuple(ent.dxf.end)[:2], lay))
        elif kind == "CIRCLE" and lay in stop_layers:
            c = tuple(ent.dxf.center)[:2]
            circles.append((c[0], c[1], stop_layers[lay]))
        elif kind == "INSERT" and col_layer and lay == col_layer:
            columns.append(tuple(ent.dxf.insert)[:2])
    return {"lines": lines, "circles": circles, "columns": columns}


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build_graph(lines, circles, tol_in: float = 0.5):
    """Graph whose nodes are line endpoints and circle centres.

    Each line is split at every endpoint or circle centre lying on it (within
    tol_in), so a circle in the middle of an aisle line becomes a node.
    tol_in only absorbs floating-point noise; a clean drawing needs nothing
    looser. Returns (graph, stops) where stops has one row per circle.
    """
    segs = [(np.asarray(a, float), np.asarray(b, float), lay) for a, b, lay in lines]
    segs = [s for s in segs if np.hypot(*(s[1] - s[0])) > 0.01]
    C = np.array([(x, y) for x, y, _ in circles], float).reshape(-1, 2)
    pts = np.array([p for a, b, _ in segs for p in (a, b)] + list(C), float)

    tree, rep = cKDTree(pts), {}
    for i, p in enumerate(pts):
        if i in rep:
            continue
        for j in tree.query_ball_point(p, tol_in):
            rep.setdefault(j, i)

    def key(i):
        return tuple(np.round(pts[rep[i]], 2))

    G = nx.Graph()
    for a, b, lay in segs:
        ab = b - a
        t = (pts - a) @ ab / (ab @ ab)
        d = np.hypot(*(pts - (a + np.outer(t, ab))).T)
        idx = np.where((d < tol_in) & (t > -1e-6) & (t < 1 + 1e-6))[0]
        ks = [key(i) for i in idx[np.argsort(t[idx], kind="stable")]]
        ks = [k for n, k in enumerate(ks) if n == 0 or k != ks[n - 1]]
        for u, v in zip(ks, ks[1:]):
            if u != v:
                G.add_edge(u, v, length_in=float(np.hypot(u[0] - v[0], u[1] - v[1])), layer=lay)

    off = len(pts) - len(C)
    rows = []
    for i, (x, y, kind) in enumerate(circles):
        node = key(off + i)
        rows.append({"stop_id": stop_id(kind, x, y), "x_in": x, "y_in": y,
                     "node_type": kind, "node": node, "on_path": node in G})
        if node in G:
            G.nodes[node]["node_type"] = kind
    return G, pd.DataFrame(rows)


def distance_matrix_ft(G: nx.Graph, stops: pd.DataFrame) -> pd.DataFrame:
    """Shortest-path distance in feet between every pair of stops."""
    nodes = stops["node"].tolist()
    D = np.empty((len(nodes), len(nodes)))
    for i, n in enumerate(nodes):
        dist = nx.single_source_dijkstra_path_length(G, n, weight="length_in")
        D[i] = [dist[m] for m in nodes]
    D = (D + D.T) / 2 / 12.0   # symmetrise away float summation-order noise
    return pd.DataFrame(np.round(D, 2), index=stops["stop_id"], columns=stops["stop_id"])


# --------------------------------------------------------------------------
# Checks -- written into the manifest, asserted by tests
# --------------------------------------------------------------------------

def network_checks(G: nx.Graph, stops: pd.DataFrame) -> dict:
    """What would make the distance table wrong, counted.

    junctions_without_circle: places three or more paths meet with no circle.
    The graph still works, but the drawing no longer documents itself.
    """
    deg = dict(G.degree())
    typed = nx.get_node_attributes(G, "node_type")
    return {
        "connected": bool(nx.is_connected(G)) if len(G) else False,
        "n_components": nx.number_connected_components(G),
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "stops_by_type": stops["node_type"].value_counts().to_dict(),
        "circles_off_path": stops.loc[~stops["on_path"], "stop_id"].tolist(),
        "duplicate_stop_ids": int(stops["stop_id"].duplicated().sum()),
        "dead_ends": sorted([f"{typed.get(n, 'vertex')}@{n[0]:.1f},{n[1]:.1f}"
                             for n, d in deg.items() if d == 1]),
        "junctions_without_circle": sorted([f"{n[0]:.1f},{n[1]:.1f}" for n, d in deg.items()
                                            if d >= 3 and n not in typed]),
        "rack_stops_at_junctions": sorted([f"{n[0]:.1f},{n[1]:.1f}" for n, d in deg.items()
                                           if d >= 3 and typed.get(n) == "rack"]),
    }
