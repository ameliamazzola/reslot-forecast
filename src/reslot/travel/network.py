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
    stop_lane                CIRCLE   outbound staging lane mouth (cross-aisle end); drop-off
    stop_dispatch            CIRCLE   dispatch board, where trucks pick up jobs; trip start
    stop_line                CIRCLE   production-line pallet pickup (and supply points)
    line_label, lane_label   TEXT     names the nearest stop (WMS code, e.g. L3)
    pd_placeholder           CIRCLE   legacy placeholder trip start / drop-off (retired)
    -st_r_sd_col             INSERT   rack sections with a building column (checks only)

Two lines are connected ONLY where they share a point: an endpoint or a stop
circle. Lines that merely cross are not joined. That is deliberate -- every
place a truck can turn has to be drawn -- and it is why every junction carries
a circle. Guarded by test_crossing_without_shared_point_is_not_a_junction.

Drawing hygiene handled here, all reported in the manifest: a circle pasted
twice is one stop; a piece of path with no stop on it is dropped (it cannot be
on any route); circles on path layers and lines on stop layers are ignored.

Units are drawing units: inches, AutoCAD World coordinates. Distances are
reported in feet.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

STOP_PREFIX = {"rack": "S", "junction": "J", "lane": "L", "dispatch": "D", "line": "LN", "pd": "PD"}


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
    label_layers = {_norm(l) for l in layers.get("labels", [])}

    lines, circles, columns, misfiled, labels = [], [], [], [], []
    for ent in iterdxf.modelspace(str(dxf_path)):
        lay, kind = _norm(ent.dxf.get("layer", "")), ent.dxftype()
        if kind == "LINE" and lay in path_layers:
            lines.append((tuple(ent.dxf.start)[:2], tuple(ent.dxf.end)[:2], lay))
        elif kind == "CIRCLE" and lay in stop_layers:
            c = tuple(ent.dxf.center)[:2]
            circles.append((c[0], c[1], stop_layers[lay]))
        elif kind == "INSERT" and col_layer and lay == col_layer:
            columns.append(tuple(ent.dxf.insert)[:2])
        elif kind in ("TEXT", "MTEXT") and lay in label_layers:
            p = tuple(ent.dxf.insert)[:2]
            txt = ent.dxf.text if kind == "TEXT" else ent.text
            labels.append((p[0], p[1], str(txt).strip(), lay))
        elif kind == "CIRCLE" and lay in path_layers:
            c = tuple(ent.dxf.center)[:2]
            misfiled.append(f"CIRCLE on {lay} @{c[0]:.1f},{c[1]:.1f}")
        elif kind == "LINE" and lay in stop_layers:
            a, b = tuple(ent.dxf.start)[:2], tuple(ent.dxf.end)[:2]
            misfiled.append(f"LINE on {lay} @{a[0]:.1f},{a[1]:.1f}->{b[0]:.1f},{b[1]:.1f}")
    # Misfiled entities are NOT used -- a circle on a path layer is not a stop,
    # a line on a stop layer is not a path. They are reported so the drawing
    # gets fixed, never guessed at.
    return {"lines": lines, "circles": circles, "columns": columns, "misfiled": misfiled,
            "labels": labels}


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
    # A circle drawn twice (copy/paste) is one stop: same kind, same stop_id.
    seen, uniq = set(), []
    for x, y, kind in circles:
        sid = stop_id(kind, x, y)
        if sid not in seen:
            seen.add(sid)
            uniq.append((x, y, kind))
    n_dup = len(circles) - len(uniq)
    circles = uniq

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
    stops = pd.DataFrame(rows)
    stops.attrs["duplicate_circles_merged"] = n_dup
    return G, stops


def label_stops(stops: pd.DataFrame, labels: list, max_in: float) -> tuple[pd.DataFrame, list]:
    """Attach each label to the nearest stop circle within max_in.

    Returns (stops with a `label` column, labels that found no stop). Labels
    name stops in WMS terms (L3, Empty Pallets); they never move a stop.
    Junctions are never labelled.
    """
    stops = stops.copy()
    stops["label"] = None
    cand = stops[stops["node_type"] != "junction"]
    xy = cand[["x_in", "y_in"]].to_numpy()
    orphans = []
    for x, y, txt, lay in labels:
        if len(xy) == 0:
            orphans.append(f"{txt}@{x:.1f},{y:.1f}"); continue
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        i = int(d.argmin())
        if d[i] > max_in:
            orphans.append(f"{txt}@{x:.1f},{y:.1f}"); continue
        stops.loc[cand.index[i], "label"] = txt
    return stops, orphans


def prune_orphan_fragments(G: nx.Graph) -> list[str]:
    """Remove connected pieces that carry no stop circle, in place.

    A fragment with no stop can never be the start or end of a route, so it
    cannot change any distance -- typically a stray 1-inch line left behind by
    an edit. Pieces WITH a stop are never removed: if they are disconnected the
    network check fails, because that is a real gap in the drawing.
    Returns a description of each removed fragment for the manifest.
    """
    typed = nx.get_node_attributes(G, "node_type")
    removed = []
    for comp in list(nx.connected_components(G)):
        if not any(n in typed for n in comp):
            length = sum(d["length_in"] for _, _, d in G.subgraph(comp).edges(data=True))
            n0 = min(comp)
            removed.append(f"{len(comp)} vertices, {length:.1f} in, @{n0[0]:.1f},{n0[1]:.1f}")
            G.remove_nodes_from(comp)
    return removed


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

def network_checks(G: nx.Graph, stops: pd.DataFrame, orphans: list | None = None,
                   misfiled: list | None = None) -> dict:
    """What would make the distance table wrong, counted.

    junctions_without_circle: places three or more paths meet with no circle.
    The graph still works, but the drawing no longer documents itself.
    misfiled_entities_ignored: circles on path layers / lines on stop layers.
    Not used; fix in the drawing.
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
        "duplicate_circles_merged": int(stops.attrs.get("duplicate_circles_merged", 0)),
        "orphan_fragments_removed": list(orphans or []),
        "misfiled_entities_ignored": list(misfiled or []),
        "has_pd": bool((stops["node_type"] == "pd").any()),
        "n_dispatch": int((stops["node_type"] == "dispatch").sum()),
        "n_lane": int((stops["node_type"] == "lane").sum()),
        # Lane mouths, line pickups and the dispatch board are spur ends on purpose.
        "dead_ends": sorted([f"{typed.get(n, 'vertex')}@{n[0]:.1f},{n[1]:.1f}"
                             for n, d in deg.items()
                             if d == 1 and typed.get(n) not in ("lane", "dispatch", "line", "pd")]),
        "junctions_without_circle": sorted([f"{n[0]:.1f},{n[1]:.1f}" for n, d in deg.items()
                                            if d >= 3 and n not in typed]),
        "rack_stops_at_junctions": sorted([f"{n[0]:.1f},{n[1]:.1f}" for n, d in deg.items()
                                           if d >= 3 and typed.get(n) == "rack"]),
    }
