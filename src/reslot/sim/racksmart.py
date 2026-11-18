"""
RackSmart as a simulator policy, using notebooks/RackSmart_final.ipynb unchanged.

The algorithm is not copied. The notebook's own definitions are read from the
.ipynb and executed as they are (cells found by content, not position):

    settings     WAGE, MPH, COST_PER_CASE, SETTINGS (the tuned values), RESLOT, file names
    inputs       the Inputs contract
    engine       World, Planner, replay, move_cost
    plus         REPEAT_MIN, PlannerPlus, limit_swaps, racksmart_sweep
    plan         MAX_CLUSTER, _ClusterPlanner, plan  (the tuned RackSmart)
    data         location_clusters, build_families, refill_inputs, MPH_RATIO -- only these,
                 taken out of the cell that also holds load_york and runs it
    run          plan_month, swap_list, check_month -- only these, not the loop that runs them

Only definitions are executed: a statement that would do work at load time (a
call, a loop, `INP = load_york()`, a file write) is refused, and so is anything
that names load_york or INP. The data cell's file reads (pd.read_csv(CLUSTERS)
etc.) run only when the adapter calls those functions, on paths it sets.

RackSmart is planned by the notebook's own plan_month(inp, month): the tours of
the look-back period, its clusters (location_clusters), its refill cost
(refill_inputs) and its swap groups (Inputs.extra["groups"]) go into plan().
The adapter gives it one period, the planning window, labelled by its first day,
and asks for the plan of the run start (labelled by its day): ISO dates compare
the way plan_month's month labels do.

This file is the ADAPTER the notebook asks for. It fills Inputs with the same
rules as load_york, on the simulator's data:

    D        driving feet between faces + depot, from the travel network table.
             Depot = trip_ends.start_stop (Stop_Dispatch). load_york looked for a
             "PD..." stop, retired on 2026-10-19.
    faces    every face the structure allows (FACE_RULE): from the location
             master + drawing only, never from stock or from picks in any window
             (load_york: every mapped level-A location in its pick file). All levels
             (team decision, 2026-11-03).
    tours    one operator x one order; consecutive picks at one location are one
             visit; tours with >= 3 visits; ordered by first pick time as load_york
             orders them. Case picks in scope (WZPCK/CS/F). Only picks before the
             planning end are read (a tour running past it is cut).
    sku      extra["sku"]: (period, face, item) -> picks, cases, from the planning
             picks. location_clusters and refill_inputs place an item at a face by
             it (the item with the most cases picked there in the look-back).
    groups   extra["groups"]: load_york's aisle fence (GG/HH vs the rest), plus
             2 x height code (SHORT 1, TALL 2). Heights come from the location master's
             putaway zone (racksmart.rack_heights.location_master): PY... = TALL,
             PZ... = SHORT, anything else = no height (load_york read a RACK_HEIGHTS file).
    cases    racksmart.move_cases:
               inventory_snapshot  CS inventory per location, summed; missing -> median
                                   location (load_york's rule). The snapshot post-dates
                                   the run start: a named lookahead exception.
               pallet_fill         snapshot-free: cases per pallet of the face's item
                                   (median CASES_PER_PALLET in planning picks) x face_fill;
                                   no item or no pallet size -> the default.
    golden   the nearest 10.8% of faces to the depot (load_york's rule).
    dpu      $ per foot: (WAGE/3600) / (MPH*5280/3600).

Notebook file settings the adapter replaces (recorded in the manifest):
CROSSWALK (the travel network's crosswalk), CLUSTERS (racksmart.clusters_path),
CASES_PER_PALLET and REFILL_ORIGIN (racksmart.refill; null = refills off, the
notebook's own "None = refills off").

Every input the plan used is recorded with the end of the data it saw
(policy.planning_inputs); the engine refuses any that ends after the run start
unless it is named in config lookahead_exceptions.

Cost model: the notebook's, unchanged -- driving $ = feet x dpu, moving $ =
COST_PER_CASE x cases in both swapped locations (notebook move_cost), plus, in
RackSmart's own money test, the refill-trip term when refills are on.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.manifest import file_hash
from .policy import Move, Policy

# key -> (how the cell is recognised, the top-level names taken from it; None = all of it).
# Exactly one code cell must match each.
CELLS = {
    "settings": (lambda s: "SETTINGS = dict(" in s and "COST_PER_CASE" in s, None),
    "inputs": (lambda s: "class Inputs" in s, None),
    "engine": (lambda s: "class World" in s and "def replay" in s and "def move_cost" in s, None),
    "plus": (lambda s: "class PlannerPlus" in s, None),
    "plan": (lambda s: "def plan(" in s and "class _ClusterPlanner" in s, None),
    "data": (lambda s: "def load_york" in s,
             ("location_clusters", "build_families", "refill_inputs", "MPH_RATIO")),
    "run": (lambda s: "def plan_month" in s, ("plan_month", "swap_list", "check_month")),
}
NEVER_RUN = ("load_york", "INP")                  # names executed code may not define or use
NEVER_RUN_TEXT = (".to_csv(",)
REQUIRED = ("Inputs", "World", "Planner", "PlannerPlus", "replay", "move_cost", "limit_swaps",
            "plan", "plan_month", "swap_list", "location_clusters", "refill_inputs",
            "WAGE", "MPH", "COST_PER_CASE", "REPEAT_MIN", "SETTINGS", "RESLOT", "MAX_CLUSTER", "MPH_RATIO")
CONSTANTS = ("WAGE", "MPH", "COST_PER_CASE", "REPEAT_MIN", "SETTINGS", "RESLOT", "MAX_CLUSTER", "MPH_RATIO")
# Notebook file settings the adapter sets before each RackSmart plan.
PATH_SETTINGS = ("CROSSWALK", "CLUSTERS", "CASES_PER_PALLET", "REFILL_ORIGIN")
GOLDEN_SHARE = 0.108                              # load_york: the nearest 10.8% of faces
HEIGHT_CODE = {"SHORT": 1, "TALL": 2}             # load_york's RACK_HEIGHTS codes
PUTAWAY_HEIGHT = {"PY": "TALL", "PZ": "SHORT"}    # putaway-zone prefix (supervisor's key, 2026-10-19)
HEIGHT_RULE = ("height at a face = TALL if the location master's putaway_zone starts PY, SHORT if PZ, "
               "none otherwise; RackSmart swaps only within (aisle fence, height) groups")
# The tuned RackSmart run (notebook SETTINGS) differs from ours here; written into every
# RackSmart manifest.
REFILL_OFF = ("refill-trip cost OFF: the notebook's tuned run has SETTINGS refill=True; off here until "
              "the refill files' date range is confirmed (racksmart.refill)")

# Where the adapter deliberately differs from the notebook's load_york. Written
# into every notebook-method manifest next to the known gaps.
ADAPTER_DIFFERENCES = [
    "depot = trip_ends.start_stop (Stop_Dispatch); load_york looked for a retired PD stop",
    "case picks in zone WZPCK only (WZPCK/CS/F); load_york kept every CS pick on a mapped face",
    "faces at all levels; load_york kept crosswalk level A only",
    "faces = every structurally usable face (FACE_RULE), picked or not; "
    "load_york used every mapped location in its pick file (two years). The golden 10.8% is of this set",
    "one planning period = racksmart.planning_window, cut at the planning end; load_york's periods are "
    "calendar months and a tour belongs to the month of its first pick (a tour crossing a month end stays whole)",
    "rack heights from the location master's putaway zone (PY = TALL, PZ = SHORT), "
    "racksmart.rack_heights; load_york read a RACK_HEIGHTS file (LOCATION, PALLET_HEIGHT)",
    "refill files from racksmart.refill (load_york: CASES_PER_PALLET, REFILL_ORIGIN); null = refills off",
]
GAPS = ["visit order after swaps: the notebook's cheapest-insertion replan, not a WMS re-route"]
HEIGHT_GAP = "PY/PZ tall/short slot rule not enforced"
CASES_GAPS = {
    "inventory_snapshot": "case counts from one inventory snapshot taken after the run start "
                          "(lookahead exception move_cases), applied to the planning month and the run",
    "pallet_fill": "case counts not observed: cases per pallet x face_fill (snapshot-free sensitivity)",
}

# The candidate face set. Structure only: the same faces whatever was picked or stocked.
FACE_RULE = ("candidate faces = locations in location_stop_crosswalk.csv (location master + layout "
             "drawing) with status 'ok', zone = scope.zone, slot_type = scope.slot_type, and a stop in "
             "stop_distance_ft.csv (TravelNetwork.usable_faces). No stock, no picks, no dates.")
FACE_ITEM_RULE = ("item at a face = the item of the last in-scope case pick at that face before the "
                  "planning end (ties on time: highest event_id); a face with no such pick has no item")
CLUSTER_ITEM_RULE = ("notebook location_clusters: a face takes the cluster of the item with the most cases "
                     "picked there in the planning window; clusters over MAX_CLUSTER faces split by co-picks "
                     "in the planning tours")
MOVE_CASES = ("inventory_snapshot", "pallet_fill")

# The comparison methods. The current notebook has no driver for them (its old
# run_month was removed); they are wired as that run_month wired them, on the
# current notebook's Planner / PlannerPlus, whose code for them is unchanged.
PREVIOUS_DRIVER = "run_month in the previous RackSmart_final.ipynb (git blob a97dacf, commit 31bcb0c)"
NOT_TUNED = ("NOT tuned like RackSmart: none of the notebook's tuned SETTINGS (lookback, K, margin, top, "
             "passes, cap, cluster, refill) apply; Planner defaults K=5, margin 1.0, no cap, no clusters, no refill")
_COMPARE = ("run as " + PREVIOUS_DRIVER + " ran its comparison methods: "
            "no stability setting (stable=False), no backtest guard, none of the tuned SETTINGS")
_FIXED_N = ("swap count fixed to the detour method's count (run_month's n_ref / n_det), "
            "not chosen by the method")
_NO_GROUPS = HEIGHT_GAP + " and no GG/HH aisle fence: the comparison methods do not use swap groups"
METHODS = {
    # uses_golden_zone: the method picks swap partners from Inputs.golden (Planner.quietest_golden)
    "racksmart":       {"label": "RackSmart (final)", "uses_golden_zone": False, "gaps": []},
    "classic_abc":     {"label": "Classic ABC (no cost test)", "uses_golden_zone": True,
                        "gaps": [NOT_TUNED, _COMPARE, _FIXED_N, _NO_GROUPS, "no cost test: a swap is kept even when "
                                 "it costs more than the driving it saves"]},
    "abc_cost_test":   {"label": "ABC with cost test", "uses_golden_zone": True, "gaps": [NOT_TUNED, _COMPARE, _NO_GROUPS]},
    "family_grouping": {"label": "Family grouping + cost test", "uses_golden_zone": False,
                        "gaps": [NOT_TUNED, _COMPARE, _NO_GROUPS, "location families from co-picks within tours "
                                 "(notebook's family_grouping), not the affinity clusters in sku_clusters.csv",
                                 "target slots are all faces in the plan, ranked by feet from dispatch"]},
}


@dataclass
class Notebook:
    ns: dict
    info: dict

    def __getattr__(self, name):
        try:
            return self.ns[name]
        except KeyError:
            raise AttributeError(name) from None


_DEFINITIONS = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)


def _definitions(path, i: int, key: str, src: str, names) -> ast.Module:
    """The statements of one cell that will run: all of them, or the named
    definitions (+ the cell's imports). Refuses anything that would do work."""
    tree = ast.parse(src)
    body = tree.body
    if names is not None:
        def named(node):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                return node.name in names
            if isinstance(node, ast.Assign):
                return any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
            return False
        body = [n for n in body if isinstance(n, (ast.Import, ast.ImportFrom)) or named(n)]
        found = {n.name for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef))} | {
            t.id for n in body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        if set(names) - found:
            raise ValueError(f"{path}: cell {i} ('{key}') no longer defines {sorted(set(names) - found)}")
    for node in body:
        is_doc = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not (isinstance(node, _DEFINITIONS) or is_doc):
            raise ValueError(f"{path}: cell {i} ('{key}') runs a {type(node).__name__} statement at load "
                             f"(line {node.lineno}); refusing to run it")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            calls = [c for c in ast.walk(node) if isinstance(c, ast.Call)]
            if any(not (isinstance(c.func, ast.Name) and c.func.id == "dict") for c in calls):
                raise ValueError(f"{path}: cell {i} ('{key}') calls a function at load (line {node.lineno}); "
                                 "refusing to run it")
        used = {getattr(n, "id", None) or getattr(n, "name", None) for n in ast.walk(node)}
        bad = sorted(used & set(NEVER_RUN))
        text = ast.get_source_segment(src, node) or ""
        bad += [t for t in NEVER_RUN_TEXT if t in text]
        if bad:
            raise ValueError(f"{path}: cell {i} ('{key}') uses {bad}; refusing to run it")
    return ast.Module(body=body, type_ignores=[])


def load_notebook(path) -> Notebook:
    """Execute the algorithm definitions of RackSmart_final.ipynb, unchanged."""
    nb = json.loads(open(path, encoding="utf-8").read())
    code = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    chosen = {}
    for key, (match, names) in CELLS.items():
        hits = [(i, s) for i, s in code if match(s)]
        if len(hits) != 1:
            raise ValueError(f"{path}: expected one '{key}' cell, found {len(hits)}")
        i, s = hits[0]
        chosen[key] = (i, _definitions(path, i, key, s, names))
    # A real module, registered, so the notebook's @dataclass can resolve itself.
    module = types.ModuleType("racksmart_notebook")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ns.update({"re": re, "pd": pd})
    for key, (i, tree) in sorted(chosen.items(), key=lambda kv: kv[1][0]):   # notebook order
        exec(compile(tree, f"{path}#cell{i}", "exec"), ns)
    missing = [n for n in REQUIRED if n not in ns]
    if missing:
        raise ValueError(f"{path}: notebook no longer defines {missing}")
    info = {"path": str(path), "sha256_16": file_hash(path),
            "cells_run": {k: i for k, (i, _) in chosen.items()},
            "taken_from_cells": {k: list(n) for k, (_, n) in CELLS.items() if n is not None},
            "constants": {k: (dict(ns[k]) if isinstance(ns[k], dict) else ns[k]) for k in CONSTANTS},
            "settings_replaced_by_adapter": list(PATH_SETTINGS),
            "env_VISITED_ONLY": os.environ.get("VISITED_ONLY", "1")}
    return Notebook(ns, info)


# --------------------------------------------------------------------------
# Adapter: Inputs from the simulator's data, load_york's rules
# --------------------------------------------------------------------------

def visit_sequences(tours, index: dict) -> list[list[int]]:
    """load_york's tour rule on workload tours: consecutive picks at one
    location are one visit; keep tours with >= 3 visits; ordered by first pick
    time (load_york sorts each period's tours by it), then (order, user)."""
    out = []
    for t in sorted(tours, key=lambda t: (t.picks[0].ts, t.order_id, t.user_id)):
        seq = []
        for p in t.picks:
            i = index[p.location]
            if not seq or seq[-1] != i:
                seq.append(i)
        if len(seq) >= 3:
            out.append(seq)
    return out


def structural_faces(net, scope: dict) -> list:
    """FACE_RULE. Takes no events on purpose."""
    return sorted(net.usable_faces(scope["zone"], scope["slot_type"]).index)


def _in_scope_picks(events: pd.DataFrame, scope: dict) -> pd.DataFrame:
    return events[(events["event_class"] == scope["event_class"]) & (events["from_zone"] == scope["zone"])
                  & (events["uom"] == scope["uom"])]


def face_items(events: pd.DataFrame, scope: dict, end: pd.Timestamp) -> pd.Series:
    """FACE_ITEM_RULE: location -> item_id (str), from picks strictly before `end`."""
    s = _in_scope_picks(events, scope)
    s = s[(s["ts"] < end) & s["from_loc"].notna() & s["item_id"].notna()]
    last = s.sort_values(["ts", "event_id"], kind="stable").drop_duplicates("from_loc", keep="last")
    return pd.Series(last["item_id"].astype(str).str.strip().to_numpy(),
                     index=last["from_loc"].astype(str).to_numpy(), name="item_id").sort_index()


def cases_from_inventory(inventory: pd.DataFrame, faces: list) -> tuple[pd.Series, dict]:
    """load_york's rule: CS inventory per location, summed; missing -> median location."""
    inv = inventory[inventory.UOM == "CS"]
    by = inv.assign(l=inv.LOCATION.astype(str).str.strip()).groupby("l").QTY.sum()
    typical = float(np.median(by.values))
    cases = pd.Series([float(by.get(l, typical)) for l in faces], index=faces)
    return cases, {"basis": "inventory_snapshot", "faces_with_inventory": int(sum(l in by.index for l in faces)),
                   "typical_cases": typical}


def cases_from_pallet_fill(events: pd.DataFrame, scope: dict, end: pd.Timestamp, items: pd.Series,
                           faces: list, face_fill: float, default_cpp=None) -> tuple[pd.Series, dict]:
    """Snapshot-free: cases at a face = cases per pallet of its item x face_fill.
    Cases per pallet per item = median CASES_PER_PALLET over planning picks before
    `end` (values <= 0 are missing). No item / no pallet size -> default_cpp, or the
    median over the faces' items when default_cpp is None."""
    if not 0 < float(face_fill) <= 1:
        raise ValueError(f"pallet_fill.face_fill must be in (0, 1], got {face_fill}")
    s = _in_scope_picks(events, scope)
    s = s[(s["ts"] < end) & (s["cases_per_pallet"] > 0)]
    cpp = s.groupby(s["item_id"].astype(str).str.strip())["cases_per_pallet"].median()
    at = {l: float(cpp[items[l]]) for l in faces if l in items.index and items[l] in cpp.index}
    if default_cpp is None:
        if not at:
            raise ValueError("pallet_fill: no CASES_PER_PALLET for any face's item in the planning picks; "
                             "set racksmart.pallet_fill.cases_per_pallet")
        default_cpp = float(np.median(list(at.values())))
    cases = pd.Series([at.get(l, float(default_cpp)) * float(face_fill) for l in faces], index=faces)
    return cases, {"basis": "pallet_fill", "face_fill": float(face_fill), "default_cases_per_pallet": float(default_cpp),
                   "faces_with_item_pallet_size": len(at)}


def sku_table(plan_picks: pd.DataFrame | None, idx: dict, label: str) -> pd.DataFrame:
    """load_york's extra["sku"]: (period, face index, item) -> picks, cases
    (FROM_QTY summed), from in-scope planning picks on the planned faces."""
    cols = ["month", "li", "FROM_ITEM", "picks", "cases"]
    if plan_picks is None or plan_picks.empty:
        return pd.DataFrame(columns=cols)
    p = plan_picks[plan_picks["from_loc"].isin(idx) & plan_picks["item_id"].notna()]
    p = p.assign(month=label, li=p["from_loc"].map(idx), FROM_ITEM=p["item_id"].map(_sku),
                 q=pd.to_numeric(p["qty"], errors="coerce").fillna(0))
    return (p.groupby(["month", "li", "FROM_ITEM"]).agg(picks=("q", "size"), cases=("q", "sum"))
            .reset_index()[cols])


def heights_from_location_master(slots: pd.DataFrame, faces: list) -> tuple[pd.Series, dict]:
    """HEIGHT_RULE: location -> TALL / SHORT from the putaway zone's PY / PZ prefix
    (canonical SLOTS). Structural: the master, not stock or picks. Locations with
    neither prefix are left out (no height)."""
    pz = slots.set_index("location")["putaway_zone"].astype("string").str.strip().str.upper()
    h = pz.str[:2].map(PUTAWAY_HEIGHT).dropna().astype(str)
    h = h[~h.index.duplicated()]
    on = h.reindex(faces)
    return h, {"rule": HEIGHT_RULE, "faces": len(faces),
               "faces_tall": int((on == "TALL").sum()), "faces_short": int((on == "SHORT").sum()),
               "faces_without_py_pz": int(on.isna().sum()),
               "faces_not_in_master": int(sum(l not in pz.index for l in faces))}


def swap_groups(faces: list, heights: pd.Series | None) -> list:
    """load_york's swap groups: aisle fence (GG/HH = 1, else 0) + 2 x height code
    (SHORT 1, TALL 2, not on file 0). Faces swap only within their group."""
    groups = [1 if re.match(r"[A-Z]+", f).group(0) in ("GG", "HH") else 0 for f in faces]
    if heights is not None:
        groups = [g + 2 * HEIGHT_CODE.get(heights.get(f), 0) for g, f in zip(groups, faces)]
    return groups


def build_inputs(nb: Notebook, faces: list, net, depot: str, cases: pd.Series,
                 clusters_path=None, plan_tours: list = (), *, plan_picks: pd.DataFrame | None = None,
                 heights: pd.Series | None = None, refill: dict | None = None,
                 labels: tuple = ("plan", "run")):
    """cases: cases per face (indexed by location). clusters_path: the notebook's
    CLUSTERS (None = no clusters). plan_picks: in-scope planning picks (events),
    for extra["sku"]. heights: location -> SHORT/TALL. refill: {cases_per_pallet_path,
    origin_path} or None (refills off). labels: (planning period, run) -- the plan
    is plan_month(inp, run label) over the one period."""
    faces = sorted(faces)
    idx = {l: i for i, l in enumerate(faces)}
    n = len(faces)
    usable = net.usable_faces("WZPCK", "F")
    missing = [l for l in faces if l not in usable.index]
    if missing:
        raise ValueError(f"faces not on the network: {missing[:5]}")
    stops = list(net.distances.index)
    sidx = {s: i for i, s in enumerate(stops)}
    Ds = net.distances.to_numpy(float)
    si = np.array([sidx[usable.at[l, "stop_id"]] for l in faces])
    D = np.zeros((n + 1, n + 1))
    D[:n, :n] = Ds[np.ix_(si, si)]
    D[:n, n] = Ds[si, sidx[depot]]
    D[n, :n] = Ds[sidx[depot], si]

    no_cases = [l for l in faces if l not in cases.index]
    if no_cases:
        raise ValueError(f"no case count for faces {no_cases[:5]}")
    cases = [float(cases[l]) for l in faces]

    golden = [int(i) for i in np.argsort(D[n, :n])[: int(round(GOLDEN_SHARE * n))]]   # load_york's expression

    dpu = (nb.WAGE / 3600.0) / (nb.MPH * 5280 / 3600.0)

    label, run_label = labels
    if not str(label) < str(run_label):
        raise ValueError(f"planning label {label!r} must sort before the run label {run_label!r} (plan_month)")
    sku = sku_table(plan_picks, idx, label)
    groups = swap_groups(faces, heights)
    stop_of = {l: usable.at[l, "stop_id"] for l in faces}
    refill = refill or {}
    paths = {"CROSSWALK": net.sources.get("crosswalk"), "CLUSTERS": clusters_path,
             "CASES_PER_PALLET": refill.get("cases_per_pallet_path"), "REFILL_ORIGIN": refill.get("origin_path")}
    paths = {k: (None if v is None else str(v)) for k, v in paths.items()}

    train = visit_sequences(plan_tours, idx)
    inp = nb.Inputs("york", D, n, cases, golden, dpu, nb.COST_PER_CASE, {label: train}, unit_note="feet",
                    extra=dict(faces=faces, groups=groups, sku=sku, run_label=run_label, notebook_paths=paths,
                               stops=(net.crosswalk, net.distances, sidx, stop_of, [stop_of[l] for l in faces])))
    return inp, train, {"faces": n, "planning_tours": len(train), "golden": len(golden),
                        "faces_with_item": int(sku["li"].nunique()), "depot": depot,
                        "periods": {"planning": str(label), "run": str(run_label)},
                        "swap_groups": {str(k): int(v) for k, v in pd.Series(groups).value_counts().sort_index().items()},
                        "faces_with_height": int(0 if heights is None else sum(l in heights.index for l in faces)),
                        "notebook_paths": paths}


def _sku(x) -> str:
    """SKU join key: text, no surrounding spaces, no leading zeros (the clusters
    file stores SKUs as numbers, the pick reports as text)."""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.lstrip("0") or "0"


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------

@dataclass
class NotebookPlanPolicy(Policy):
    """One plan from a notebook method, applied for the whole run (the swaps
    happen at t = 0). `name` is the method key (METHODS)."""
    nb: Notebook
    inp: object
    moved: dict                       # face index -> face index (notebook's `moved`)
    plan: dict = field(default_factory=dict)
    name: str = "racksmart"

    def __post_init__(self):
        self.faces = self.inp.extra["faces"]
        self.index = {l: i for i, l in enumerate(self.faces)}
        # The notebook's evaluation World (check_month: We), for replan and move_cost.
        self.world = self.nb.World(self.inp.D, self.inp.depot, self.inp.cases, self.inp.golden,
                                   self.inp.dollars_per_unit, self.inp.cost_per_case)

    def location_for(self, location: str, t: float) -> str:
        i = self.index.get(location)
        return self.faces[self.moved[i]] if i in self.moved else location

    def moves(self) -> list[Move]:
        return [Move(0.0, self.faces[a], self.faces[b], float(self.inp.cases[a]))
                for a, b in sorted(self.moved.items())]

    def move_cost(self, moves) -> float:
        """The notebook's move_cost on the plan: COST_PER_CASE x cases in both
        locations, once per swap. (RackSmart's refill-trip term is part of its
        money test while planning, not of the move cost; check_month leaves it out too.)"""
        return float(self.nb.move_cost(self.world, self.moved))

    def route(self, locations: tuple, t: float) -> list[int]:
        """The notebook's replan: moved stops re-inserted by cheapest insertion,
        the rest keep their order. Picks of one visit stay together."""
        if not any(self.index.get(l) in self.moved for l in locations):
            return list(range(len(locations)))
        visits = []                                   # (face index, [pick positions])
        for j, l in enumerate(locations):
            i = self.index[l]
            if visits and visits[-1][0] == i:
                visits[-1][1].append(j)
            else:
                visits.append((i, [j]))
        new = self.world.replan([i for i, _ in visits], self.moved)
        unmoved, moved_from = {}, {}
        for i, js in visits:
            if i in self.moved:
                moved_from.setdefault(self.moved[i], []).extend(js)
            else:
                unmoved.setdefault(i, []).append(js)
        order = []
        for x in new:
            if x in moved_from:
                order += moved_from.pop(x)
            else:
                order += unmoved[x].pop(0)
        if sorted(order) != list(range(len(locations))):
            raise ValueError("replan lost or repeated a pick")
        return order


RackSmartPolicy = NotebookPlanPolicy          # name used before the comparison methods


def _sizes(train: list) -> list:
    return [len(train) // 2, len(train) - len(train) // 2]


def use_paths(nb: Notebook, inp) -> dict:
    """Set the notebook's file settings (CROSSWALK, CLUSTERS, CASES_PER_PALLET,
    REFILL_ORIGIN) to the ones these Inputs were built with. They are module
    globals of the notebook, read by location_clusters / refill_inputs."""
    paths = inp.extra["notebook_paths"]
    for k in PATH_SETTINGS:
        nb.ns[k] = paths.get(k)
    return dict(paths)


def plan_policy(nb: Notebook, inp, train: list) -> NotebookPlanPolicy:
    """RackSmart as the notebook plans a month: plan_month(inp, run label), over
    the one planning period in inp.tours_by_period."""
    (label,) = inp.tours_by_period
    if inp.tours_by_period[label] is not train and inp.tours_by_period[label] != train:
        raise ValueError("train must be the Inputs' planning tours")
    paths = use_paths(nb, inp)
    run_label = inp.extra["run_label"]
    moved, used = nb.plan_month(inp, run_label)
    if moved is None:
        raise ValueError(f"plan_month found no usable look-back period before {run_label!r} "
                         f"(periods {list(inp.tours_by_period)}, RESLOT {nb.RESLOT!r})")
    st = nb.SETTINGS
    lc = nb.location_clusters(inp, used, train) if (nb.CLUSTERS and st["cluster"]) else {}
    info = {"method": "racksmart", "swaps": len(moved) // 2, "driver": "notebook plan_month",
            "planned_from": list(used), "run_label": run_label, "settings": dict(st), "reslot": nb.RESLOT,
            "clusters": bool(lc), "faces_with_cluster": len(lc),
            "faces_in_split_clusters": sum("." in str(c) for c in lc.values()),
            "refill": bool(st["refill"] and nb.CASES_PER_PALLET), "swap_groups": len(set(inp.extra["groups"])),
            "notebook_paths": paths, "backtest_guard": False,
            "differs_from_tuned_run": [] if st["refill"] and nb.CASES_PER_PALLET else [REFILL_OFF]}
    return NotebookPlanPolicy(nb, inp, dict(moved), info, name="racksmart")


def plan_comparison(nb: Notebook, inp, train: list) -> dict:
    """The comparison methods, wired as PREVIOUS_DRIVER wired them, on one planner:
    detour+cost (for n_det), classic ABC with n_ref = n_det, ABC with cost
    test, family grouping with n_det and the cost test."""
    sizes = _sizes(train)
    Wp = nb.World(inp.D, inp.depot, inp.cases, inp.golden, inp.dollars_per_unit, inp.cost_per_case)
    pl = nb.PlannerPlus(Wp, train, part_sizes=sizes, stable=False)
    _, n_det, _ = pl.plan("detour+cost")
    plans = {"classic_abc": pl.plan("classic_abc", n_ref=n_det),
             "abc_cost_test": pl.plan("velocity+cost"),
             "family_grouping": pl.family_grouping(n_det, cost_test=True)}
    out = {}
    for key, (moved, n, _) in plans.items():
        info = {"method": key, "swaps": int(n), "detour_swaps_n_ref": int(n_det), "sizes": sizes,
                "stable": False, "backtest_guard": False,
                "cost_test": key != "classic_abc", "driver": PREVIOUS_DRIVER}
        out[key] = NotebookPlanPolicy(nb, inp, dict(moved), info, name=key)
    return out


def plan_method(nb: Notebook, inp, train: list, method: str) -> NotebookPlanPolicy:
    if method not in METHODS:
        raise ValueError(f"unknown notebook method {method!r}; available: {sorted(METHODS)}")
    return plan_policy(nb, inp, train) if method == "racksmart" else plan_comparison(nb, inp, train)[method]


def workbook_replay(policy: NotebookPlanPolicy, tours) -> dict:
    """The notebook's own scoring (check_month) on the simulated week's
    tours: feet saved by the plan, valued with its cost model."""
    test = visit_sequences(tours, policy.index)
    W = policy.world
    saved, before, share = policy.nb.replay(W, test, policy.moved)
    cost = policy.move_cost(None)
    return {"what": f"notebook scoring of {METHODS[policy.name]['label']} (feet and $), not the SimPy timing",
            "tours": len(test), "feet_before": float(before), "feet_saved": float(saved),
            "walking_cut_pct": round(100 * share, 1), "gross_usd": round(saved * W.dpu, 2),
            "move_cost_usd": round(cost, 2), "net_usd": round(saved * W.dpu - cost, 2)}


def load_racksmart(cfg: dict, net, method: str = "racksmart"):
    """Plan one notebook method; returns (policy, manifest info, input files).
    Takes no run-window events: the plan can only see the planning inputs."""
    nb, inp, train, base, files = prepare_planning(cfg, net)
    policy = plan_method(nb, inp, train, method)
    # every input the plan saw, with the end of its data; engine.check_planning_inputs
    # refuses any that ends after the run start unless config names it an exception
    policy.planning_inputs = base["planning_inputs"]
    return policy, method_info(policy, base), files


def method_info(policy: NotebookPlanPolicy, base: dict) -> dict:
    m = METHODS[policy.name]
    cases_gap = [CASES_GAPS[base["move_cases"]["basis"]]] if "move_cases" in base else []
    rs_gaps = []
    if policy.name == "racksmart":
        if not (base.get("inputs") or {}).get("faces_with_height"):
            rs_gaps.append(HEIGHT_GAP + ": racksmart.rack_heights not set (swap groups = aisle fence only)")
        rs_gaps += policy.plan.get("differs_from_tuned_run", [])
    return {**base, "method": policy.name, "label": m["label"], "uses_golden_zone": m["uses_golden_zone"],
            "plan": policy.plan,
            "moves": len(policy.moves()), "move_cost_usd": policy.move_cost(None),
            "known_gaps": GAPS + cases_gap + rs_gaps + m["gaps"]}


def cluster_cutoff(clusters_path) -> tuple[pd.Timestamp, dict]:
    """The exclusive data cutoff the affinity scripts recorded next to the
    clusters file (<stem>_meta.json). Refuses clusters with no record, or a
    record written for a different file."""
    path = Path(clusters_path)
    meta_path = path.with_name(path.stem + "_meta.json")
    if not meta_path.exists():
        raise ValueError(f"{path} has no {meta_path.name}: its data cutoff is unknown. Rebuild the clusters "
                         "with src/affinity (build_matrix.py CUTOFF, then frequent_itemsets.py, affinity_graph.py)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("sku_clusters_sha256_16") != file_hash(path):
        raise ValueError(f"{meta_path.name} was written for a different {path.name}; rerun affinity_graph.py")
    return pd.Timestamp(meta["cutoff_exclusive"]), meta


def prepare_planning(cfg: dict, net):
    """Read the planning inputs named in config.racksmart (shared by every
    notebook method) and build the notebook's Inputs. Returns
    (notebook, inputs, planning tours, manifest info, input files to hash).
    Nothing from the run window is passed in."""
    from ..io.wms_picks import load_wms_picks
    from .engine import window_bounds
    from .workload import build_workload

    rc, scope = cfg["racksmart"], cfg["scope"]
    nb = load_notebook(rc["notebook"])
    start, end = window_bounds({"window": rc["planning_window"]})
    # Only the planning month is read out of the 365-day reports, plus one day before
    # it so the window rule can see tours that start before it. Nothing at or after
    # the planning end is read: a tour that runs past it is cut there.
    read = (start - pd.Timedelta(days=1), end)
    parts = {p: load_wms_picks(p, rc["planning_ts_format"], window=read) for p in rc["planning_picks"]}
    per_file = {str(p): len(df) for p, df in parts.items()}
    if not any(per_file.values()):
        raise ValueError(f"no picks between {read[0].date()} and {read[1].date()} in {list(per_file)}: "
                         "check racksmart.planning_window and planning_picks")
    planning = pd.concat([df for df in parts.values() if len(df)], ignore_index=True)
    n_raw = len(planning)
    planning = planning.drop_duplicates("event_id").reset_index(drop=True)
    plan_work = build_workload(planning, net, scope, cfg["tour"], start, end)

    faces = structural_faces(net, scope)
    items = face_items(planning, scope, end)
    files = [rc["notebook"], *rc["planning_picks"]]
    picks_end = str(end)
    inputs = {
        "planning_picks": {"end": picks_end, "files": list(rc["planning_picks"])},
        "faces": {"end": None, "structural": True, "rule": FACE_RULE},
        "face_items": {"end": picks_end, "rule": FACE_ITEM_RULE},
    }

    mode = rc.get("move_cases", "inventory_snapshot")
    if mode == "inventory_snapshot":
        if not rc.get("inventory_as_of"):
            raise ValueError("racksmart.inventory_as_of is required with move_cases: inventory_snapshot "
                             "(the snapshot's date is what the lookahead check compares)")
        inventory = pd.read_csv(rc["inventory_path"])                    # as load_york reads it
        cases, cases_info = cases_from_inventory(inventory, faces)
        inputs["move_cases"] = {"end": str(pd.Timestamp(rc["inventory_as_of"])), "basis": mode,
                                "file": rc["inventory_path"]}
        files.append(rc["inventory_path"])
    elif mode == "pallet_fill":
        pf = rc.get("pallet_fill") or {}
        cases, cases_info = cases_from_pallet_fill(planning, scope, end, items, faces,
                                                   pf.get("face_fill", 0.75), pf.get("cases_per_pallet"))
        inputs["move_cases"] = {"end": picks_end, "basis": mode,
                                "rule": "cases per pallet of the face's item x face_fill, from planning picks"}
    else:
        raise ValueError(f"racksmart.move_cases must be one of {MOVE_CASES}, got {mode!r}")

    cluster_meta = None
    if rc.get("clusters_path"):
        cutoff, cluster_meta = cluster_cutoff(rc["clusters_path"])
        inputs["clusters"] = {"end": str(cutoff), "file": rc["clusters_path"],
                              "face_item_rule": CLUSTER_ITEM_RULE, "face_items_end": picks_end}
        files.append(rc["clusters_path"])

    heights, heights_info = None, None
    master = (rc.get("rack_heights") or {}).get("location_master")
    if master:
        from ..io.locations import load_locations
        heights, heights_info = heights_from_location_master(load_locations(master), faces)
        inputs["rack_heights"] = {"end": None, "structural": True, "file": master, "rule": HEIGHT_RULE}
        files.append(master)

    refill = rc.get("refill") or {}
    if refill.get("cases_per_pallet_path"):
        if not refill.get("as_of"):
            raise ValueError("racksmart.refill.as_of is required when refills are on: the last date of the "
                             "data the refill files were derived from (what the lookahead check compares)")
        inputs["refill"] = {"end": str(pd.Timestamp(refill["as_of"])),
                            "files": [p for p in (refill["cases_per_pallet_path"], refill.get("origin_path")) if p]}
        files += inputs["refill"]["files"]
    else:
        refill = {}

    in_window = _in_scope_picks(planning, scope)
    in_window = in_window[(in_window["ts"] >= start) & (in_window["ts"] < end)]
    labels = (str(start.date()), str(end.date()))     # planning period, run start (= planning end)
    inp, train, built = build_inputs(nb, faces, net, cfg["trip_ends"]["start_stop"], cases,
                                     rc.get("clusters_path"), plan_work.tours, plan_picks=in_window,
                                     heights=heights, refill=refill, labels=labels)
    base = {"notebook": nb.info, "planning_window": dict(rc["planning_window"]),
            "planning_inputs": inputs,
            "planning_rows": {"read_window": [str(read[0].date()), str(read[1].date())],
                              "read": n_raw, "per_file": per_file, "after_dedup_by_event_id": len(planning)},
            "planning_workload": {k: plan_work.counts[k] for k in
                                  ("picks_kept", "picks_dropped", "dropped_by_reason", "tours")},
            "inputs": built, "move_cases": cases_info,
            "faces_planned_but_not_picked_in_planning": int(len(set(faces) - set(items.index))),
            "rack_heights": heights_info,
            "adapter_differences": ADAPTER_DIFFERENCES}
    if cluster_meta is not None:
        base["clusters_meta"] = cluster_meta
    return nb, inp, train, base, files


def golden_table(policy: NotebookPlanPolicy) -> pd.DataFrame:
    """The plan's golden-zone faces (Inputs.golden: load_york's nearest 10.8% to the depot)."""
    return pd.DataFrame({"location": [policy.faces[i] for i in policy.inp.golden]})


def swap_table(policy: NotebookPlanPolicy) -> pd.DataFrame:
    """The notebook's swap_list for the plan. An empty plan keeps the columns
    (swap_list builds its frame from rows, so it would have none)."""
    out = policy.nb.swap_list(policy.inp, policy.moved)
    return out if len(out.columns) else pd.DataFrame(columns=["location_a", "location_b", "cases_a", "cases_b"])
