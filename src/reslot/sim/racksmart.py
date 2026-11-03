"""
RackSmart as a simulator policy, using notebooks/RackSmart_final.ipynb unchanged.

The algorithm is not copied. The notebook's own cells are read from the .ipynb
and executed as they are:

    settings     # ---- SETTINGS cell: WAGE, MPH, COST_PER_CASE, DEMAND_SCALE, COST_MULT
    Inputs       the input contract
    engine       World, Planner, replay, move_cost
    PlannerPlus  the search steps
    final        add_cluster_partners, racksmart_final, swap_list

Not executed: load_york (the York adapter -- it reads files and runs at import),
and the run / save cells. The final cell uses `re` and `pd`, which the notebook
imports in the load_york cell; they are supplied to the namespace as modules.

This file is the ADAPTER the notebook asks for ("a new dataset = a new adapter
that fills this in"). It fills Inputs with the same rules as load_york, on the
simulator's data:

    D        driving feet between faces + depot, from the travel network table.
             Depot = trip_ends.start_stop (Stop_Dispatch). load_york looked for a
             "PD..." stop, retired on 2026-10-19.
    faces    sorted usable case-pick faces that appear in the planning source or
             the simulated window (load_york: every mapped location in its pick file).
             All levels: load_york kept level A only; here a face at any level
             is a candidate (team decision, 2026-11-03).
    tours    one operator x one order; consecutive picks at one location are one
             visit; tours with >= 3 visits; ordered by (order, user) as load_york's
             groupby does. Case picks only (WZPCK/CS/F); load_york had no UOM filter.
    cases    CS inventory per location, summed; missing -> median location.
    golden   within 60 ft of the front (max y) of its aisle.
    dpu      $ per foot: (WAGE/3600) / (MPH*5280/3600).
    clusters location -> cluster of the product holding the most cases there.

Cost model: the notebook's, unchanged -- driving $ = feet x dpu, moving $ =
COST_PER_CASE x cases in both swapped locations (notebook move_cost). Not
enforced, because the notebook does not: the PY/PZ tall/short slot rule.
"""

from __future__ import annotations

import json
import os
import re
import sys
import types
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..utils.manifest import file_hash
from .policy import Move, Policy

# marker -> how a cell is recognised. Exactly one code cell must match each.
CELLS = {
    "settings": lambda s: s.lstrip().startswith("# ---- SETTINGS"),
    "inputs": lambda s: "class Inputs" in s,
    "engine": lambda s: "class World" in s and "def replay" in s and "def move_cost" in s,
    "plus": lambda s: "class PlannerPlus" in s,
    "final": lambda s: "def racksmart_final" in s,
}
NEVER_RUN = ("def load_york", "def run_month", ".to_csv(")
REQUIRED = ("Inputs", "World", "Planner", "PlannerPlus", "replay", "move_cost",
            "racksmart_final", "swap_list", "add_cluster_partners",
            "WAGE", "MPH", "COST_PER_CASE", "DEMAND_SCALE", "COST_MULT", "REPEAT_MIN")
# Where the adapter deliberately differs from the notebook's load_york. Written
# into every RackSmart manifest next to the known gaps.
ADAPTER_DIFFERENCES = [
    "depot = trip_ends.start_stop (Stop_Dispatch); load_york looked for a retired PD stop",
    "case picks only (WZPCK/CS/F); load_york had no zone/UOM filter",
    "faces at all levels; load_york kept crosswalk level A only",
    "faces = usable faces seen in the planning month or the simulated week; "
    "load_york used every mapped location in its pick file (a year)",
]
GAPS = ["PY/PZ tall/short slot rule not enforced (the notebook does not enforce it)",
        "case counts from one inventory snapshot, applied to the planning month and the simulated week",
        "visit order after swaps: the notebook's cheapest-insertion replan, not a WMS re-route"]

# The notebook's methods, as its run_month compares them. Each is planned once,
# before the simulated week, on the same planning tours, Inputs and cost model.
_COMPARE = ("run as the notebook's run_month runs its comparison methods: "
            "no stability setting (stable=False), no backtest guard")
_FIXED_N = ("swap count fixed to the detour method's count (run_month's n_ref / n_det), "
            "not chosen by the method")
METHODS = {
    # uses_golden_zone: the method picks swap partners from Inputs.golden (Planner.quietest_golden)
    "racksmart":       {"label": "RackSmart (final)", "uses_golden_zone": False, "gaps": []},
    "classic_abc":     {"label": "Classic ABC (no cost test)", "uses_golden_zone": True,
                        "gaps": [_COMPARE, _FIXED_N, "no cost test: a swap is kept even when it costs "
                                 "more than the driving it saves"]},
    "abc_cost_test":   {"label": "ABC with cost test", "uses_golden_zone": True, "gaps": [_COMPARE]},
    "family_grouping": {"label": "Family grouping + cost test", "uses_golden_zone": False,
                        "gaps": [_COMPARE, "location families from co-picks within tours (notebook's "
                                 "family_grouping), not the affinity clusters in sku_clusters.csv",
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


def load_notebook(path) -> Notebook:
    """Execute the algorithm cells of RackSmart_final.ipynb, unchanged."""
    nb = json.loads(open(path, encoding="utf-8").read())
    code = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    chosen = {}
    for key, match in CELLS.items():
        hits = [(i, s) for i, s in code if match(s)]
        if len(hits) != 1:
            raise ValueError(f"{path}: expected one '{key}' cell, found {len(hits)}")
        chosen[key] = hits[0]
    for key, (i, s) in chosen.items():
        bad = [m for m in NEVER_RUN if m in s]
        if bad:
            raise ValueError(f"{path}: cell {i} ('{key}') also contains {bad}; refusing to run it")
    # A real module, registered, so the notebook's @dataclass can resolve itself.
    module = types.ModuleType("racksmart_notebook")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ns.update({"re": re, "pd": pd})
    for key, (i, s) in sorted(chosen.items(), key=lambda kv: kv[1][0]):   # notebook order
        exec(compile(s, f"{path}#cell{i}", "exec"), ns)
    missing = [n for n in REQUIRED if n not in ns]
    if missing:
        raise ValueError(f"{path}: notebook no longer defines {missing}")
    info = {"path": str(path), "sha256_16": file_hash(path),
            "cells_run": {k: i for k, (i, _) in chosen.items()},
            "constants": {k: ns[k] for k in ("WAGE", "MPH", "COST_PER_CASE", "DEMAND_SCALE",
                                             "COST_MULT", "REPEAT_MIN")},
            "env_VISITED_ONLY": os.environ.get("VISITED_ONLY", "1")}
    return Notebook(ns, info)


# --------------------------------------------------------------------------
# Adapter: Inputs from the simulator's data, load_york's rules
# --------------------------------------------------------------------------

def visit_sequences(tours, index: dict) -> list[list[int]]:
    """load_york's tour rule on workload tours: consecutive picks at one
    location are one visit; keep tours with >= 3 visits; (order, user) order."""
    out = []
    for t in sorted(tours, key=lambda t: (t.order_id, t.user_id)):
        seq = []
        for p in t.picks:
            i = index[p.location]
            if not seq or seq[-1] != i:
                seq.append(i)
        if len(seq) >= 3:
            out.append(seq)
    return out


def build_inputs(nb: Notebook, faces: list, net, depot: str, inventory: pd.DataFrame,
                 clusters: pd.DataFrame | None, plan_tours: list):
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

    inv = inventory[inventory.UOM == "CS"]
    cases_by = inv.assign(l=inv.LOCATION.astype(str).str.strip()).groupby("l").QTY.sum()
    typical = float(np.median(cases_by.values))
    cases = [float(cases_by.get(l, typical)) for l in faces]

    f = net.crosswalk.loc[faces, ["stop_x_in", "stop_y_in"]].astype(float)
    ymax = f.groupby(f.stop_x_in.round(0)).stop_y_in.transform("max")
    golden = [int(i) for i in np.where(((ymax - f.stop_y_in).abs() / 12.0 <= 60).values)[0]]

    dpu = (nb.WAGE / 3600.0) / (nb.MPH * 5280 / 3600.0)

    loc_cluster = {}
    if clusters is not None:
        cof = dict(zip(clusters.sku.astype(int), clusters.cluster.astype(int)))
        top = (inv.assign(l=inv.LOCATION.astype(str).str.strip()).sort_values("QTY", ascending=False)
               .drop_duplicates("l").set_index("l").ITEM)
        loc_cluster = {idx[l]: cof[int(i)] for l, i in top.items() if l in idx and int(i) in cof}

    train = visit_sequences(plan_tours, idx)
    inp = nb.Inputs("york", D, n, cases, golden, dpu, nb.COST_PER_CASE, {"plan": train},
                    unit_note="feet", extra=dict(faces=faces, loc_cluster=loc_cluster))
    return inp, train, {"faces": n, "planning_tours": len(train), "golden": len(golden),
                        "faces_with_inventory": int(sum(l in cases_by.index for l in faces)),
                        "typical_cases": typical, "faces_with_cluster": len(loc_cluster),
                        "depot": depot}


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
        # The notebook's evaluation World (run_month: We), for replan and move_cost.
        self.world = self.nb.World(self.inp.D, self.inp.depot, self.inp.cases, self.inp.golden,
                                   self.inp.dollars_per_unit * self.nb.DEMAND_SCALE,
                                   self.inp.cost_per_case * self.nb.COST_MULT)

    def location_for(self, location: str, t: float) -> str:
        i = self.index.get(location)
        return self.faces[self.moved[i]] if i in self.moved else location

    def moves(self) -> list[Move]:
        return [Move(0.0, self.faces[a], self.faces[b], float(self.inp.cases[a]))
                for a, b in sorted(self.moved.items())]

    def move_cost(self, moves) -> float:
        """The notebook's move_cost on the plan: COST_PER_CASE x COST_MULT x
        cases in both locations, once per swap."""
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


def plan_policy(nb: Notebook, inp, train: list) -> NotebookPlanPolicy:
    """racksmart_final as run_month calls it: two halves, notebook scales."""
    sizes = _sizes(train)
    moved, n, blocked, repeat = nb.racksmart_final(inp, train, sizes, demand_scale=nb.DEMAND_SCALE,
                                                   cost_mult=nb.COST_MULT)
    return NotebookPlanPolicy(nb, inp, dict(moved), {"method": "racksmart", "swaps": int(n),
                                                     "blocked_by_backtest_guard": bool(blocked),
                                                     "top7_repeat": repeat, "sizes": sizes},
                              name="racksmart")


def plan_comparison(nb: Notebook, inp, train: list) -> dict:
    """The notebook's comparison methods, in run_month's order on one planner:
    detour+cost (for n_det), classic ABC with n_ref = n_det, ABC with cost
    test, family grouping with n_det and the cost test."""
    sizes = _sizes(train)
    Wp = nb.World(inp.D, inp.depot, inp.cases, inp.golden,
                  inp.dollars_per_unit * nb.DEMAND_SCALE, inp.cost_per_case * nb.COST_MULT)
    pl = nb.PlannerPlus(Wp, train, part_sizes=sizes, stable=False)
    _, n_det, _ = pl.plan("detour+cost")
    plans = {"classic_abc": pl.plan("classic_abc", n_ref=n_det),
             "abc_cost_test": pl.plan("velocity+cost"),
             "family_grouping": pl.family_grouping(n_det, cost_test=True)}
    out = {}
    for key, (moved, n, _) in plans.items():
        info = {"method": key, "swaps": int(n), "detour_swaps_n_ref": int(n_det), "sizes": sizes,
                "stable": False, "backtest_guard": False,
                "cost_test": key != "classic_abc"}
        out[key] = NotebookPlanPolicy(nb, inp, dict(moved), info, name=key)
    return out


def plan_method(nb: Notebook, inp, train: list, method: str) -> NotebookPlanPolicy:
    if method not in METHODS:
        raise ValueError(f"unknown notebook method {method!r}; available: {sorted(METHODS)}")
    return plan_policy(nb, inp, train) if method == "racksmart" else plan_comparison(nb, inp, train)[method]


def workbook_replay(policy: NotebookPlanPolicy, tours) -> dict:
    """The notebook's own scoring (run_month: score) on the simulated week's
    tours: feet saved by the plan, valued with its cost model."""
    test = visit_sequences(tours, policy.index)
    W = policy.world
    saved, before, share = policy.nb.replay(W, test, policy.moved)
    cost = policy.move_cost(None)
    return {"what": f"notebook scoring of {METHODS[policy.name]['label']} (feet and $), not the SimPy timing",
            "tours": len(test), "feet_before": float(before), "feet_saved": float(saved),
            "walking_cut_pct": round(100 * share, 1), "gross_usd": round(saved * W.dpu, 2),
            "move_cost_usd": round(cost, 2), "net_usd": round(saved * W.dpu - cost, 2)}


def load_racksmart(cfg: dict, net, sim_events: pd.DataFrame, method: str = "racksmart"):
    """Plan one notebook method; returns (policy, manifest info, input files)."""
    nb, inp, train, base, files = prepare_planning(cfg, net, sim_events)
    policy = plan_method(nb, inp, train, method)
    return policy, method_info(policy, base), files


def method_info(policy: NotebookPlanPolicy, base: dict) -> dict:
    m = METHODS[policy.name]
    return {**base, "method": policy.name, "label": m["label"], "uses_golden_zone": m["uses_golden_zone"],
            "plan": policy.plan,
            "moves": len(policy.moves()), "move_cost_usd": policy.move_cost(None),
            "known_gaps": GAPS + m["gaps"]}


def prepare_planning(cfg: dict, net, sim_events: pd.DataFrame):
    """Read the planning inputs named in config.racksmart (shared by every
    notebook method) and build the notebook's Inputs. Returns
    (notebook, inputs, planning tours, manifest info, input files to hash)."""
    from ..io.wms_picks import load_wms_picks
    from .engine import window_bounds
    from .workload import build_workload

    rc, scope = cfg["racksmart"], cfg["scope"]
    nb = load_notebook(rc["notebook"])
    start, end = window_bounds({"window": rc["planning_window"]})
    # Only the planning month is read out of the 365-day reports, plus one day each
    # side so the window rule can see tours that start before it or run past its end.
    read = (start - pd.Timedelta(days=1), end + pd.Timedelta(days=1))
    parts = {p: load_wms_picks(p, rc["planning_ts_format"], window=read) for p in rc["planning_picks"]}
    per_file = {str(p): len(df) for p, df in parts.items()}
    if not any(per_file.values()):
        raise ValueError(f"no picks between {read[0].date()} and {read[1].date()} in {list(per_file)}: "
                         "check racksmart.planning_window and planning_picks")
    planning = pd.concat([df for df in parts.values() if len(df)], ignore_index=True)
    n_raw = len(planning)
    planning = planning.drop_duplicates("event_id").reset_index(drop=True)
    plan_work = build_workload(planning, net, scope, cfg["tour"], start, end)
    inventory = pd.read_csv(rc["inventory_path"])                    # as load_york reads it
    clusters = pd.read_csv(rc["clusters_path"]) if rc.get("clusters_path") else None
    faces = faces_for(net, planning, sim_events, scope=scope)
    inp, train, built = build_inputs(nb, faces, net, cfg["trip_ends"]["start_stop"], inventory,
                                     clusters, plan_work.tours)
    base = {"notebook": nb.info, "planning_window": dict(rc["planning_window"]),
            "planning_rows": {"read_window": [str(read[0].date()), str(read[1].date())],
                              "read": n_raw, "per_file": per_file, "after_dedup_by_event_id": len(planning)},
            "planning_workload": {k: plan_work.counts[k] for k in
                                  ("picks_kept", "picks_dropped", "dropped_by_reason", "tours")},
            "inputs": built, "inventory_snapshot": rc["inventory_path"],
            "adapter_differences": ADAPTER_DIFFERENCES}
    files = [rc["notebook"], *rc["planning_picks"], rc["inventory_path"]]
    if rc.get("clusters_path"):
        files.append(rc["clusters_path"])
    return nb, inp, train, base, files


def golden_table(policy: NotebookPlanPolicy) -> pd.DataFrame:
    """The plan's golden-zone faces (Inputs.golden, load_york's 60 ft rule)."""
    return pd.DataFrame({"location": [policy.faces[i] for i in policy.inp.golden]})


def swap_table(policy: NotebookPlanPolicy) -> pd.DataFrame:
    """The notebook's swap_list for the plan."""
    return policy.nb.swap_list(policy.inp, policy.moved)


def faces_for(net, *event_frames, scope: dict) -> list:
    """Usable case-pick faces that appear in any of the given EVENTS frames."""
    usable = net.usable_faces(scope["zone"], scope["slot_type"]).index
    seen = set()
    for ev in event_frames:
        s = ev[(ev["event_class"] == scope["event_class"]) & (ev["from_zone"] == scope["zone"])
               & (ev["uom"] == scope["uom"])]
        seen |= set(s["from_loc"].dropna().astype(str))
    return sorted(seen & set(usable))
