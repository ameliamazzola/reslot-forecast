"""Visit order for generated tours: how York would sequence an order's picks.

A generated order's lines come out in SKU-draw order, which is random; real
tours are visited in the order the operator scanned them. A routing rule puts a
generated tour's picks in a York-like order once, when the source builds the
tour, so every policy receives the same sequence -- exactly as every policy
receives a real tour's observed scan order in replay (static keeps it; the
notebook methods re-insert moved stops with the notebook's replan).

Rules (all on the travel network's stops and crosswalk; feet from net.feet):

  draw               as drawn (the generator's order; random)
  work_path          ascending WMS WORK_PATH from the location master: the
                     sequence the WMS routes pickers through (decisions
                     2026-09-14: median |Spearman| 0.928 against scan order)
  aisle_serpentine   aisles in sweep order from the end nearer dispatch; the
                     first aisle visited from its front (bay 1, the dock end =
                     largest y) toward the back, the next back to front, ...
  nearest_neighbour  from the dispatch stop, always the nearest unvisited stop
                     (network feet); ties in input order

Picks at one stop stay together in every rule. Which rule is used is chosen
by scripts/calibrate_routing.py: each rule re-orders the real planning-month
tours, and the rule whose feet per pick comes closest to the tours' actual scan
order wins (config generator.routing.rule: calibrated reads that choice).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RULES = ("draw", "work_path", "aisle_serpentine", "nearest_neighbour")


class Router:
    def __init__(self, net, scope: dict, dispatch_stop: str, work_path: pd.Series | None = None):
        self.net, self.dispatch = net, dispatch_stop
        faces = net.usable_faces(scope["zone"], scope["slot_type"])
        self.stop_of = faces["stop_id"].to_dict()
        self.aisle_of = faces["aisle"].to_dict()
        xw = net.crosswalk.loc[faces.index]
        self.y_of = pd.to_numeric(xw["stop_y_in"], errors="coerce").to_dict()
        x = pd.to_numeric(xw["stop_x_in"], errors="coerce")
        ax = x.groupby(xw["aisle"]).mean().sort_values()
        aisles = list(ax.index)
        # sweep direction: start from the end whose first aisle is nearer dispatch
        near = {a: min(net.feet(dispatch_stop, s) for s in faces.loc[faces["aisle"] == a, "stop_id"]) for a in
                (aisles[0], aisles[-1])} if aisles else {}
        if aisles and near[aisles[-1]] < near[aisles[0]]:
            aisles = aisles[::-1]
        self.aisle_rank = {a: i for i, a in enumerate(aisles)}
        self.work_path = None if work_path is None else work_path.to_dict()

    def order(self, locations: list, rule: str) -> list[int]:
        """A permutation of range(len(locations)): the visit order under `rule`."""
        n = len(locations)
        if rule == "draw" or n < 2:
            return list(range(n))
        if rule == "work_path":
            if self.work_path is None:
                raise ValueError("routing rule work_path needs the location master's WORK_PATH")
            key = [(self.work_path.get(l, np.inf), i) for i, l in enumerate(locations)]
            key = [(np.inf if pd.isna(k) else k, i) for k, i in key]
            return [i for _, i in sorted(key)]
        if rule == "aisle_serpentine":
            visited = sorted({self.aisle_of[l] for l in locations}, key=lambda a: self.aisle_rank[a])
            k = {a: j for j, a in enumerate(visited)}
            def key(i):
                l = locations[i]
                y = self.y_of[l]
                return (k[self.aisle_of[l]], -y if k[self.aisle_of[l]] % 2 == 0 else y, i)
            return sorted(range(n), key=key)
        if rule == "nearest_neighbour":
            stops = [self.stop_of[l] for l in locations]
            left, out, here = list(range(n)), [], self.dispatch
            while left:
                j = min(left, key=lambda i: (self.net.feet(here, stops[i]), i))
                same = [i for i in left if stops[i] == stops[j]]     # the whole stop at once
                out += same
                left = [i for i in left if stops[i] != stops[j]]
                here = stops[j]
            return out
        raise ValueError(f"unknown routing rule {rule!r}; one of {RULES}")

    def aisle_changes(self, locations: list) -> int:
        """Consecutive picks in different aisles, as the engine marks cross_aisle."""
        return int(sum(self.aisle_of[a] != self.aisle_of[b] for a, b in zip(locations, locations[1:])))

    def feet(self, locations: list) -> float:
        """Pick-to-pick feet in the given order (dispatch legs excluded), as the engine sums them."""
        s = [self.stop_of[l] for l in locations]
        return float(sum(self.net.feet(a, b) for a, b in zip(s, s[1:])))


def work_path_series(location_master) -> pd.Series:
    from ..io.locations import load_locations
    m = load_locations(location_master)
    return m.set_index("location")["path"]


def resolve_rule(routing: dict | None, run_start=None) -> tuple[str, dict]:
    """The rule to use and what the manifest records about it. rule 'calibrated'
    reads calibrate_routing.py's output (its window must end by the run start)."""
    routing = routing or {}
    rule = routing.get("rule", "draw")
    if rule != "calibrated":
        if rule not in RULES:
            raise ValueError(f"generator.routing.rule must be 'calibrated' or one of {RULES}, got {rule!r}")
        return rule, {"rule": rule, "how": "set in config"}
    path = Path(routing["calibration_path"])
    if not path.exists():
        raise ValueError(f"{path} not found: run scripts/calibrate_routing.py (or set generator.routing.rule)")
    cal = json.loads(path.read_text(encoding="utf-8"))
    if run_start is not None and pd.Timestamp(cal["window"][1]) > pd.Timestamp(run_start):
        raise ValueError(f"routing calibration window ends {cal['window'][1]}, after the run start {run_start}")
    return cal["chosen"], {"rule": cal["chosen"], "how": "calibrated", "calibration": str(path),
                           "calibration_window": cal["window"], "feet_per_pick": cal["feet_per_pick"]}


CANDIDATES = ("work_path", "aisle_serpentine", "nearest_neighbour")


def calibrate(tours: list, router: Router, rules=RULES, seed: int = 0) -> dict:
    """Re-order each real tour (>= 2 picks) under each rule; compare pick-to-pick
    feet per pick with the tours' actual scan order. Chosen: the candidate rule
    (not 'draw') whose feet per pick is closest to actual.

    'draw' on a real tour would just be its scan order, so here it is a seeded
    random shuffle -- what a generated tour without routing looks like. It is
    reported as the baseline, never chosen."""
    tours = [t for t in tours if len(t.picks) >= 2]
    picks = sum(len(t.picks) for t in tours)
    rng = np.random.default_rng(seed)
    actual_ft, rule_ft, rho = 0.0, {r: 0.0 for r in rules}, {r: [] for r in rules}
    actual_ac, rule_ac, lines = 0, {r: 0 for r in rules}, 0
    for t in tours:
        locs = [p.location for p in t.picks]
        lines += t.n_lines
        actual_ft += router.feet(locs)
        actual_ac += router.aisle_changes(locs)
        for r in rules:
            o = [int(i) for i in rng.permutation(len(locs))] if r == "draw" else router.order(locs, r)
            rule_ft[r] += router.feet([locs[i] for i in o])
            rule_ac[r] += router.aisle_changes([locs[i] for i in o])
            if len(o) >= 3:
                rank = np.empty(len(o))
                rank[o] = np.arange(len(o))
                rho[r].append(pd.Series(rank).corr(pd.Series(np.arange(len(o))), method="spearman"))
    actual = actual_ft / picks if picks else 0.0
    per = {r: rule_ft[r] / picks if picks else 0.0 for r in rules}
    cands = [r for r in rules if r in CANDIDATES]
    chosen = min(cands, key=lambda r: (abs(per[r] - actual), cands.index(r)))
    per_line = lambda x: x / lines if lines else 0.0
    return {"tours": len(tours), "picks": picks, "lines": lines, "rows_per_line": picks / lines if lines else None,
            "feet_per_line": {"actual_scan_order": per_line(actual_ft), **{r: per_line(rule_ft[r]) for r in rules}},
            "aisle_changes_per_line": {"actual_scan_order": per_line(actual_ac),
                                       **{r: per_line(rule_ac[r]) for r in rules}},
            "feet_per_pick": {"actual_scan_order": actual, **per},
            "abs_diff_from_actual": {r: abs(per[r] - actual) for r in rules},
            "median_spearman_with_scan_order": {r: (float(np.nanmedian(v)) if v else None) for r, v in rho.items()},
            "chosen": chosen, "candidates": cands, "draw_seed": seed,
            "rule": "chosen = the candidate rule whose pick-to-pick feet per pick is closest to the actual scan "
                    "order's; draw = seeded random order, the no-routing baseline, never chosen"}
