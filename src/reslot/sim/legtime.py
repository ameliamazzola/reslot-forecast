"""
Leg-time models the simulator can use (registered in engine.LEG_TIME_MODELS).

network_feet -- the 2026-09-14 model refit on network feet (scripts/refit_travel_feet.py
on branch travel/refit-feet):

    seconds = fixed + per_case * qty + cross_aisle * I[aisle changed] + distance(feet)

It is read from the refit's own outputs, nothing is fitted or typed in here:

    travel_cost_ft.csv   delta_ft -> seconds_same_aisle (fixed + distance) and
                         seconds_cross_aisle (+ cross_aisle), the fitted curve on a grid
    manifest.json        coefficients (per_case_s), the network it was fitted on, and
                         the 2026-11-02 review outcome

and it refuses to load unless that review outcome is "adopted" and the refit ran on
the same distance table as the sim. Between grid points the curve is interpolated
linearly; beyond the last point it is held flat (as the fitted isotonic curve is).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..travel.artifacts import DISTANCE_FILE
from ..utils.manifest import file_hash

SOURCE = "scan_interval_proxy"


class LegModelError(ValueError):
    pass


def _input_hash(manifest: dict, filename: str):
    for i in manifest.get("inputs", []):
        if Path(i["path"]).name == filename:
            return i["sha256_16"]
    return None


class NetworkFeetLegModel:
    name = "network_feet"

    def __init__(self, cfg: dict):
        lt = cfg["leg_time"]
        man_path, table_path = Path(lt["calibration_manifest"]), Path(lt["cost_table"])
        for p in (man_path, table_path):
            if not p.exists():
                raise LegModelError(f"{p} not found: run scripts/refit_travel_feet.py (branch travel/refit-feet)")
        m = json.loads(man_path.read_text())

        review = m.get("review") or {}
        if review.get("outcome") != "adopted":
            raise LegModelError(f"{man_path}: review outcome is {review.get('outcome')!r}; the sim only uses a "
                                "leg model the 2026-11-02 review adopted")
        if m.get("distance_unit") != "ft":
            raise LegModelError(f"{man_path}: not a network-feet fit (distance_unit {m.get('distance_unit')!r})")
        sim_dist = Path(cfg["travel"]["dir"]) / DISTANCE_FILE
        if _input_hash(m, DISTANCE_FILE) != file_hash(sim_dist):
            raise LegModelError(f"the leg model was fitted on a different {DISTANCE_FILE} than {sim_dist}")

        t = pd.read_csv(table_path)
        if set(t.get("source", pd.Series(dtype=str)).dropna()) != {SOURCE}:
            raise LegModelError(f"{table_path}: expected source={SOURCE!r}")
        coef = m["coefficients"]
        d = t["delta_ft"].to_numpy(float)
        same, cross = t["seconds_same_aisle"].to_numpy(float), t["seconds_cross_aisle"].to_numpy(float)
        if not np.allclose(cross - same, coef["cross_aisle_s"], atol=1e-6):
            raise LegModelError(f"{table_path} and {man_path} are from different fits (cross-aisle gap)")
        if np.any(np.diff(d) <= 0) or np.any(np.diff(same) < -1e-9):
            raise LegModelError(f"{table_path}: grid not increasing or curve not monotone")
        self._d, self._same, self._cross = d, same, cross
        self._per_case = float(coef["per_case_s"])
        self.files = [str(man_path), str(table_path)]
        self.provenance = {"name": self.name, "source": SOURCE, "manifest": str(man_path),
                           "cost_table": str(table_path), "fit_git_sha": m.get("git_sha"),
                           "fit_config_hash": m.get("config_hash"), "coefficients": coef,
                           "review": {k: review.get(k) for k in ("outcome", "n_legs", "n_trips",
                                                                  "mae_difference_feet_minus_work_path",
                                                                  "bootstrap")}}

    def predict(self, leg) -> float:
        curve = self._cross if leg.cross_aisle else self._same
        return float(np.interp(leg.feet, self._d, curve) + self._per_case * leg.qty)
