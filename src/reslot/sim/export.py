"""
Bundle sim runs and the travel network into one JSON file for ui/index.html.

Input: one or more run directories written by scripts/run_sim.py over the same
window and network (e.g. static and racksmart), and the travel network.
Output: a single JSON object the page reads in the browser. Nothing in it is
recomputed with new rules -- it is the runs' own outputs, regrouped by day
and by stop:

  network   stops (x, y, type, label), and per rack stop the case-pick faces,
            aisle and feet from dispatch (travel network tables)
  runs[]    per run: its manifest as written, the window's dates, and per day
            tours, picks, pick feet, dispatch feet (untimed), simulated pick
            seconds, dropped picks by reason, picks by stop; a notebook
            method's swaps with their stops, and its golden-zone stops when
            the method uses the zone
  affinity  (optional) the affinity graph from src/affinity/affinity_graph.py, unchanged

Day of a tour = the day it arrived (first kept pick scan). Day of a dropped
pick = the day of its scan, clamped into the window (a tour that runs past
midnight on the last day keeps its late picks on the last day).

The bundle is derived from C&D data: it goes under data/processed/ (gitignored).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..travel.artifacts import TravelNetwork

SCHEMA = "racksmart-ui/1"
RUN_FILES = ("manifest.json", "tours.csv", "legs.csv", "dropped_picks.csv")


class BundleError(ValueError):
    pass


def _clean(v):
    """JSON-safe: numpy scalars to Python, NaN/inf to null."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _input_hash(manifest: dict, filename: str):
    for i in manifest.get("inputs", []):
        if Path(i["path"]).name == filename:
            return i["sha256_16"]
    return None


def read_run(run_dir) -> dict:
    d = Path(run_dir)
    missing = [f for f in RUN_FILES if not (d / f).exists()]
    if missing:
        raise BundleError(f"{d}: not a sim run directory, missing {missing}")
    run = {"dir": str(d), "manifest": json.loads((d / "manifest.json").read_text()),
           "tours": pd.read_csv(d / "tours.csv", dtype={"order_id": str, "user_id": str}),
           "legs": pd.read_csv(d / "legs.csv"),
           "dropped": pd.read_csv(d / "dropped_picks.csv", dtype=str)}
    swaps = d / "swaps.csv"
    run["swaps"] = pd.read_csv(swaps, dtype={"location_a": str, "location_b": str}) if swaps.exists() else None
    golden = d / "golden_faces.csv"
    run["golden"] = pd.read_csv(golden, dtype=str)["location"].tolist() if golden.exists() else []
    if "first_stop" not in run["tours"]:
        raise BundleError(f"{d}: tours.csv has no first_stop column -- re-run scripts/run_sim.py")
    return run


def network_block(net: TravelNetwork, stops: pd.DataFrame, dispatch: str, scope: dict) -> dict:
    faces = net.usable_faces(scope["zone"], scope["slot_type"])
    per_stop = faces.groupby("stop_id").agg(faces=("aisle", "size"), aisle=("aisle", "first"))
    out = []
    for r in stops.itertuples(index=False):
        s = {"id": r.stop_id, "x": float(r.x_in), "y": float(r.y_in), "type": r.node_type,
             "label": None if pd.isna(r.label) else str(r.label)}
        if r.node_type == "rack":
            s["faces"] = int(per_stop.at[r.stop_id, "faces"]) if r.stop_id in per_stop.index else 0
            s["aisle"] = per_stop.at[r.stop_id, "aisle"] if r.stop_id in per_stop.index else None
            s["ft_from_dispatch"] = net.feet(dispatch, r.stop_id) if net.has_stop(r.stop_id) else None
        out.append(s)
    return {"dispatch": dispatch, "stops": out, "summary": net.summary()}


def run_block(run: dict, net: TravelNetwork, dates: list) -> dict:
    m, t, legs, dr = run["manifest"], run["tours"], run["legs"], run["dropped"]
    n = len(dates)
    t = t.assign(day=(t["arrival_s"] // 86400).astype(int).clip(0, n - 1))
    by_day = t.groupby("day")

    def per_day(series):
        return [float(series.get(i, 0.0)) for i in range(n)]

    legs = legs.merge(t[["tour_id", "day"]], on="tour_id", how="left")
    visits = pd.concat([t[["day", "first_stop"]].rename(columns={"first_stop": "stop"}),
                        legs[["day", "to_stop"]].rename(columns={"to_stop": "stop"})])
    sp = {int(day): {s: int(c) for s, c in g.groupby("stop").size().items()}
          for day, g in visits.groupby("day")} if len(visits) else {}
    stop_picks = [sp.get(i, {}) for i in range(n)]

    start = pd.Timestamp(dates[0])
    dr_day = ((pd.to_datetime(dr["ts"]) - start).dt.days.clip(0, n - 1)) if len(dr) else pd.Series(dtype=int)
    dropped_by_day = [{} for _ in range(n)]
    for day, reason in zip(dr_day, dr.get("reason", [])):
        dropped_by_day[int(day)][reason] = dropped_by_day[int(day)].get(reason, 0) + 1

    swaps = []
    if run["swaps"] is not None:
        stop_of = net.crosswalk["stop_id"]
        # simulated picks per location under this run's layout (where products sat during the week)
        at = pd.concat([t["first_location"], legs["to_location"]]).value_counts()
        for r in run["swaps"].itertuples(index=False):
            swaps.append({"location_a": r.location_a, "location_b": r.location_b,
                          "stop_a": stop_of.get(r.location_a), "stop_b": stop_of.get(r.location_b),
                          "cases_a": float(r.cases_a), "cases_b": float(r.cases_b),
                          # picks moved: the week's picks of both swapped products, which sit
                          # at these two locations after the swap
                          "picks": int(at.get(r.location_a, 0) + at.get(r.location_b, 0))})

    return {"name": Path(run["dir"]).name, "policy": m["policy"], "leg_time_model": m.get("leg_time_model"),
            "manifest": {k: v for k, v in m.items() if k != "config"}, "config": m.get("config"),
            "daily": {"tours": per_day(by_day.size()), "picks": per_day(by_day["n_picks"].sum()),
                      "pick_feet": per_day(by_day["pick_feet"].sum()),
                      "dispatch_feet_untimed": per_day(by_day["dispatch_feet"].sum()),
                      "sim_pick_seconds": per_day(by_day["busy_s"].sum()),
                      "dropped": dropped_by_day},
            "stop_picks": stop_picks, "swaps": swaps,
            # golden-zone stops, only for a method that picks partners from the zone
            "golden_stops": sorted(set(net.crosswalk["stop_id"].reindex(run["golden"]).dropna()))
                            if (m.get("notebook_plan") or {}).get("uses_golden_zone") else []}


def build_bundle(run_dirs: list, net: TravelNetwork, stops: pd.DataFrame, created_utc: str,
                 network_hashes: dict, affinity: dict | None = None) -> dict:
    """network_hashes: {"stop_distance_ft.csv": sha, "location_stop_crosswalk.csv": sha}
    of the network files given -- every run must have used exactly these."""
    if not run_dirs:
        raise BundleError("no run directories given")
    runs = [read_run(d) for d in run_dirs]
    first = runs[0]["manifest"]
    for r in runs:
        m = r["manifest"]
        if m["window"] != first["window"]:
            raise BundleError(f"{r['dir']}: window {m['window']} differs from {first['window']}")
        for f, h in network_hashes.items():
            if _input_hash(m, f) != h:
                raise BundleError(f"{r['dir']}: ran on a different {f} than the network given")
        # The UI compares policies tour for tour: every run must replay the same picks.
        if (m.get("inputs") or [{}])[0].get("sha256_16") != (first.get("inputs") or [{}])[0].get("sha256_16"):
            raise BundleError(f"{r['dir']}: replayed a different picks file than {runs[0]['dir']}")
    policies = [r["manifest"]["policy"] for r in runs]
    if len(set(policies)) != len(policies):
        raise BundleError(f"two runs with the same policy: {policies}")

    w = first["window"]
    dates = [str(d.date()) for d in pd.date_range(w["start"], w["end"], freq="D")]
    cfg = first.get("config") or {}
    dispatch = first.get("trip_ends", {}).get("start_stop") or cfg.get("trip_ends", {}).get("start_stop")
    scope = first.get("scope") or cfg.get("scope")
    bundle = {"schema": SCHEMA, "created_utc": created_utc, "dates": dates, "window": w,
              "network": network_block(net, stops, dispatch, scope),
              "runs": [run_block(r, net, dates) for r in runs]}
    if affinity is not None:
        if affinity.get("schema") != "racksmart-affinity/1":
            raise BundleError("affinity graph is not src/affinity/graph_export.py output")
        bundle["affinity"] = affinity          # as written by affinity_graph.py, unchanged
    return _clean(bundle)
