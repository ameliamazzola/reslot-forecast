"""Compare two sets of sim runs policy by policy, and say whether the ranking moved.

Used for (scripts/compare_runs.py):
  before / after   the same window run on two commits (e.g. the lookahead fix)
  sensitivity      the same commit, move_cases inventory_snapshot vs pallet_fill

A set is the run_sim.py output folders in one directory for one window. Every
input number is read from the runs' manifests. Static is the reference inside
each set (it has no plan, so it is the same under both move_cases settings and
a pallet_fill set borrows the main static run). The headline is computed from
those numbers with the same function run_sim.py uses (src/reslot/sim/headline.py),
against the set's own static run, so runs from before the headline existed
compare too.

Two rankings are reported, best first, over the notebook methods in both sets:
  headline_net   PRIMARY: net labor savings = operator-hours saved vs static x W
                 - move cost (with payback days in the table)
  notebook_net   secondary: the notebook's own feet scoring on the run's tours (net $)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import headline as H

VARIANT_SUFFIX = {"inventory_snapshot": "", "pallet_fill": "_pallet_fill"}
COLUMNS = ["policy", "swaps", "faces", "faces_with_cluster", "move_cases", "lookahead_exceptions_used",
           "move_cost_usd", "notebook_driving_cut_pct", "notebook_gross_usd", "notebook_net_usd",
           "sim_pick_feet", "sim_pick_feet_saved_vs_static", "sim_busy_hours", "sim_busy_hours_saved_vs_static",
           "labor_saved_usd", "headline_net_usd", "payback_days", "git_sha", "git_dirty"]
RANKINGS = (("headline_net", "headline_net_usd"), ("notebook_net", "notebook_net_usd"))   # primary first


def _summary(m: dict) -> dict:
    nbp = m.get("notebook_plan") or {}
    rep = nbp.get("workbook_replay") or {}
    used = (m.get("planning_inputs") or {}).get("lookahead_exceptions_used")
    return {
        "policy": m["policy"],
        "swaps": (nbp.get("plan") or {}).get("swaps", 0),
        "faces": (nbp.get("inputs") or {}).get("faces"),
        "faces_with_cluster": (nbp.get("inputs") or {}).get("faces_with_cluster"),
        # runs from before the lookahead fix used the snapshot and recorded no planning inputs
        "move_cases": (nbp.get("move_cases") or {}).get("basis", "inventory_snapshot" if nbp else None),
        "lookahead_exceptions_used": ",".join(used) if used else ("not recorded" if used is None else ""),
        "move_cost_usd": float(m["moves"]["cost"]),
        "notebook_driving_cut_pct": rep.get("walking_cut_pct", 0.0),
        "notebook_gross_usd": rep.get("gross_usd", 0.0),
        "notebook_net_usd": rep.get("net_usd", 0.0),
        "sim_pick_feet": float(m["descriptive"]["pick_feet"]),
        "sim_busy_hours": float(m["not_findings"]["busy_hours"]),
        "git_sha": str(m.get("git_sha", ""))[:7],
        "git_dirty": bool(m.get("git_dirty")),
    }


def load_set(directory, start: str, end: str, move_cases: str = "inventory_snapshot",
             rate: float | None = None) -> tuple[pd.DataFrame, list]:
    """One row per policy run in `directory` for window [start, end] with the given
    move_cases variant (static: always the main run). Returns (table, manifest paths).
    rate: W ($ per operator-hour) for the headline; None leaves the headline columns empty."""
    if move_cases not in VARIANT_SUFFIX:
        raise ValueError(f"move_cases must be one of {sorted(VARIANT_SUFFIX)}")
    tag = f"_{start.replace('-', '')}_{end.replace('-', '')}"
    rows, paths = {}, []
    for d in sorted(Path(directory).glob("sim_*")):
        mp = d / "manifest.json"
        if not mp.exists():
            continue
        name = d.name
        is_pf = name.endswith(VARIANT_SUFFIX["pallet_fill"])
        base = name[: -len(VARIANT_SUFFIX["pallet_fill"])] if is_pf else name
        if not base.endswith(tag):
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        static = m["policy"] == "static"
        if not static and is_pf != (move_cases == "pallet_fill"):
            continue
        if m["policy"] in rows:
            raise ValueError(f"two runs of {m['policy']!r} for {start}..{end} in {directory} "
                             f"({rows[m['policy']]['_dir']} and {name})")
        rows[m["policy"]] = {**_summary(m), "_dir": name}
        paths.append(mp)
    if "static" not in rows:
        raise ValueError(f"no static run for {start}..{end} in {directory}: it is the reference")
    t = pd.DataFrame(rows.values()).drop(columns="_dir")
    ref = t.set_index("policy").loc["static"]
    t["sim_pick_feet_saved_vs_static"] = ref["sim_pick_feet"] - t["sim_pick_feet"]
    t["sim_busy_hours_saved_vs_static"] = ref["sim_busy_hours"] - t["sim_busy_hours"]
    days = H.window_days({"start": start, "end": end})
    hl = [H.headline(ref["sim_busy_hours"], h, c, rate, days) if rate else {}
          for h, c in zip(t["sim_busy_hours"], t["move_cost_usd"])]
    t["labor_saved_usd"] = [x.get("labor_saved_usd") for x in hl]
    t["headline_net_usd"] = [x.get("net_usd") for x in hl]
    t["payback_days"] = [x.get("payback_days") for x in hl]
    return t[COLUMNS].sort_values("policy").reset_index(drop=True), paths


def ranking(t: pd.DataFrame, by: str) -> list:
    """Notebook methods, best first (ties broken by policy name)."""
    m = t[t["policy"] != "static"]
    return m.sort_values([by, "policy"], ascending=[False, True])["policy"].tolist()


def compare(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> tuple[pd.DataFrame, dict]:
    """Side-by-side table (policy x metric, a / b / b - a) and the rankings
    (headline_net first when both sets carry it, then notebook_net)."""
    common = sorted(set(a["policy"]) & set(b["policy"]))
    if len(common) < 2:
        raise ValueError(f"need at least static and one method in both sets; in both: {common}")
    a, b = (x[x["policy"].isin(common)].set_index("policy") for x in (a, b))
    rows = []
    for pol in common:
        for col in COLUMNS[1:]:
            va, vb = a.at[pol, col], b.at[pol, col]
            num = all(pd.api.types.is_number(v) and not pd.api.types.is_bool(v) for v in (va, vb))
            rows.append({"policy": pol, "metric": col, label_a: va, label_b: vb,
                         "change": (vb - va) if num and pd.notna(va) and pd.notna(vb) else None})
    table = pd.DataFrame(rows)
    rank = {}
    for by, col in RANKINGS:
        if a[col].isna().all() or b[col].isna().all():
            continue
        ra, rb = ranking(a.reset_index(), col), ranking(b.reset_index(), col)
        rank[by] = {label_a: ra, label_b: rb, "changed": ra != rb}
    return table, rank


def wide(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str,
         metrics=("swaps", "move_cost_usd", "sim_busy_hours_saved_vs_static", "headline_net_usd", "payback_days",
                  "notebook_net_usd")) -> pd.DataFrame:
    """One row per policy, chosen metrics as <metric> [a] / [b] -- for printing."""
    common = sorted(set(a["policy"]) & set(b["policy"]))
    a, b = (x.set_index("policy").loc[common] for x in (a, b))
    out = pd.DataFrame(index=common)
    for m in metrics:
        out[f"{m} [{label_a}]"] = a[m]
        out[f"{m} [{label_b}]"] = b[m]
    return out
