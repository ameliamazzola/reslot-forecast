"""The headline metric: net labor savings against static, and payback days.

    operator-hours  sum over tours of simulated busy time (first to last pick scan,
                    leg-time model); manifest not_findings.busy_hours
    hours saved     static operator-hours - policy operator-hours, same window,
                    same tours, same leg model
    net $           hours saved x W - move cost (the plan's one-time cost)
    payback days    move cost / (hours saved x W / window days); none when the
                    plan saves no hours (it never pays back), 0 when it moves nothing

W is config headline.labor_rate_usd_per_hour, with its source in
headline.labor_rate_source. Secondary, reported beside it: the notebook's own
feet scoring (notebook_plan.workbook_replay.net_usd).

Caveat carried with every figure: dispatch legs are untimed (decisions
2026-11-02), so operator-hours cover first-to-last scan only; queueing and the
operator count do not enter. Leg times are scan_interval_proxy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEFINITION = ("net labor savings = (static operator-hours - policy operator-hours) x W - move cost; "
              "payback days = move cost / (hours saved x W / window days). Operator-hours = simulated "
              "first-to-last pick scan per tour under the leg-time model")
CAVEAT = ("dispatch legs untimed (decisions 2026-11-02): operator-hours cover first-to-last pick scan only; "
          "leg times are scan_interval_proxy")
# what must be identical for a static run to be this run's reference
SAME_AS_STATIC = ("window", "operators", "leg_time_model", "leg_time", "scope", "trip_ends")


def labor_rate(cfg: dict) -> tuple[float, str]:
    h = cfg.get("headline") or {}
    if h.get("labor_rate_usd_per_hour") is None or not h.get("labor_rate_source"):
        raise ValueError("config headline.labor_rate_usd_per_hour and headline.labor_rate_source are required")
    w = float(h["labor_rate_usd_per_hour"])
    if w <= 0:
        raise ValueError(f"headline.labor_rate_usd_per_hour must be > 0, got {w}")
    return w, str(h["labor_rate_source"])


def headline(static_hours: float, policy_hours: float, move_cost: float, rate: float, days: int) -> dict:
    saved = float(static_hours) - float(policy_hours)
    labor = saved * rate
    move = float(move_cost)
    per_day = labor / days
    if move <= 0:
        payback = 0.0
    elif per_day <= 0:
        payback = None                           # saves nothing: never pays back
    else:
        payback = move / per_day
    return {"operator_hours_static": float(static_hours), "operator_hours_policy": float(policy_hours),
            "operator_hours_saved": saved, "labor_rate_usd_per_hour": float(rate),
            "labor_saved_usd": labor, "move_cost_usd": move, "net_usd": labor - move,
            "window_days": int(days), "labor_saved_per_day_usd": per_day,
            "payback_days": payback, "pays_back_in_window": payback is not None and payback <= days}


def window_days(window: dict) -> int:
    return (pd.Timestamp(window["end"]) - pd.Timestamp(window["start"])).days + 1


def static_mismatch(static_manifest: dict, run_extra: dict, run_inputs: dict) -> list:
    """Why a static run's manifest cannot be this run's reference ([] = it can).
    run_inputs: {file name: sha256_16} of this run's picks, network and leg-model files."""
    norm = lambda v: json.loads(json.dumps(v, sort_keys=True, default=str))   # as the manifest stores it
    why = [k for k in SAME_AS_STATIC if norm(static_manifest.get(k)) != norm(run_extra.get(k))]
    have = {Path(i["path"]).name: i["sha256_16"] for i in static_manifest.get("inputs", [])}
    why += [f"input {n}" for n, h in run_inputs.items() if have.get(n) != h]
    if static_manifest.get("policy") != "static":
        why.append("not a static run")
    return why


def read_static(path) -> dict | None:
    p = Path(path) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
