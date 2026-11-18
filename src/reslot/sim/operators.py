"""Operators per clock hour, the reference value behind config `operators`.

Definition (decisions 2026-11-02): over the clock hours of the window that have at
least one in-scope case pick (event_class pick, zone WZPCK, UOM CS), the number of
distinct USER_NAMEs who made one; the reference is the median. Not a measured
headcount: an operator picking on both sides of an hour boundary counts in both.
"""

from __future__ import annotations

import pandas as pd


def operators_per_hour(events: pd.DataFrame, scope: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Distinct case-pick operators per clock hour in [start, end)."""
    s = events[(events["event_class"] == scope["event_class"]) & (events["from_zone"] == scope["zone"])
               & (events["uom"] == scope["uom"]) & (events["ts"] >= start) & (events["ts"] < end)
               & events["user_id"].notna()]
    hourly = s.groupby(s["ts"].dt.floor("h"))["user_id"].nunique()
    if hourly.empty:
        raise ValueError(f"no in-scope case picks between {start} and {end}")
    by_day = s.groupby(s["ts"].dt.date)["user_id"].nunique()
    return {"window": [str(start), str(end)], "hours_with_case_picks": int(len(hourly)),
            "median": float(hourly.median()), "p90": float(hourly.quantile(0.9)), "max": int(hourly.max()),
            "distinct_operators_in_window": int(s["user_id"].nunique()),
            "distinct_operators_per_day_median": float(by_day.median()),
            "case_picks": int(len(s)),
            "definition": "median over clock hours with >= 1 in-scope case pick of distinct USER_NAME"}
