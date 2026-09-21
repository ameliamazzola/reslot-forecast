"""
Build the (item, date) demand panel -- the forecaster's input.

Two sources, and the difference between them is the point:

  from_demand()     order_received_ts. Clean demand. What we want.
  from_pick_proxy() pick timestamps from OneTrack. Contaminated: a pick happens
                    when the WMS released the wave, not when the customer
                    ordered. Usable to build and debug the pipeline before the
                    extract lands; NEVER a number that goes in the report
                    without the caveat attached.
"""

from __future__ import annotations

import warnings

import pandas as pd

from ..schemas import PANEL, validate


def _finish(g: pd.DataFrame, source: str, eaches_per_case: float | dict | None) -> pd.DataFrame:
    if isinstance(eaches_per_case, dict):
        epc = g["item_id"].map(eaches_per_case).fillna(1.0)
    else:
        epc = eaches_per_case or 1.0
    g["qty_cases"] = g["qty_eaches"] / epc
    g["source"] = source
    return validate(g, PANEL, f"panel[{source}]", strict=True)


def from_demand(
    demand: pd.DataFrame,
    include_cancelled: bool = True,
    eaches_per_case: float | dict | None = None,
) -> pd.DataFrame:
    """Daily ordered quantity per item, dated by when the order arrived.

    include_cancelled=True by default and should stay that way: a cancelled
    line is demand the DC was asked to serve. Excluding it shaves exactly the
    peaks the anticipatory policy is supposed to catch.
    """
    d = demand if include_cancelled else demand[demand["status"] != "cancelled"]
    d = d.assign(date=d["order_received_ts"].dt.normalize())
    g = (d.groupby(["item_id", "date"])
           .agg(qty_eaches=("qty_ordered_eaches", "sum"),
                n_lines=("order_line", "size"),
                n_orders=("order_id", "nunique"))
           .reset_index())
    return _finish(g, "demand", eaches_per_case)


def from_pick_proxy(events: pd.DataFrame, eaches_per_case=None) -> pd.DataFrame:
    """Interim panel from pick events. See module docstring for the caveat."""
    p = events[events["event_class"] == "pick"].copy()
    if len(p) and p["qty_eaches"].isna().all():
        # The WMS pick report has no FROM_EACH. Summing all-null gives 0, so
        # every qty column below would read zero and anything built on
        # qty_eaches (intermittency, ABC) would be silently wrong.
        warnings.warn(
            "qty_eaches is null for every pick in this source; panel qty "
            "columns will be 0. Use n_lines (pick count), e.g. "
            "intermittency(panel, value_col='n_lines').", stacklevel=2)
    p["date"] = p["ts"].dt.normalize()
    g = (p.groupby(["item_id", "date"])
           .agg(qty_eaches=("qty_eaches", "sum"),
                n_lines=("event_id", "size"),
                n_orders=("order_id", "nunique"))
           .reset_index())
    return _finish(g, "pick_proxy", eaches_per_case)


def densify(panel: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Fill missing (item, date) cells with zero.

    Required before any time-series model: a SKU with no order on Tuesday has
    demand 0, not a gap. This is also where the C-class intermittency becomes
    visible -- check the zero share here before assuming SARIMA fits the tail.
    """
    full = pd.MultiIndex.from_product(
        [panel["item_id"].unique(),
         pd.date_range(panel["date"].min(), panel["date"].max(), freq=freq)],
        names=["item_id", "date"])
    out = (panel.set_index(["item_id", "date"])
                .reindex(full)
                .reset_index())
    src = panel["source"].iloc[0] if len(panel) else "unknown"
    out[["qty_eaches", "qty_cases"]] = out[["qty_eaches", "qty_cases"]].fillna(0.0)
    out[["n_lines", "n_orders"]] = out[["n_lines", "n_orders"]].fillna(0).astype("Int64")
    out["source"] = out["source"].fillna(src)
    return out


def abc_classify(panel: pd.DataFrame, window_days: int | None = None) -> pd.DataFrame:
    """Pareto class per item by volume share. Recompute on a rolling window to
    see churn -- an item whose class changes is exactly what the anticipatory
    policy is meant to catch, and the churn rate is a headline descriptive stat.
    """
    p = panel
    if window_days:
        cutoff = p["date"].max() - pd.Timedelta(days=window_days)
        p = p[p["date"] >= cutoff]
    vol = p.groupby("item_id")["qty_eaches"].sum().sort_values(ascending=False)
    share = vol.cumsum() / vol.sum()
    cls = pd.cut(share, [0, 0.80, 0.95, 1.01], labels=["A", "B", "C"])
    return cls.rename("abc_class").reset_index()


def intermittency(panel: pd.DataFrame, value_col: str = "qty_eaches") -> pd.DataFrame:
    """Zero-day share per item. >0.7 means Croston/TSB territory, not SARIMA.

    value_col="n_lines" for sources with no eaches (the WMS pick report).
    """
    d = densify(panel)
    v = d[value_col].astype("float64")
    return (d.assign(zero=v == 0, val=v)
             .groupby("item_id")
             .agg(zero_share=("zero", "mean"), mean_qty=("val", "mean"))
             .reset_index())
