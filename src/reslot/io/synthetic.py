"""
Synthetic demand generator emitting the DEMAND_LINES schema.

Why this exists: C&D data access is the project's single point of failure.
Every downstream component is built and tested against this generator, so the
pipeline is complete and demonstrable before the extract arrives -- and the
extract becomes a validation input rather than a prerequisite.

Every term below is a knob we set deliberately and can defend in committee.
Nothing here is "data that looks plausible"; it is a process we can write down.

    demand(sku, t) = baseline(sku) + seasonal(t) + peak_events(t) + noise

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import DEMAND_LINES, validate

# ABC structure: share of SKUs, share of volume
ABC = {"A": {"sku_share": 0.20, "vol_share": 0.80},
       "B": {"sku_share": 0.30, "vol_share": 0.15},
       "C": {"sku_share": 0.50, "vol_share": 0.05}}


def generate_demand(
    n_items: int = 300,
    days: int = 730,
    start: str = "2024-01-01",
    peak_multiplier: float = 2.5,
    n_peaks: int = 12,
    peak_width_days: int = 4,
    weekly_amplitude: float = 0.35,
    annual_amplitude: float = 0.20,
    noise_cv: float = 0.25,
    churn_events: int = 15,      # C-class SKUs that surge into A behaviour
    seed: int = 42,
) -> pd.DataFrame:
    """Return order lines at (item, day) grain, DEMAND_LINES schema.

    churn_events is the parameter that matters most: the re-slotting payoff
    lives in SKU velocity churn, not in peak-to-valley magnitude. Sweep this.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)

    # --- per-SKU baseline from ABC structure -----------------------------
    classes, baselines = [], []
    for cls, p in ABC.items():
        k = int(round(n_items * p["sku_share"]))
        per_sku = (p["vol_share"] * 1000.0) / max(k, 1)
        classes += [cls] * k
        baselines += list(rng.gamma(shape=2.0, scale=per_sku / 2.0, size=k))
    classes, baselines = classes[:n_items], np.array(baselines[:n_items])
    items = [f"SKU{i:05d}" for i in range(len(classes))]

    # --- shared calendar components --------------------------------------
    weekly = 1 + weekly_amplitude * np.sin(2 * np.pi * t / 7.0)
    annual = 1 + annual_amplitude * np.sin(2 * np.pi * t / 365.25)

    peak_days = rng.choice(days, size=n_peaks, replace=False)
    peaks = np.ones(days)
    for d in peak_days:
        lo, hi = max(0, d - peak_width_days), min(days, d + peak_width_days)
        ramp = np.exp(-0.5 * ((np.arange(lo, hi) - d) / (peak_width_days / 2)) ** 2)
        peaks[lo:hi] += (peak_multiplier - 1) * ramp

    # --- velocity churn: some C-class SKUs become hot mid-run ------------
    churn = np.ones((len(items), days))
    c_idx = [i for i, c in enumerate(classes) if c == "C"]
    for i in rng.choice(c_idx, size=min(churn_events, len(c_idx)), replace=False):
        onset = rng.integers(days // 4, days - 30)
        churn[i, onset:] = rng.uniform(5, 15)

    lam = baselines[:, None] * weekly * annual * peaks * churn
    lam = lam * rng.lognormal(0, noise_cv, size=lam.shape)
    qty = rng.poisson(np.clip(lam, 0, None))

    # --- explode to order lines ------------------------------------------
    ii, dd = np.nonzero(qty)
    df = pd.DataFrame({
        "item_id": [items[i] for i in ii],
        "order_received_ts": dates[dd],
        "qty_ordered_eaches": qty[ii, dd].astype(float),
    })
    df["order_id"] = "SO" + (rng.integers(0, len(df) // 6 + 1, len(df))).astype(str)
    df["order_line"] = df.groupby("order_id").cumcount() + 1
    df["customer_id"] = "C" + rng.integers(1, 40, len(df)).astype(str)
    df["order_type"] = "customer"
    df["status"] = "shipped"
    # ~3% shorted, ~2% cancelled -- these are demand and must not be dropped
    short = rng.random(len(df)) < 0.03
    cancel = rng.random(len(df)) < 0.02
    df["qty_short_eaches"] = np.where(short, df["qty_ordered_eaches"] * 0.3, 0.0)
    df.loc[cancel, "status"] = "cancelled"
    df["qty_shipped_eaches"] = np.where(
        cancel, 0.0, df["qty_ordered_eaches"] - df["qty_short_eaches"])
    lead = pd.to_timedelta(rng.integers(1, 4, len(df)), unit="D")
    df["requested_ship_date"] = df["order_received_ts"] + lead
    df["actual_ship_ts"] = df["requested_ship_date"]
    df["uom_ordered"] = "EA"
    df["warehouse"] = "1004"

    meta = pd.DataFrame({"item_id": items, "abc_class": classes})
    out = validate(df, DEMAND_LINES, "synthetic_demand", strict=True)
    return out.merge(meta, on="item_id", how="left")
