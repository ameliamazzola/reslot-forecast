"""
Loader for the C&D order-line extract -- the real demand signal.

STATUS: STUB. The extract has not been received. This file exists so the
pipeline downstream of it can be built and tested today against
synthetic.py, which emits the identical schema.

When the extract arrives:
  1. Fill RENAME with the analyst's actual column names.
  2. Run scripts/validate_extract.py against the 5-row sample FIRST.
  3. Nothing else in the repo should need to change. If it does, the schema
     contract leaked and that is a bug worth fixing before going further.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schemas import DEMAND_LINES, validate

# TODO: fill in once the analyst confirms table/column names.
RENAME: dict[str, str] = {
    # "ORDER_NBR":        "order_id",
    # "ORDER_LINE_NBR":   "order_line",
    # "ITEM_NBR":         "item_id",
    # "ORDER_RECEIVED_DT":"order_received_ts",
    # "REQ_SHIP_DT":      "requested_ship_date",
    # "QTY_ORDERED":      "qty_ordered_eaches",
}


def load_demand(path: str | Path, strict: bool = False) -> pd.DataFrame:
    """Read the order-line extract and return canonical DEMAND_LINES.

    strict=False while the extract is still partial. Flip to True once the
    full pull is in hand -- after that, a missing column should stop the run.
    """
    path = Path(path)
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else (
        pd.read_excel(path) if path.suffix.lower().startswith(".xls") else pd.read_csv(path)
    )
    df = raw.rename(columns=RENAME)
    return validate(df, DEMAND_LINES, "demand_lines", strict=strict)


def quality_report(demand: pd.DataFrame) -> dict:
    """Checks to run the moment real data lands. Each one maps to a question
    we already asked the analyst -- this is how we verify the answers."""
    d = demand
    return {
        "rows": len(d),
        "date_min": d["order_received_ts"].min(),
        "date_max": d["order_received_ts"].max(),
        "months_covered": d["order_received_ts"].dt.to_period("M").nunique(),
        "n_items": d["item_id"].nunique(),
        "n_orders": d["order_id"].nunique(),
        # Did they keep cancels/shorts, or filter to shipped-only?
        "pct_cancelled": (d["status"] == "cancelled").mean() * 100,
        "pct_with_short": (d["qty_short_eaches"].fillna(0) > 0).mean() * 100,
        # Is order_received_ts genuinely distinct from ship, or a copy of it?
        "pct_received_eq_shipped": (
            d["order_received_ts"].dt.date == d["actual_ship_ts"].dt.date
        ).mean() * 100,
        "null_received_ts_pct": d["order_received_ts"].isna().mean() * 100,
        # Intermittency: what share of (item, day) cells are zero? Drives
        # whether SARIMA is even the right family for the C-class tail.
        "median_lines_per_order": d.groupby("order_id").size().median(),
    }
