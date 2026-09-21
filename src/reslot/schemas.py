"""
Canonical schemas. This module is the contract between us and C&D.

Everything downstream (panel builder, forecaster, dispatcher) reads ONLY these
schemas. Nothing downstream ever sees a raw WMS column name. That means:

  * The synthetic generator and the real C&D extract are interchangeable.
  * When the real extract lands, the only file that changes is the loader.
  * A schema mismatch fails loudly at ingest, not silently three steps later.

Tables:

  EVENTS        what the warehouse DID   (OneTrack / movement log, WMS pick report)
  DEMAND_LINES  what customers ASKED FOR (order header + line, not yet received)
  ITEM_MASTER   static SKU attributes    (needed for UOM conversion)
  SLOTS         every storage location   (WMS location master)
  TRIP_LEGS     consecutive pick pairs   (the travel model's fitting unit)
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------
# EVENTS — canonical form of the OneTrack export
# --------------------------------------------------------------------------
EVENTS = {
    "event_id": "int64",          # source NO
    "ts": "datetime64[ns]",       # CREATE_DT
    "event_class": "string",      # our taxonomy: pick / putaway / replen / ...
    "transaction_type": "string", # raw, kept for audit
    "work_type": "string",        # raw, kept for audit
    "user_id": "string",
    "item_id": "string",
    "from_loc": "string",
    "to_loc": "string",
    "from_zone": "string",
    "to_zone": "string",
    "qty": "float64",             # in uom
    "uom": "string",
    "qty_eaches": "float64",      # normalised: FROM_EACH
    "lic": "string",              # pallet license plate
    "order_id": "string",
    "order_line": "Int64",
    "customer_id": "string",
    "load_id": "string",
    "lot": "string",
    "warehouse": "string",
}

# --------------------------------------------------------------------------
# DEMAND_LINES — what we are asking the WMS analyst for
# --------------------------------------------------------------------------
# One row per order line. This is the ONLY legitimate input to the forecaster.
# Writing it down now, before the data arrives, is what lets us build and test
# the whole pipeline against synthetic data and swap in the real extract later.
DEMAND_LINES = {
    "order_id": "string",
    "order_line": "Int64",
    "item_id": "string",
    "customer_id": "string",
    "order_type": "string",            # customer / transfer / production / return
    "status": "string",                # shipped / cancelled / short / open
    "order_received_ts": "datetime64[ns]",   # <-- the demand timestamp
    "requested_ship_date": "datetime64[ns]",
    "actual_ship_ts": "datetime64[ns]",
    "qty_ordered_eaches": "float64",
    "qty_shipped_eaches": "float64",
    "qty_short_eaches": "float64",     # shorts and cancels ARE demand
    "uom_ordered": "string",
    "warehouse": "string",
}

ITEM_MASTER = {
    "item_id": "string",
    "description": "string",
    "eaches_per_case": "float64",
    "cases_per_layer": "float64",
    "layers_per_pallet": "float64",
    "product_family": "string",
    "lot_controlled": "boolean",
    "storage_type": "string",          # rack / floor / hive
}

# (item_id, date) grain. The forecaster's input and output live here.
PANEL = {
    "item_id": "string",
    "date": "datetime64[ns]",
    "qty_eaches": "float64",
    "qty_cases": "float64",
    "n_lines": "Int64",
    "n_orders": "Int64",
    "source": "string",   # "demand" (real) or "pick_proxy" (contaminated, interim)
}


# --------------------------------------------------------------------------
# SLOTS -- canonical form of the WMS location master
# --------------------------------------------------------------------------
# One row per storage location in the building, whether or not it saw
# activity. Locations with no transactions are the free capacity re-slotting
# moves into, so the slot model must be built from THIS table, never from
# locations observed in events.
#
# C&D's WMS populates no x/y/z. `path` is WORK_PATH: the sequence the WMS
# routes pickers through. It is ORDINAL -- it orders slots, it does not
# measure feet between them. Validated as a travel coordinate; see
# docs/decisions.md.
SLOTS = {
    "location": "string",
    "slot_type": "string",        # F = pick face, R = reserve rack (master TYPE)
    "zone": "string",             # WORK_ZONE
    "path": "float64",            # WORK_PATH
    "putaway_zone": "string",
    "putaway_path": "float64",
    "aisle": "string",            # parsed from the code, see io/onetrack.parse_location
    "capacity": "float64",        # STD_CPCT
    "pick_uom": "string",
}

# --------------------------------------------------------------------------
# TRIP_LEGS -- consecutive pick pairs within one operator's work on one order
# --------------------------------------------------------------------------
# elapsed_s is time between pick SCANS, not a time study. It contains
# scanning, pallet building and any pause the operator took. Anything fitted
# on it carries source="scan_interval_proxy", same convention as the panel's
# "pick_proxy".
TRIP_LEGS = {
    "trip_id": "int64",
    "ts": "datetime64[ns]",
    "from_loc": "string",
    "delta_path": "float64",
    "cross_aisle": "int64",
    "qty": "float64",
    "elapsed_s": "float64",
}

class SchemaError(ValueError):
    pass


def _null_for(dtype: str):
    if dtype.startswith("float"):
        return float("nan")
    if dtype.startswith("datetime"):
        return pd.NaT
    return pd.NA


def validate(df: pd.DataFrame, schema: dict, name: str, strict: bool = True) -> pd.DataFrame:
    """Check a frame against a schema and coerce dtypes.

    strict=True  -> missing columns raise. Use for anything feeding the report.
    strict=False -> missing columns are added as null. Use while a source is
                    still partial (e.g. first C&D sample extract).
    """
    missing = [c for c in schema if c not in df.columns]
    if missing and strict:
        raise SchemaError(f"{name}: missing required columns {missing}")

    out = df.copy()
    for col in missing:
        # Null of the right kind for the target dtype. A bare pd.NA cannot be
        # cast to float64 or datetime64, which broke strict=False for any
        # source missing a float column (e.g. qty_eaches on the WMS pick report).
        out[col] = _null_for(schema[col])

    extra = [c for c in out.columns if c not in schema]
    if extra:
        # keep them, but they are not part of the contract
        pass

    for col, dtype in schema.items():
        try:
            out[col] = out[col].astype(dtype)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"{name}.{col}: cannot cast to {dtype} ({exc})") from exc

    ordered = list(schema) + extra
    return out[ordered]
