"""Fast tests. Run with: pytest -q

These guard the schema contract, which is the thing most likely to break
silently when the real C&D extract replaces the synthetic generator.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from reslot.features import panel as P
from reslot.forecast.baselines import SeasonalNaive
from reslot.io.onetrack import parse_location
from reslot.io.synthetic import generate_demand
from reslot.schemas import DEMAND_LINES, PANEL, SchemaError, validate


def test_synthetic_matches_contract():
    d = generate_demand(n_items=20, days=90, seed=1)
    validate(d, DEMAND_LINES, "demand", strict=True)


def test_generator_is_seeded():
    a = generate_demand(n_items=10, days=60, seed=7)
    b = generate_demand(n_items=10, days=60, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_panel_contract_and_density():
    d = generate_demand(n_items=10, days=60, seed=2)
    p = P.densify(P.from_demand(d))
    validate(p, PANEL, "panel", strict=True)
    assert len(p) == p["item_id"].nunique() * p["date"].nunique()


def test_cancelled_lines_are_demand():
    """Dropping cancels must reduce volume -- if it doesn't, the flag is dead."""
    d = generate_demand(n_items=30, days=120, seed=3)
    with_c = P.from_demand(d, include_cancelled=True)["qty_eaches"].sum()
    without = P.from_demand(d, include_cancelled=False)["qty_eaches"].sum()
    assert with_c > without


def test_forecast_shape():
    d = generate_demand(n_items=5, days=90, seed=4)
    p = P.densify(P.from_demand(d))
    f = SeasonalNaive(7).fit_predict(p, horizon=14)
    assert set(f.columns) == {"item_id", "date", "yhat"}
    assert len(f) == 5 * 14


def test_missing_column_raises_when_strict():
    with pytest.raises(SchemaError):
        validate(pd.DataFrame({"order_id": ["1"]}), DEMAND_LINES, "x", strict=True)


@pytest.mark.parametrize("code,aisle,bay", [
    ("KK038A", "KK", 38), ("GG244A", "GG", 244), ("D161E", "D", 161),
    ("RT-23", None, None), ("PL29", None, None), ("V-0028", None, None),
])
def test_location_parse(code, aisle, bay):
    got = parse_location(code)
    assert got["aisle"] == aisle and got["bay"] == bay


def test_non_strict_fills_missing_float_and_datetime():
    """strict=False must null-fill every dtype, not just the nullable ones.
    Regression: a bare pd.NA could not be cast to float64."""
    from reslot.schemas import EVENTS
    df = pd.DataFrame({"event_id": [1], "ts": [pd.Timestamp("2026-09-01")]})
    out = validate(df, EVENTS, "events", strict=False)
    assert out["qty_eaches"].isna().all() and out["qty_eaches"].dtype == "float64"
