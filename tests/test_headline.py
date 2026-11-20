"""The headline metric (src/reslot/sim/headline.py): net labor savings vs static
and payback days."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from reslot.sim import headline as H


def test_net_labor_savings_and_payback():
    h = H.headline(static_hours=100.0, policy_hours=90.0, move_cost=91.0, rate=45.5, days=28)
    assert h["operator_hours_saved"] == 10.0 and h["labor_saved_usd"] == pytest.approx(455.0)
    assert h["net_usd"] == pytest.approx(455.0 - 91.0)
    assert h["payback_days"] == pytest.approx(91.0 / (455.0 / 28))           # 5.6 days
    assert h["pays_back_in_window"]


def test_payback_never_when_no_hours_saved_and_zero_with_no_moves():
    worse = H.headline(100.0, 101.0, 50.0, 45.5, 28)
    assert worse["payback_days"] is None and not worse["pays_back_in_window"] and worse["net_usd"] < -50
    assert H.headline(100.0, 100.0, 50.0, 45.5, 28)["payback_days"] is None
    free = H.headline(100.0, 99.0, 0.0, 45.5, 28)
    assert free["payback_days"] == 0.0 and free["pays_back_in_window"]
    slow = H.headline(100.0, 99.9, 1000.0, 45.5, 28)                         # pays back, after the window
    assert slow["payback_days"] > 28 and not slow["pays_back_in_window"]


def test_rate_comes_from_config_with_its_source():
    assert H.labor_rate({"headline": {"labor_rate_usd_per_hour": 45.5, "labor_rate_source": "x"}}) == (45.5, "x")
    with pytest.raises(ValueError, match="labor_rate_source"):
        H.labor_rate({"headline": {"labor_rate_usd_per_hour": 45.5}})
    with pytest.raises(ValueError, match="> 0"):
        H.labor_rate({"headline": {"labor_rate_usd_per_hour": 0, "labor_rate_source": "x"}})


def test_window_days_are_inclusive():
    assert H.window_days({"start": "2026-10-12", "end": "2026-11-08"}) == 28


def test_a_static_run_is_the_reference_only_when_it_ran_on_the_same_things():
    run = {"window": {"start": "2026-10-12", "end": "2026-11-08", "days": 28}, "operators": 6,
           "leg_time_model": "network_feet", "leg_time": {"a": (1, 2)}, "scope": {"uom": "CS"}, "trip_ends": {}}
    st = {**run, "policy": "static", "leg_time": {"a": [1, 2]},          # as JSON stores it
          "inputs": [{"path": "x/picks.csv", "sha256_16": "aa"}, {"path": "t/stop_distance_ft.csv", "sha256_16": "bb"}]}
    files = {"picks.csv": "aa", "stop_distance_ft.csv": "bb"}
    assert H.static_mismatch(st, run, files) == []
    assert H.static_mismatch({**st, "operators": 7}, run, files) == ["operators"]
    assert H.static_mismatch(st, run, {**files, "picks.csv": "cc"}) == ["input picks.csv"]
    assert "not a static run" in H.static_mismatch({**st, "policy": "racksmart"}, run, files)
