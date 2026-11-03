"""Tests for the network_feet leg-time model (src/reslot/sim/legtime.py).
Synthetic refit outputs only: a cost table and manifest written the way
scripts/refit_travel_feet.py writes them."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pytest

import test_sim as TS
from reslot.sim import engine as E
from reslot.sim.engine import Leg
from reslot.sim.legtime import LegModelError, NetworkFeetLegModel
from reslot.utils.manifest import file_hash

COEF = {"fixed_s": 20.0, "per_case_s": 5.0, "per_path_unit_s": 0.0, "cross_aisle_s": 30.0}


def write_fit(tmp, travel_dir, outcome="adopted", gap=30.0, dist_hash=None):
    """travel_cost_ft.csv: same = 20 + 0.5*feet up to 40 ft, then flat; cross = same + gap."""
    d = np.arange(0, 60, 10, dtype=float)
    same = 20 + 0.5 * np.minimum(d, 40)
    pd.DataFrame({"delta_ft": d, "seconds_same_aisle": same, "seconds_cross_aisle": same + gap,
                  "source": "scan_interval_proxy"}).to_csv(tmp / "travel_cost_ft.csv", index=False)
    h = dist_hash or file_hash(travel_dir / "stop_distance_ft.csv")
    (tmp / "manifest.json").write_text(json.dumps({
        "distance_unit": "ft", "coefficients": COEF, "git_sha": "abc", "config_hash": "def",
        "inputs": [{"path": "picks.csv", "sha256_16": "x"}, {"path": "data/stop_distance_ft.csv", "sha256_16": h}],
        "review": {"outcome": outcome, "n_legs": 100, "n_trips": 20,
                   "mae_difference_feet_minus_work_path": -0.5, "bootstrap": {"interval": [-1, 0.1]}}}))
    return {"travel": {"dir": str(travel_dir)},
            "leg_time": {"model": "network_feet", "calibration_manifest": str(tmp / "manifest.json"),
                         "cost_table": str(tmp / "travel_cost_ft.csv")}}


def leg(feet, cross=False, qty=1.0):
    return Leg("A", "B", "S1", "S2", feet, cross, qty)


def test_predicts_the_fitted_curve_plus_handling(tmp_path, travel_dir):
    m = NetworkFeetLegModel(write_fit(tmp_path, travel_dir))
    assert m.predict(leg(0)) == pytest.approx(20 + 5)              # fixed + 1 case
    assert m.predict(leg(15)) == pytest.approx(20 + 7.5 + 5)       # between grid points: linear
    assert m.predict(leg(15, cross=True, qty=3)) == pytest.approx(20 + 7.5 + 30 + 15)
    assert m.predict(leg(500)) == m.predict(leg(40))               # beyond the grid: flat
    assert m.provenance["review"]["outcome"] == "adopted" and len(m.files) == 2


@pytest.mark.parametrize("outcome", ["rejected", "not_reviewed", None])
def test_refuses_a_fit_the_review_did_not_adopt(tmp_path, travel_dir, outcome):
    with pytest.raises(LegModelError, match="review outcome"):
        NetworkFeetLegModel(write_fit(tmp_path, travel_dir, outcome=outcome))


def test_refuses_a_fit_on_another_network(tmp_path, travel_dir):
    with pytest.raises(LegModelError, match="different stop_distance_ft.csv"):
        NetworkFeetLegModel(write_fit(tmp_path, travel_dir, dist_hash="0000000000000000"))


def test_refuses_a_table_from_another_fit(tmp_path, travel_dir):
    with pytest.raises(LegModelError, match="different fits"):
        NetworkFeetLegModel(write_fit(tmp_path, travel_dir, gap=31.0))


def test_refuses_when_the_refit_has_not_been_run(tmp_path, travel_dir):
    cfg = write_fit(tmp_path, travel_dir)
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(LegModelError, match="refit_travel_feet.py"):
        NetworkFeetLegModel(cfg)


def test_engine_builds_it_from_config_and_records_where_it_came_from(tmp_path, travel_dir, net):
    cfg = {**TS.cfg(), **write_fit(tmp_path, travel_dir)}
    model = E.make_leg_model(cfg)
    ev = TS.events([("O1", "U1", "2026-10-05 08:00:00", "FF001A", 1), ("O1", "U1", "2026-10-05 08:01:00", "GG001A", 2)])
    out = E.run(cfg, ev, net, model)
    # FF001A (S1) -> GG001A (S3): 30 ft, aisle change, 2 cases -> 20 + 15 + 30 + 10
    assert out.result.legs["sim_s"].tolist() == pytest.approx([75.0])
    assert out.extra["leg_time"]["review"]["outcome"] == "adopted" and out.extra["leg_time_model"] == "network_feet"


# fixtures shared with test_sim
travel_dir = TS.travel_dir
net = TS.net
