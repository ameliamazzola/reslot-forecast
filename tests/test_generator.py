"""Live-mode order source: GeneratorSource around the team's OrderGenerator
(src/reslot/orders.py), the planning-window profile, empirical mode, SKU -> face
layouts, the control log, and the generated-vs-real comparison. Synthetic data only.
"""
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pytest
import yaml

import test_racksmart as TR
import test_sim as TS
from reslot.io.wms_picks import load_wms_picks
from reslot.orders import OrderGenerator
from reslot.sim import engine as E
from reslot.sim import generator as G
from reslot.sim.orders import GeneratorSource
from reslot.sim.policy import make_policy
from reslot.travel.artifacts import TravelNetwork

FMT = "%m/%d/%Y %I:%M:%S %p"
SCOPE = TR.SCOPE
PLAN_START, PLAN_END = pd.Timestamp("2026-09-12"), pd.Timestamp("2026-10-12")
RUN_START = PLAN_END
TG_START = RUN_START
SKUS = [f"2000{k:04d}" for k in range(20)]                 # SKU k sits at face FF0kkA
GROUPS = [SKUS[:10], SKUS[10:]]                            # orders draw from one group: two clusters
CLUSTER_CFG = {"min_support": 0.005, "min_jaccard": 0.10, "resolution": 1.0, "seed": 42}


@pytest.fixture
def net():
    return TravelNetwork(*TR.network())


def pick_rows(start, days, n_orders, seed=1, first_order=500000):
    """Pick-report rows: each order takes 1-4 SKUs from one group, 1-5 cases each,
    released at a random minute; one operator per order."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_orders):
        t0 = pd.Timestamp(start) + pd.Timedelta(minutes=int(rng.integers(days * 1440)))
        grp = GROUPS[i % 2]
        k = int(rng.choice([1, 2, 4], p=[0.5, 0.3, 0.2]))
        for j, s in enumerate(rng.choice(grp, size=k, replace=False)):
            rows.append({"TYPE": "009", "CREATE_DT": (t0 + pd.Timedelta(seconds=30 * j)).strftime(FMT),
                         "USER_NAME": f"U{i % 6}", "FROM_ITEM": s, "FROM_LOCATION": f"FF{SKUS.index(s):03d}A",
                         "TO_LOCATION": "V-1", "FROM_QTY": int(rng.choice([1, 2, 5])), "FROM_UOM": "CS",
                         "ORD_NO": str(first_order + i), "CUST_NAME": "X", "FROM_ZONE": "WZPCK", "TO_ZONE": "WZEQP"})
    df = pd.DataFrame(rows)
    df.insert(0, "NO", range(1, len(df) + 1))
    return df


def events(tmp_path, name="plan.csv", **kw):
    p = tmp_path / name
    pick_rows(**kw).to_csv(p, index=False)
    return load_wms_picks(p, FMT)


@pytest.fixture
def profile(tmp_path):
    ev = events(tmp_path, start=PLAN_START, days=30, n_orders=900)
    prof = G.fit_profile(ev, SCOPE, PLAN_START, PLAN_END, CLUSTER_CFG)
    from reslot.sim.racksmart import face_items
    prof["face_items"] = {str(k): str(v) for k, v in face_items(ev, SCOPE, PLAN_END).items()}
    return prof


def gcfg(**over):
    c = {"mode": "default", "seed": 7, "orders_per_day": None, "affinity": 0.5, "sku_to_face": "synthetic",
         "layout_seed": 3, "default": {"shift_start_hour": 6, "shift_minutes": 960, "lines_per_order_mean": 3.0,
                                       "min_lines": 1, "max_lines": 6, "cases_per_line_mean": 2.0}}
    c.update(over)
    return c


def source(net, profile, days=2, **over):
    return G.make_generator_source(gcfg(**over), SCOPE, net, profile, RUN_START, days)


def drawn(src):
    return [(t.order_id, t.arrival_s, tuple((p.location, p.qty) for p in t.picks)) for t in src.orders()]


# --------------------------------------------------------------------------
# the profile: planning window only
# --------------------------------------------------------------------------

def test_profile_is_fitted_on_the_planning_window_only(tmp_path, profile):
    assert profile["window"] == [str(PLAN_START), str(PLAN_END)] and profile["orders"] == 900
    assert set(profile["lines_per_order_hist"]) == {1, 2, 4}
    assert set(profile["cases_per_line_hist"]) <= {1, 2, 4, 5, 6, 7, 10}
    assert sum(profile["hour_of_day_share"]) == pytest.approx(1) and len(profile["day_of_week_share"]) == 7
    assert profile["orders_per_day"] == pytest.approx(30.0) and set(profile["sku_rates"]) == set(SKUS)
    # SKUs never ordered together never share a cluster; pairs below the support /
    # Jaccard cut leave a SKU unclustered
    cl = profile["clusters"]
    a, b = ({cl[s] for s in g if s in cl} for g in GROUPS)
    assert a and b and not (a & b) and profile["skus_with_cluster"] == len(cl) > 4
    # orders after the planning end change nothing
    plan = pick_rows(start=PLAN_START, days=30, n_orders=900)
    later = pick_rows(start=RUN_START, days=3, n_orders=50, seed=9, first_order=900000)
    later["NO"] += 10**6
    pd.concat([plan, later]).to_csv(tmp_path / "both.csv", index=False)
    both = G.fit_profile(load_wms_picks(tmp_path / "both.csv", FMT), SCOPE, PLAN_START, PLAN_END, CLUSTER_CFG)
    assert {k: v for k, v in both.items() if k != "face_items"} == \
           {k: v for k, v in profile.items() if k != "face_items"}


# --------------------------------------------------------------------------
# GeneratorSource
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["default", "empirical"])
def test_same_seed_same_orders(net, profile, mode):
    a, b = drawn(source(net, profile, mode=mode)), drawn(source(net, profile, mode=mode))
    assert a == b and len(a) > 20
    assert drawn(source(net, profile, mode=mode, seed=8)) != a


def test_default_mode_is_the_generator_as_written(net, profile):
    """Same orders as calling OrderGenerator directly over each day's shift."""
    src = source(net, profile, days=2)
    list(src.orders())
    d = gcfg()["default"]
    gen = OrderGenerator(G.demand_profile(profile, list(src.sku_face)), profile["orders_per_day"],
                         shift_minutes=d["shift_minutes"], lines_per_order_mean=d["lines_per_order_mean"],
                         min_lines=d["min_lines"], max_lines=d["max_lines"],
                         cases_per_line_mean=d["cases_per_line_mean"], affinity=0.5, seed=7)
    direct = [o for day in range(2) for m in range(360, 360 + 960) for o in gen.arrivals(day * 1440 + m)]
    assert [(o.order_id, o.minute, o.lines) for o in direct] == \
           [(o.order_id, o.minute, o.lines) for o in src.generated]


@pytest.mark.parametrize("how", ["synthetic", "real"])
@pytest.mark.parametrize("mode", ["default", "empirical"])
def test_every_line_maps_to_an_ok_stop_and_releases_never_go_back(net, profile, how, mode):
    src = source(net, profile, mode=mode, sku_to_face=how)
    tours = list(src.orders())
    usable = net.usable_faces(SCOPE["zone"], SCOPE["slot_type"])
    ok = set(net.crosswalk.index[net.crosswalk["status"] == "ok"])
    locs = {p.location for t in tours for p in t.picks}
    assert locs and locs <= set(usable.index) and locs <= ok
    assert all(net.feet("D", usable.at[l, "stop_id"]) >= 0 for l in locs)        # stop in the distance table
    arr = [t.arrival_s for t in tours]
    assert arr == sorted(arr) and all(t.user_id is None for t in tours)
    assert all(len(t.picks) == len({p.location for p in t.picks}) for t in tours)  # one line per SKU
    if how == "real":                               # the SKU's face is where it was last picked
        assert src.sku_face == {s: f"FF{SKUS.index(s):03d}A" for s in SKUS}


def test_empirical_mode_stays_inside_the_fitted_support(net, profile):
    profile = {**profile, "lines_per_order_hist": {1: 5, 2: 3, 4: 2}, "cases_per_line_hist": {1: 4, 3: 1}}
    src = source(net, profile, days=7, mode="empirical", orders_per_day=200)
    tours = list(src.orders())
    assert len(tours) > 1000
    assert {len(t.picks) for t in tours} == {1, 2, 4}
    assert {p.qty for t in tours for p in t.picks} == {1.0, 3.0}


def test_empirical_arrivals_follow_the_hour_and_day_shares(net, profile):
    hours = [0.0] * 24
    hours[10] = 1.0                                                     # every order at 10:00-11:00
    days = [0.0] * 7
    days[0] = 1.0                                                       # Mondays only
    src = source(net, {**profile, "hour_of_day_share": hours, "day_of_week_share": days},
                 days=7, mode="empirical", orders_per_day=100, hour_profile="single")
    ts = [RUN_START + pd.Timedelta(seconds=t.arrival_s) for t in src.orders()]
    assert {t.hour for t in ts} == {10} and {t.dayofweek for t in ts} == {0}
    assert 600 < len(ts) < 800                                          # ~ 100 x 7 in that one hour


def test_split_hour_profile_uses_weekday_hours_on_weekdays_and_weekend_hours_on_weekends(net, profile):
    wd, we = [0.0] * 24, [0.0] * 24
    wd[9], we[20] = 1.0, 1.0
    prof = {**profile, "hour_of_day_share_weekday": wd, "hour_of_day_share_weekend": we}
    src = source(net, prof, days=14, mode="empirical", orders_per_day=50)
    ts = [RUN_START + pd.Timedelta(seconds=t.arrival_s) for t in src.orders()]
    assert {t.hour for t in ts if t.dayofweek < 5} == {9} and {t.hour for t in ts if t.dayofweek >= 5} == {20}
    assert src.layout["hour_profile"] == "split" and "weekday / weekend" in src.label
    # the first version stays available: one profile for every day
    single = source(net, prof, days=14, mode="empirical", orders_per_day=50, hour_profile="single")
    assert single.layout["hour_profile"] == "single" and drawn(single) != drawn(source(net, prof, days=14,
                                                                                         mode="empirical", orders_per_day=50))
    old = {k: v for k, v in profile.items() if not k.startswith("hour_of_day_share_")}
    with pytest.raises(ValueError, match="rerun fit_generator_profile"):
        source(net, old, mode="empirical")
    assert drawn(source(net, old, mode="empirical", hour_profile="single"))
    with pytest.raises(ValueError, match="hour_profile"):
        source(net, profile, mode="empirical", hour_profile="both")


def test_profile_fits_weekday_and_weekend_hours_from_their_own_orders(tmp_path):
    rows = pick_rows(start=PLAN_START, days=30, n_orders=600)
    ts = pd.to_datetime(rows["CREATE_DT"], format=FMT)
    # weekend orders all at 21:00, weekday orders keep their random times
    first = rows.groupby("ORD_NO")["CREATE_DT"].transform("first")
    wkend = pd.to_datetime(first, format=FMT).dt.dayofweek >= 5
    rows.loc[wkend, "CREATE_DT"] = ts[wkend].dt.normalize().add(pd.Timedelta(hours=21)).dt.strftime(FMT)
    rows.to_csv(tmp_path / "p.csv", index=False)
    prof = G.fit_profile(load_wms_picks(tmp_path / "p.csv", FMT), SCOPE, PLAN_START, PLAN_END, CLUSTER_CFG)
    assert prof["hour_of_day_share_weekend"][21] == pytest.approx(1.0)
    assert prof["hour_of_day_share_weekday"][21] < 0.2 and prof["orders_weekday"] + prof["orders_weekend"] == 600
    assert sum(prof["hour_of_day_share_weekday"]) == pytest.approx(1)


def test_noise_floor_is_the_generator_against_itself(net, profile):
    g = gcfg(mode="empirical")
    end = RUN_START + pd.Timedelta(days=3)
    nf = G.noise_floor(g, SCOPE, net, profile, RUN_START, end, pairs=2, seed_base=100)
    assert nf["seed_pairs"] == [[101, 102], [103, 104]]
    assert set(nf["scores"]) == {"orders_per_hour", "orders_per_hour_weekday", "orders_per_hour_weekend",
                                 "lines_per_order", "cases_per_line", "share_1_2_line_orders", "sku_concentration"}
    for v in nf["scores"].values():
        assert len(v["values"]) == 2 and 0 <= v["mean"] <= v["max"] <= 1
    assert nf["scores"]["lines_per_order"]["max"] > 0                      # different seeds, different draws
    assert G.noise_floor(g, SCOPE, net, profile, RUN_START, end, pairs=2, seed_base=100) == nf


def _library(profile):
    return {tuple((sku, int(c)) for sku, c in o) for o in profile["order_library"]}


def test_resample_draws_whole_planning_orders_seeded(net, profile):
    assert len(profile["order_library"]) == profile["orders"] == 900
    src = source(net, profile, days=3, mode="resample", orders_per_day=200)
    tours = list(src.orders())
    assert len(tours) > 300 and src.describe()["mode_label"].startswith("resample")
    lib = _library(profile)
    for o in src.generated:                                         # each one a real order, exactly
        assert tuple(o.lines) in lib
    assert {len(t.picks) for t in tours} <= {len(o) for o in lib}
    assert src.layout["order_library"] == {"orders_in_library": 900, "orders_kept": 900, "lines_left_out_no_face": 0}
    fresh = drawn(source(net, profile, days=3, mode="resample", orders_per_day=200))
    assert fresh == drawn(source(net, profile, days=3, mode="resample", orders_per_day=200))   # same seed
    assert drawn(source(net, profile, days=3, mode="resample", orders_per_day=200, seed=8)) != fresh
    # same arrival profile as empirical: Mondays only, 10:00-11:00 only
    hours, days = [0.0] * 24, [0.0] * 7
    hours[10], days[0] = 1.0, 1.0
    prof = {**profile, "hour_of_day_share_weekday": hours, "hour_of_day_share_weekend": hours,
            "day_of_week_share": days}
    ts = [TG_START + pd.Timedelta(seconds=t.arrival_s)
          for t in source(net, prof, days=7, mode="resample", orders_per_day=100).orders()]
    assert ts and {t.hour for t in ts} == {10} and {t.dayofweek for t in ts} == {0}


def test_resample_needs_a_profile_with_the_order_library(net, profile):
    old = {k: v for k, v in profile.items() if k != "order_library"}
    with pytest.raises(ValueError, match="order library"):
        source(net, old, mode="resample")


def test_resample_leaves_out_lines_of_skus_without_a_face(net, profile):
    """Real layout: a SKU last picked nowhere usable has no face; its lines go, the rest of the order stays."""
    prof = {**profile, "face_items": {k: v for k, v in profile["face_items"].items() if v != SKUS[0]}}
    src = source(net, prof, days=2, mode="resample", orders_per_day=200, sku_to_face="real")
    list(src.orders())
    info = src.layout["order_library"]
    assert info["lines_left_out_no_face"] > 0 and SKUS[0] not in {s for o in src.generated for s, _ in o.lines}


def test_the_engine_runs_generated_orders(net, profile):
    src = source(net, profile, days=1)
    trip = dict(TR.ENDS)
    res = E.simulate(src, 2, make_policy("static"), net, TS.Synthetic(), SCOPE, trip)
    assert len(res.tours) == src.describe()["orders_generated"] > 0


def test_every_control_change_is_logged_with_its_sim_time(net, profile):
    src = source(net, profile, days=1)
    it = src.orders()
    first = [next(it) for _ in range(5)]
    src.set_affinity(0.9, sim_time=first[-1].arrival_s)
    src.set_speed(4.0, sim_time=first[-1].arrival_s + 60)
    with pytest.raises(ValueError, match="speed"):
        src.set_speed(0, sim_time=0)
    d = src.describe()
    assert [(c["control"], c["value"], c["previous"]) for c in d["controls"]] == [("affinity", 0.9, 0.5), ("speed", 4.0, 1.0)]
    assert d["controls"][0]["sim_time_s"] == first[-1].arrival_s and d["controls"][0]["orders_generated_before"] == 5
    assert src.gen.affinity == 0.9 and d["affinity_initial"] == 0.5 and d["affinity_now"] == 0.9
    assert d["type"] == "generator" and d["citable"] is False and GeneratorSource.citable is False


def test_profile_after_the_run_start_is_refused(net, profile):
    with pytest.raises(ValueError, match="after the run start"):
        G.make_generator_source(gcfg(), SCOPE, net, profile, RUN_START - pd.Timedelta(days=1), 1)


# --------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------

def test_tvd():
    assert G.tvd({1: 5, 2: 5}, {1: 1, 2: 1}) == 0
    assert G.tvd({1: 3}, {2: 7}) == 1
    assert G.tvd({1: 1, 2: 1}, {1: 1}) == pytest.approx(0.5)


def test_compare_scores_identical_sides_zero_and_applies_thresholds_only_where_set(tmp_path):
    ev = events(tmp_path, start=RUN_START, days=7, n_orders=300)
    st = G.order_stats(G.order_lines(ev, SCOPE, RUN_START, RUN_START + pd.Timedelta(days=7)),
                       RUN_START, RUN_START + pd.Timedelta(days=7))
    s = G.compare_stats(st, st, {"thresholds": {"lines_per_order": 0.05}})
    assert all(v["tvd"] == 0 for v in s.values())
    assert s["lines_per_order"]["pass"] is True and s["cases_per_line"]["pass"] is None
    other = {**st, "lines_per_order_hist": {10: 1}, "share_1_2_line_orders": 0.0}
    s2 = G.compare_stats(other, st, {"thresholds": {"lines_per_order": 0.05}})
    assert s2["lines_per_order"]["tvd"] == 1 and s2["lines_per_order"]["pass"] is False
    assert s2["share_1_2_line_orders"]["tvd"] == pytest.approx(st["share_1_2_line_orders"])


def test_fit_and_compare_scripts_end_to_end(tmp_path, monkeypatch):
    D, xw = TR.network()
    (tmp_path / "travel").mkdir()
    D.to_csv(tmp_path / "travel" / "stop_distance_ft.csv")
    xw.to_csv(tmp_path / "travel" / "location_stop_crosswalk.csv", index=False)
    pick_rows(start=PLAN_START, days=30, n_orders=900).to_csv(tmp_path / "plan.csv", index=False)
    run = pick_rows(start=RUN_START, days=28, n_orders=800, seed=5, first_order=900000)
    run.assign(TYPE="9", TRANS_TYPE="PICK").to_csv(tmp_path / "run.csv", index=False)
    cfg = yaml.safe_load((TR.REPO / "configs" / "sim.yaml").read_text())
    cfg["data"]["picks_path"] = str(tmp_path / "run.csv")
    cfg["travel"]["dir"] = str(tmp_path / "travel")
    cfg["paths"]["out"] = str(tmp_path / "out")
    cfg["racksmart"].update(planning_picks=[str(tmp_path / "plan.csv")], source_checks=None)
    cfg["generator"]["profile_path"] = str(tmp_path / "gen" / "profile.json")
    cfg["generator"]["mode"] = "empirical"
    (tmp_path / "sim.yaml").write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(sys, "argv", ["fit_generator_profile.py", str(tmp_path / "sim.yaml")])
    runpy.run_path(str(TR.REPO / "scripts" / "fit_generator_profile.py"), run_name="__main__")
    prof = json.loads((tmp_path / "gen" / "profile.json").read_text())
    m = json.loads((tmp_path / "gen" / "manifest.json").read_text())
    assert prof["orders"] == 900 and m["profile_summary"]["orders"] == 900 and m["inputs"][0]["sha256_16"]
    assert "2026-10-12" in m["lookahead"]

    monkeypatch.setattr(sys, "argv", ["compare_generator.py", str(tmp_path / "sim.yaml")])
    runpy.run_path(str(TR.REPO / "scripts" / "compare_generator.py"), run_name="__main__")
    d = tmp_path / "out" / "generator_compare_empirical_split_20261012_20261108"
    res = json.loads((d / "comparison.json").read_text())
    th = cfg["generator"]["compare"]["thresholds"]
    assert set(res["scores"]) == set(th) == {"orders_per_hour", "orders_per_hour_weekday", "orders_per_hour_weekend",
                                             "lines_per_order", "cases_per_line", "share_1_2_line_orders",
                                             "sku_concentration"}
    assert res["real"]["orders"] == 800
    assert all(v["threshold"] == th[k] and isinstance(v["pass"], bool) for k, v in res["scores"].items())
    nf = cfg["generator"]["compare"]["noise_floor"]
    assert len(res["noise_floor"]["seed_pairs"]) == nf["seed_pairs"] == 5
    assert all(0 <= v["noise_floor_mean"] <= v["noise_floor_max"] for v in res["scores"].values())
    assert res["scores"]["lines_per_order"]["tvd"] < 0.1               # same process both sides here
    m = json.loads((d / "manifest.json").read_text())
    assert m["mode"] == "generator_compare" and m["generator_mode"] == "empirical" and m["hour_profile"] == "split"
    assert m["generator_mode_label"].startswith("empirical")

    monkeypatch.setattr(sys, "argv", ["compare_generator.py", str(tmp_path / "sim.yaml"), "--mode", "resample"])
    runpy.run_path(str(TR.REPO / "scripts" / "compare_generator.py"), run_name="__main__")
    d = tmp_path / "out" / "generator_compare_resample_split_20261012_20261108"
    res = json.loads((d / "comparison.json").read_text())
    assert all(v["threshold"] == th[k] and isinstance(v["pass"], bool) for k, v in res["scores"].items())
    assert res["source"]["mode"] == "resample" and len(res["noise_floor"]["seed_pairs"]) == 5


# --------------------------------------------------------------------------
# VMS / vitamin-brand exclusion, mode labels, the repo's config
# --------------------------------------------------------------------------

EXCLUDE = {"vms_zone": "WZVMS", "vms_keep_pattern": "ARRID", "vitamin_pattern": "VITFUS|CRITTER"}


def test_vms_and_vitamin_brand_skus_are_left_out_of_the_profile_and_clusters(tmp_path):
    """The affinity pipeline's filter: a SKU picked in WZVMS is out unless ARRID; a
    discontinued vitamin brand is out wherever it sits; a missing description is no match."""
    rows = pick_rows(start=PLAN_START, days=30, n_orders=400)
    rows["FROM_DESCR"] = "SOAP"
    ts = rows["CREATE_DT"].iloc[0]
    extra = pd.DataFrame([
        # VMS SKU, also case-picked in WZPCK on an order of its own and on order 500000
        {"FROM_ITEM": "90000001", "FROM_ZONE": "WZVMS", "FROM_LOCATION": "FF001A", "ORD_NO": "700001", "FROM_DESCR": "VITAFUSION GUMMY"},
        {"FROM_ITEM": "90000001", "FROM_ZONE": "WZPCK", "FROM_LOCATION": "FF001A", "ORD_NO": "700002", "FROM_DESCR": "VITAFUSION GUMMY"},
        {"FROM_ITEM": "90000001", "FROM_ZONE": "WZPCK", "FROM_LOCATION": "FF001A", "ORD_NO": "500000", "FROM_DESCR": "VITAFUSION GUMMY"},
        # active brand stored in WZVMS: kept
        {"FROM_ITEM": "90000002", "FROM_ZONE": "WZVMS", "FROM_LOCATION": "FF002A", "ORD_NO": "700003", "FROM_DESCR": "ARRID XX 2.6OZ"},
        {"FROM_ITEM": "90000002", "FROM_ZONE": "WZPCK", "FROM_LOCATION": "FF002A", "ORD_NO": "700004", "FROM_DESCR": "ARRID XX 2.6OZ"},
        # discontinued vitamin brand, never in WZVMS: still out
        {"FROM_ITEM": "90000003", "FROM_ZONE": "WZPCK", "FROM_LOCATION": "FF003A", "ORD_NO": "700005", "FROM_DESCR": "LIL CRITTERS"},
        # WZVMS with no description: no keep match -> out (as the pipeline's na=False)
        {"FROM_ITEM": "90000004", "FROM_ZONE": "WZVMS", "FROM_LOCATION": "FF004A", "ORD_NO": "700006", "FROM_DESCR": None},
        {"FROM_ITEM": "90000004", "FROM_ZONE": "WZPCK", "FROM_LOCATION": "FF004A", "ORD_NO": "700007", "FROM_DESCR": None},
    ]).assign(TYPE="009", CREATE_DT=ts, USER_NAME="U9", TO_LOCATION="V-1", FROM_QTY=1, FROM_UOM="CS",
              CUST_NAME="X", TO_ZONE="WZEQP")
    extra["NO"] = range(10**6, 10**6 + len(extra))
    pd.concat([rows, extra]).to_csv(tmp_path / "p.csv", index=False)
    ev = load_wms_picks(tmp_path / "p.csv", FMT)
    assert ev["item_descr"].notna().any()
    prof = G.fit_profile(ev, SCOPE, PLAN_START, PLAN_END, CLUSTER_CFG, exclude=EXCLUDE)
    assert prof["excluded_skus"] == ["90000001", "90000003", "90000004"]
    assert "90000002" in prof["sku_rates"] and not set(prof["excluded_skus"]) & set(prof["sku_rates"])
    assert not set(prof["excluded_skus"]) & set(prof["clusters"])
    assert not set(prof["excluded_skus"]) & {sku for o in prof["order_library"] for sku, _ in o}
    x = prof["exclusion"]
    assert (x["vms_skus"], x["vitamin_brand_skus"], x["vms_skus_without_description"]) == (2, 1, 1)
    # orders 700002, 700005, 700007 had only excluded lines; order 500000 keeps its other lines
    assert x["orders_dropped"] == 3 and x["lines_dropped"] == 4 and prof["orders"] == 400 + 1
    none = G.fit_profile(ev, SCOPE, PLAN_START, PLAN_END, CLUSTER_CFG)
    assert none["exclusion"]["rule"] == "none" and none["excluded_skus"] == [] and "90000001" in none["sku_rates"]


def test_modes_are_labelled(net, profile):
    assert source(net, profile).describe()["mode_label"] == "team generator as written (06:00 + 960 min shift)"
    assert source(net, profile, mode="empirical").describe()["mode_label"].startswith("empirical")


def test_repo_config_runs_live_empirical_with_thresholds_set():
    g = yaml.safe_load((TR.REPO / "configs" / "sim.yaml").read_text())["generator"]
    assert g["mode"] == "empirical" and g["exclude"] == EXCLUDE and g["hour_profile"] == "split"
    assert g["compare"]["noise_floor"] == {"seed_pairs": 5, "seed_base": 1000}
    assert g["compare"]["thresholds"] == {"orders_per_hour": 0.15, "orders_per_hour_weekday": 0.15,
                                          "orders_per_hour_weekend": 0.15, "lines_per_order": 0.10,
                                          "cases_per_line": 0.10, "share_1_2_line_orders": 0.05,
                                          "sku_concentration": 0.10}
