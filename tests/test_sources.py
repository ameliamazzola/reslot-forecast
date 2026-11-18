"""Planning picks across exports (src/reslot/sim/sources.py): two formats, one
deduplicated union, and the overlap check that refuses to plan when exports
disagree. Plus the operators-per-hour reference count. Synthetic data only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import pytest
import yaml

import test_no_lookahead as NL
import test_racksmart as TR
from reslot.io.wms_picks import load_wms_picks
from reslot.sim import racksmart as RS
from reslot.sim import sources as S
from reslot.sim.operators import operators_per_hour

FMT = "%m/%d/%Y %I:%M:%S %p"
SCOPE = TR.SCOPE
TOL = yaml.safe_load((TR.REPO / "configs" / "sim.yaml").read_text())["racksmart"]["source_checks"]["tolerance"]
DAY = "2026-08-24"


def report():
    """Pick-report format (TYPE 009, no TRANS_TYPE)."""
    return TR._pick_rows(TR.PLAN, DAY)


def as_transactions(df):
    """The same rows as a transactions export (TYPE 9, TRANS_TYPE PICK)."""
    return df.assign(TYPE="9", TRANS_TYPE="PICK")


def overlap(tmp_path, a_df, b_df, start=f"{DAY} 00:00", end=f"{DAY} 18:00"):
    """Synthetic picks run 00:00 - about 18:20 on DAY: the default overlap sits inside that."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a_df.to_csv(a, index=False)
    b_df.to_csv(b, index=False)
    return {"tolerance": dict(TOL), "overlaps": [{"name": "t", "a": [str(a)], "b": [str(b)],
                                                  "start": start, "end": end}]}


def _shift(df, rows, seconds):
    ts = pd.to_datetime(df["CREATE_DT"], format=FMT)
    ts.iloc[rows] = ts.iloc[rows] + pd.Timedelta(seconds=seconds)
    return df.assign(CREATE_DT=ts.dt.strftime(FMT))


# --------------------------------------------------------------------------
# reading and deduplicating across formats
# --------------------------------------------------------------------------

def test_both_formats_read_as_the_same_picks_and_dedup_to_one_copy(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    report().to_csv(a, index=False)
    as_transactions(report()).to_csv(b, index=False)
    ev, info = S.read_picks([str(a), {"path": str(b)}], FMT, (pd.Timestamp(DAY), pd.Timestamp("2026-08-26")))
    assert info[str(a)]["format"] == "pick report" and info[str(b)]["format"] == "transactions export"
    assert (ev["event_class"] == "pick").all() and len(ev) == 2 * len(report())
    assert len(S.dedup(ev)) == len(report())


def test_an_entry_can_carry_its_own_timestamp_format(tmp_path):
    p = tmp_path / "iso.csv"
    df = report()
    df["CREATE_DT"] = pd.to_datetime(df["CREATE_DT"], format=FMT).dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(p, index=False)
    ev, info = S.read_picks([{"path": str(p), "ts_format": "%Y-%m-%d %H:%M:%S"}], FMT,
                            (pd.Timestamp(DAY), pd.Timestamp("2026-08-26")))
    assert len(ev) == len(df) and info[str(p)]["ts_format"] == "%Y-%m-%d %H:%M:%S"
    with pytest.raises(ValueError, match="did not parse"):
        S.read_picks([str(p)], FMT, (pd.Timestamp(DAY), pd.Timestamp("2026-08-26")))


def test_dedup_keeps_different_transactions_with_the_same_number_apart():
    df = pd.DataFrame({"event_id": [1, 1, 2], "event_class": ["pick", "pick", "pick"],
                       "order_id": ["o", "o", "o"], "user_id": ["u", "u", "u"],
                       "from_loc": ["FF000A", "FF000A", "FF000A"], "item_id": ["1", "1", "1"],
                       "ts": pd.to_datetime(["2026-09-01 08:00"] * 3)})
    assert S.dedup(df)["event_id"].tolist() == [1, 2]
    df.loc[1, "from_loc"] = "FF001A"                          # same NO, another pick: not a copy
    assert len(S.dedup(df)) == 3


# --------------------------------------------------------------------------
# the overlap check
# --------------------------------------------------------------------------

def test_identical_exports_in_two_formats_pass(tmp_path):
    r = S.source_check(overlap(tmp_path, report(), as_transactions(report())), FMT, SCOPE)
    ov = r["overlaps"][0]
    assert ov["ok"] and ov["a_vs_b"]["legs_a"] == ov["a_vs_b"]["legs_b"] == ov["a_vs_b"]["matched"] > 400
    assert ov["a_vs_b"]["abs_seconds_max"] == 0 and ov["union_vs_a"]["legs_b"] == ov["a_vs_b"]["legs_a"]


def test_a_missing_pick_is_refused(tmp_path):
    b = as_transactions(report()).drop(index=[30])
    with pytest.raises(ValueError, match="found on the other side"):
        S.source_check(overlap(tmp_path, report(), b), FMT, SCOPE)


def test_times_must_agree_within_a_second(tmp_path):
    assert S.source_check(overlap(tmp_path, report(), _shift(report(), [30], 1)), FMT, SCOPE)["overlaps"][0]["ok"]
    with pytest.raises(ValueError, match=r"agree within 1 s"):
        S.source_check(overlap(tmp_path, report(), _shift(report(), [30], 5)), FMT, SCOPE)


def test_different_transaction_numbers_make_dedup_unsafe_and_are_refused(tmp_path):
    b = as_transactions(report())
    b["NO"] = b["NO"] + 500_000
    with pytest.raises(ValueError, match="deduplication by NO is not safe") as e:
        S.source_check(overlap(tmp_path, report(), b), FMT, SCOPE)
    assert "deduplicated union vs a" in str(e.value)          # the union would double every leg


def test_an_export_starting_late_is_compared_where_both_hold_rows(tmp_path):
    late = report()
    ts = pd.to_datetime(late["CREATE_DT"], format=FMT)
    b = as_transactions(late[ts >= pd.Timestamp(f"{DAY} 08:30")])
    r = S.source_check(overlap(tmp_path, late, b, start=f"{DAY} 08:00", end=f"{DAY} 18:00"), FMT, SCOPE)
    ov = r["overlaps"][0]
    assert ov["ok"] and ov["compared"][0] == f"{DAY} 08:30:00" and 0.9 < ov["cover"] < 1
    with pytest.raises(ValueError, match="overlap on only"):
        S.source_check(overlap(tmp_path, late, b, start=f"{DAY} 07:00", end=f"{DAY} 10:00"), FMT, SCOPE)


def test_planning_refuses_when_the_exports_disagree(tmp_path):
    cfg, n = NL._planning_cfg(tmp_path)
    good = overlap(tmp_path, report(), as_transactions(report()))
    cfg["racksmart"]["source_checks"] = good
    pol, info, files = RS.load_racksmart(cfg, n)
    assert info["source_checks"]["overlaps"][0]["ok"] and str(tmp_path / "b.csv") in files
    cfg["racksmart"]["source_checks"] = overlap(tmp_path, report(), as_transactions(report()).drop(index=[30]))
    with pytest.raises(ValueError, match="not planning"):
        RS.load_racksmart(cfg, n)


def test_planning_reads_both_formats_and_counts_each_pick_once(tmp_path):
    cfg, n = NL._planning_cfg(tmp_path)
    plan = report()
    half = len(plan) // 2
    plan.iloc[: half + 10].to_csv(tmp_path / "r.csv", index=False)               # pick report
    as_transactions(plan.iloc[half:]).to_csv(tmp_path / "t.csv", index=False)    # transactions, 10 rows overlap
    cfg["racksmart"]["planning_picks"] = [str(tmp_path / "r.csv"), {"path": str(tmp_path / "t.csv")}]
    _, _, train, base, _ = RS.prepare_planning(cfg, n)
    rows = base["planning_rows"]
    assert rows["read"] == len(plan) + 10 and rows["after_dedup_by_event_id"] == len(plan)
    assert {i["format"] for i in rows["files"].values()} == {"pick report", "transactions export"}
    assert len(train) == 220


# --------------------------------------------------------------------------
# operators per hour, and the repo's config
# --------------------------------------------------------------------------

def test_operators_per_hour_counts_distinct_case_pickers_per_clock_hour(tmp_path):
    p = tmp_path / "w.csv"
    report().to_csv(p, index=False)
    ev = load_wms_picks(p, FMT)
    r = operators_per_hour(ev, SCOPE, pd.Timestamp(DAY), pd.Timestamp("2026-08-26"))
    hourly = ev.groupby(ev["ts"].dt.floor("h"))["user_id"].nunique()
    assert r["median"] == hourly.median() and r["max"] == hourly.max() == 5        # U0..U4
    assert r["case_picks"] == len(ev)
    with pytest.raises(ValueError, match="no in-scope case picks"):
        operators_per_hour(ev, SCOPE, pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02"))


def test_repo_config_runs_four_weeks_after_a_30_day_plan():
    cfg = yaml.safe_load((TR.REPO / "configs" / "sim.yaml").read_text())
    assert cfg["window"] == {"start": "2026-10-12", "end": "2026-11-08"}
    assert pd.Timestamp("2026-10-12").day_name() == "Monday" and pd.Timestamp("2026-11-08").day_name() == "Sunday"
    rc = cfg["racksmart"]
    assert rc["planning_window"] == {"start": "2026-09-12", "end": "2026-10-11"}
    assert len(S.entry_paths(rc["planning_picks"])) == 4 and cfg["data"]["picks_path"] in rc["planning_picks"]
    assert [o["name"] for o in rc["source_checks"]["overlaps"]] == ["365day_vs_sep_oct", "sep_oct_vs_oct_nov"]
    assert "move_cases" in cfg["lookahead_exceptions"] and "8 days after" in cfg["lookahead_exceptions"]["move_cases"]
