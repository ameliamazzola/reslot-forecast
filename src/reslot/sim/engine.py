"""
SimPy loop: case-pick tours arrive, wait for a free operator, then pick.

What it simulates (docs/decisions.md, 2026-11-02 entries):

  * Operators are a pool of fixed size (config `operators`).
  * A tour arrives at its first kept pick scan and holds one operator from its
    first pick scan to its last. Between consecutive picks it spends exactly
    what the leg-time model predicts for that leg.
  * Trips start and end at Stop_Dispatch. The dispatch -> first pick and last
    pick -> dispatch legs are counted in FEET ONLY (dispatch_legs: untimed);
    the drop-off is provisional. Both are checked against config, and nothing
    else is accepted.
  * The policy says where each product sits; the travel network turns that
    location into a stop and gives feet between stops (a table lookup).

Leg time is pluggable: the engine calls LegTimeModel.predict(leg) -> seconds
and knows nothing else about it. No real model is wired in here.

Moves: a policy's relocation moves must be priced by a move_cost function;
the engine refuses to run a policy that reports moves without one, so no
policy re-slots for free. Static reports none.

No randomness: the same inputs give the same run. Queueing and operator-count
outputs are not findings while dispatch legs are untimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd
import simpy

from ..travel.artifacts import TravelNetwork

SUPPORTED_TRIP_ENDS = {"dispatch_legs": "untimed", "drop_off_mode": "provisional"}


@dataclass(frozen=True)
class Leg:
    """One pick-to-pick leg as the leg-time model sees it. Same terms as the
    2026-09-14 model: the arriving pick's qty, whether the aisle changed, and
    the distance between the two stops (feet, from the network)."""
    from_location: str
    to_location: str
    from_stop: str
    to_stop: str
    feet: float
    cross_aisle: bool
    qty: float


class LegTimeModel(Protocol):
    name: str

    def predict(self, leg: Leg) -> float:
        """Seconds from the previous pick scan to this pick's scan."""
        ...


def check_trip_ends(trip_ends: dict, net: TravelNetwork) -> None:
    for key, value in SUPPORTED_TRIP_ENDS.items():
        if trip_ends.get(key) != value:
            raise ValueError(f"trip_ends.{key} must be {value!r} in v1 "
                             f"(docs/decisions.md 2026-11-02), got {trip_ends.get(key)!r}")
    net.require_stops(trip_ends["start_stop"], trip_ends["end_stop"])


@dataclass
class SimResult:
    tours: pd.DataFrame       # one row per tour
    legs: pd.DataFrame        # one row per simulated pick-to-pick leg
    moves: list               # policy moves, as reported
    move_cost: float          # priced by move_cost (0 for static)


TOUR_COLUMNS = ["tour_id", "order_id", "user_id", "n_picks", "n_dropped", "complete",
                "first_location", "first_stop",
                "arrival_s", "start_s", "end_s", "wait_s", "busy_s", "obs_span_s",
                "pick_feet", "dispatch_feet", "operators_busy_at_start"]
LEG_COLUMNS = ["tour_id", "leg", "from_location", "to_location", "from_stop", "to_stop",
               "feet", "cross_aisle", "qty", "sim_s", "obs_s"]


class _Release(simpy.events.Event):
    """An order's release, scheduled URGENT: at a tie with a tour ending (or any
    other event) the release is handled first. That is the order the engine has
    always had -- its arrival timers were all created at t=0, before any other
    event, so they won every tie -- and keeps replay results unchanged."""

    def __init__(self, env, delay: float):
        super().__init__(env)
        self._ok, self._value = True, None
        env.schedule(self, simpy.events.URGENT, delay)


def simulate(orders, operators: int, policy, net: TravelNetwork, leg_model: LegTimeModel,
             scope: dict, trip_ends: dict,
             move_cost: Callable[[list], float] | None = None,
             env: simpy.Environment | None = None, observer: Callable[[str, dict], None] | None = None) -> SimResult:
    """Run every order the source releases; one row per tour.

    orders:   an OrderSource (anything with .orders()) or an iterable of Tours.
              Orders are pulled one at a time in release order (arrival_s, then
              tour_id); an order released earlier than the previous one is refused.
    env:      the SimPy environment -- a plain Environment for replay (default),
              a RealtimeEnvironment for live.
    observer: optional callback(event, payload) for streaming ('release', 'start',
              'leg', 'done'); it sees results only, never future orders.
    """
    from .orders import ArrivedOrders

    if int(operators) < 1:
        raise ValueError("operators must be at least 1")
    check_trip_ends(trip_ends, net)
    faces = net.usable_faces(scope["zone"], scope["slot_type"])
    stop_of, aisle_of = faces["stop_id"].to_dict(), faces["aisle"].to_dict()
    start_stop, end_stop = trip_ends["start_stop"], trip_ends["end_stop"]

    env = env or simpy.Environment()
    pool = simpy.Resource(env, capacity=int(operators))
    view = ArrivedOrders(env)
    tour_rows, leg_rows = [], []
    emit = observer or (lambda event, payload: None)
    on_arrival = getattr(policy, "on_arrival", None)

    def place(location: str) -> str:
        where = policy.location_for(location, env.now)
        if where not in stop_of:
            raise ValueError(f"policy {policy.name!r} put {location!r} at {where!r}, "
                             "which is not a usable pick face")
        return where

    def run(t):
        with pool.request() as req:
            yield req
            start, busy_at_start = env.now, pool.count
            emit("start", {"tour_id": t.tour_id, "t": start, "busy": busy_at_start})
            # Visit order: the policy's route (static: observed scan order).
            order = list(policy.route(tuple(p.location for p in t.picks), env.now))
            if sorted(order) != list(range(len(t.picks))):
                raise ValueError(f"policy {policy.name!r} route is not a permutation of tour {t.tour_id}")
            j0 = order[0]
            first = place(t.picks[j0].location)
            prev_j, prev_loc, prev_stop = j0, first, stop_of[first]
            dispatch_feet = net.feet(start_stop, prev_stop)
            pick_feet = 0.0
            for i, j in enumerate(order[1:], start=1):
                p = t.picks[j]
                loc = place(p.location)
                stop = stop_of[loc]
                leg = Leg(prev_loc, loc, prev_stop, stop, net.feet(prev_stop, stop),
                          aisle_of[prev_loc] != aisle_of[loc], p.qty)
                seconds = float(leg_model.predict(leg))
                if not seconds >= 0:
                    raise ValueError(f"leg model {leg_model.name!r} returned {seconds} s")
                # An observed leg is only comparable if it is the same leg that
                # happened: the previous visit is the previous observed pick and
                # neither product moved (always true under static).
                same_leg = (prev_j == j - 1 and loc == p.location
                            and prev_loc == t.picks[prev_j].location)
                leg_rows.append({"tour_id": t.tour_id, "leg": i, "from_location": prev_loc,
                                 "to_location": loc, "from_stop": prev_stop, "to_stop": stop,
                                 "feet": leg.feet, "cross_aisle": leg.cross_aisle, "qty": p.qty,
                                 "sim_s": seconds, "obs_s": p.obs_leg_s if same_leg else None})
                pick_feet += leg.feet
                yield env.timeout(seconds)
                emit("leg", {"tour_id": t.tour_id, "t": env.now, "to_stop": stop, "feet": leg.feet})
                prev_j, prev_loc, prev_stop = j, loc, stop
            dispatch_feet += net.feet(prev_stop, end_stop)
            tour_rows.append({"tour_id": t.tour_id, "order_id": t.order_id, "user_id": t.user_id,
                              "n_picks": len(t.picks), "n_dropped": t.n_dropped,
                              "complete": t.complete,
                              # the first visit has no leg; recorded so every pick has a stop
                              "first_location": first, "first_stop": stop_of[first],
                              "arrival_s": t.arrival_s,
                              "start_s": start, "end_s": env.now, "wait_s": start - t.arrival_s,
                              "busy_s": env.now - start, "obs_span_s": t.obs_span_s,
                              "pick_feet": pick_feet, "dispatch_feet": dispatch_feet,
                              "operators_busy_at_start": busy_at_start})
            emit("done", {"tour_id": t.tour_id, "t": env.now, "first_stop": stop_of[first],
                          "pick_feet": pick_feet})

    def releases():
        """Pull orders one at a time; release each at its time, then start its tour."""
        stream = orders.orders() if hasattr(orders, "orders") else iter(
            sorted(orders, key=lambda t: (t.arrival_s, t.tour_id)))
        last = None
        for t in stream:
            key = (t.arrival_s, t.tour_id)
            if last is not None and key < last:
                raise ValueError(f"order {t.tour_id} released at {t.arrival_s}, before the previous order")
            last = key
            if t.arrival_s > env.now:
                yield _Release(env, t.arrival_s - env.now)
            view._release(t)
            emit("release", {"tour_id": t.tour_id, "t": env.now, "n_picks": len(t.picks)})
            if on_arrival is not None:
                on_arrival(t, view)
            env.process(run(t))

    env.process(releases())
    env.run()

    moves = list(policy.moves())
    if moves and move_cost is None:
        raise ValueError(f"policy {policy.name!r} made {len(moves)} moves but no move_cost "
                         "was given: re-slotting is never free")
    cost = float(move_cost(moves)) if moves else 0.0
    tours_df = pd.DataFrame(tour_rows, columns=TOUR_COLUMNS).sort_values("tour_id").reset_index(drop=True)
    legs_df = pd.DataFrame(leg_rows, columns=LEG_COLUMNS).sort_values(["tour_id", "leg"]).reset_index(drop=True)
    return SimResult(tours_df, legs_df, moves, cost)


# --------------------------------------------------------------------------
# Out-of-time check of the leg model (2026-11-02) -- NOT a sim validation
# --------------------------------------------------------------------------

CHECK_SOURCE = "scan_interval_proxy"
CHECK_CAVEAT = ("observed seconds are intervals between pick scans; they include scanning, "
                "pallet building and operator pauses; not a direct time study")


def _spearman(a: pd.Series, b: pd.Series):
    """None where undefined (fewer than two values, or a constant side)."""
    if len(a) < 2 or a.nunique() < 2 or b.nunique() < 2:
        return None
    return float(a.corr(b, method="spearman"))


def leg_model_check(res: SimResult, check: dict, model_name: str) -> dict:
    """Simulated vs observed, on the observed side's terms.

    Primary: pick-to-pick legs with an observed time (never a leg after a
    dropped pick; dispatch legs are never in the table), from tours with at
    least min_picks in-scope picks, observed seconds inside the calibration's
    leg bounds. Secondary: first-to-last scan span of complete tours (no
    dropped picks), simulated busy time vs observed span -- both measured
    from first to last pick scan. The filters choose which legs and tours are
    compared; they never change what was simulated.
    """
    t = res.tours
    eligible = t.loc[(t["n_picks"] + t["n_dropped"]) >= check["min_picks"], "tour_id"]
    legs = res.legs[res.legs["obs_s"].notna() & res.legs["tour_id"].isin(eligible)]
    legs = legs[legs["obs_s"].between(check["leg_min_seconds"], check["leg_max_seconds"])]
    err = legs["sim_s"] - legs["obs_s"]
    primary = {"n_legs": int(len(legs)), "n_tours": int(legs["tour_id"].nunique())}
    if len(legs):
        primary.update({
            "mae_s": float(err.abs().mean()), "medae_s": float(err.abs().median()),
            "mean_error_s": float(err.mean()),      # positive: model slower than observed
            "median_obs_s": float(legs["obs_s"].median()),
            "median_sim_s": float(legs["sim_s"].median()),
            "spearman": _spearman(legs["sim_s"], legs["obs_s"])})

    s = t[t["complete"] & ((t["n_picks"] + t["n_dropped"]) >= check["min_picks"]) & (t["obs_span_s"] > 0)]
    secondary = {"n_tours": int(len(s))}
    if len(s):
        e = (s["busy_s"] - s["obs_span_s"]).abs()
        secondary.update({
            "median_abs_pct_error": float((e / s["obs_span_s"]).median()),
            "total_sim_over_obs": float(s["busy_s"].sum() / s["obs_span_s"].sum()),
            "median_obs_span_s": float(s["obs_span_s"].median()),
            "median_sim_span_s": float(s["busy_s"].median()),
            "spearman": _spearman(s["busy_s"], s["obs_span_s"])})
    return {"what": "out-of-time check of the leg model, not a sim validation",
            "leg_time_model": model_name, "filters": dict(check),
            "primary_pick_to_pick_legs": primary, "secondary_first_to_last_scan": secondary,
            "source": CHECK_SOURCE, "caveat": CHECK_CAVEAT}


# --------------------------------------------------------------------------
# One run, from loaded inputs (file I/O lives in scripts/run_sim.py)
# --------------------------------------------------------------------------

# Leg-time models the run script can name in config (leg_time.model). Each one
# refuses to load unless its fit was adopted by the 2026-11-02 review.
from .legtime import NetworkFeetLegModel  # noqa: E402

LEG_TIME_MODELS: dict[str, Callable[[dict], LegTimeModel]] = {NetworkFeetLegModel.name: NetworkFeetLegModel}

NOT_FINDINGS = ("queueing and operator-count outputs are not findings while dispatch legs "
                "are untimed (docs/decisions.md 2026-11-02)")


def make_leg_model(cfg: dict) -> LegTimeModel:
    name = (cfg.get("leg_time") or {}).get("model")
    if name is None:
        raise ValueError("leg_time.model is null: no leg-time model has been adopted yet "
                         "(docs/decisions.md 2026-11-02, refit entry)")
    if name not in LEG_TIME_MODELS:
        raise ValueError(f"unknown leg_time.model {name!r}; registered: {sorted(LEG_TIME_MODELS)}")
    return LEG_TIME_MODELS[name](cfg)


def window_bounds(cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Config dates are inclusive days -> [start, day after end)."""
    start = pd.Timestamp(cfg["window"]["start"])
    end = pd.Timestamp(cfg["window"]["end"]) + pd.Timedelta(days=1)
    if end <= start:
        raise ValueError(f"window ends before it starts: {cfg['window']}")
    return start, end


@dataclass
class RunOutput:
    result: SimResult
    dropped: pd.DataFrame
    extra: dict               # everything the manifest records beyond config/git/seed/inputs
    tours: list               # the workload's tours, as simulated


def check_planning_inputs(policy, start: pd.Timestamp) -> None:
    """No lookahead in planning: a policy that planned before the run must have
    used inputs that end at or before the run start (planning_inputs_end is the
    exclusive end of the data its plan saw)."""
    end = getattr(policy, "planning_inputs_end", None)
    if end is not None and pd.Timestamp(end) > start:
        raise ValueError(f"policy {policy.name!r} planned on inputs up to {end}, after the run start {start}")


def run(cfg: dict, events: pd.DataFrame, net: TravelNetwork, leg_model: LegTimeModel,
        policy=None, move_cost=None) -> RunOutput:
    """One replay run from loaded inputs (the citable mode)."""
    from .orders import ReplaySource
    from .policy import make_policy

    start, end = window_bounds(cfg)
    policy = policy or make_policy(cfg["policy"])
    check_planning_inputs(policy, start)
    ends = cfg["trip_ends"]
    source = ReplaySource(events, net, cfg["scope"], cfg["tour"], start, end,
                          picks_path=(cfg.get("data") or {}).get("picks_path"))
    work = source.work
    res = simulate(source, int(cfg["operators"]), policy, net, leg_model,
                   cfg["scope"], ends, move_cost=move_cost)
    t = res.tours
    extra = {
        "mode": "replay",
        "source": source.describe(),
        "window": {"start": cfg["window"]["start"], "end": cfg["window"]["end"],
                   "days": (end - start).days},
        "operators": int(cfg["operators"]),
        "policy": policy.name,
        "leg_time_model": leg_model.name,
        "leg_time": getattr(leg_model, "provenance", {"name": leg_model.name}),
        "dispatch_legs": ends["dispatch_legs"],
        "drop_off_mode": ends["drop_off_mode"],
        "trip_ends": dict(ends),
        "scope": dict(cfg["scope"]),
        "network": net.summary(),
        "workload": work.counts,
        "moves": {"n": len(res.moves), "cost": res.move_cost},
        "leg_model_check": leg_model_check(res, cfg["leg_model_check"], leg_model.name),
        "descriptive": {"tours": int(len(t)), "picks": int(t["n_picks"].sum()),
                        "pick_feet": float(t["pick_feet"].sum()),
                        "dispatch_feet_untimed": float(t["dispatch_feet"].sum())},
        "not_findings": {"note": NOT_FINDINGS,
                         "busy_hours": float(t["busy_s"].sum() / 3600),
                         "wait_hours": float(t["wait_s"].sum() / 3600),
                         "tours_that_waited": int((t["wait_s"] > 0).sum()),
                         "peak_operators_busy": int(t["operators_busy_at_start"].max()) if len(t) else 0},
    }
    return RunOutput(res, work.dropped, extra, work.tours)
