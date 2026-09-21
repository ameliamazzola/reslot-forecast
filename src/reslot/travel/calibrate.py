"""
Travel-time model, calibrated from observed pick-to-pick intervals.

C&D's WMS has no x/y/z, but every slot has a WORK_PATH: the order the WMS
routes pickers through the building. Consecutive picks within one trip give a
paired observation -- path distance against elapsed seconds -- and fitting it
turns WORK_PATH into a travel coordinate measured from the operation rather
than assumed from a crane model.

Additive model:

    seconds = fixed + per_case * qty + cross_aisle * I[aisle changed]
                    + distance(delta_path)

Quantity is in the model because it dominates. Leaving it out attributes
handling time to distance and inflates the apparent payoff of re-slotting --
the error this project most needs to avoid.

Parametric terms: median regression (resists the long tail of operator
pauses). Distance term: isotonic on the residual -- monotone, no assumed shape,
encoding only "further is never faster".

Every result carries source="scan_interval_proxy". Same convention as the
panel's "pick_proxy": the interval between scans is not a time study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

from ..schemas import TRIP_LEGS, validate

SOURCE = "scan_interval_proxy"
CAVEAT = ("travel time inferred from elapsed seconds between consecutive pick "
          "scans; includes scanning, pallet building and operator pauses; not "
          "a direct time study")

_TERMS = ["qty", "delta_path", "cross_aisle"]


def _tag(d: dict) -> dict:
    return {"source": SOURCE, "caveat": CAVEAT, **d}


# --------------------------------------------------------------------------
# Legs
# --------------------------------------------------------------------------

def pick_face_stream(events: pd.DataFrame, slots: pd.DataFrame,
                     zone: str, uom: str) -> pd.DataFrame:
    """Case picks from the pick face, joined to slot path and aisle.

    Geometry comes from SLOTS, not from parsing event codes, so the slot table
    is the single source of truth for where things are.
    """
    geo = slots.set_index("location")[["path", "aisle"]]
    s = events[(events["event_class"] == "pick")
               & (events["from_zone"] == zone)
               & (events["uom"] == uom)]
    s = s.join(geo, on="from_loc", how="inner")
    return s.dropna(subset=["path"]).sort_values("ts").reset_index(drop=True)


def extract_legs(stream: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Consecutive pick pairs within a trip -> TRIP_LEGS.

    A trip is one operator working one order. Gap bounds drop breaks, shift
    changes and non-pick work; both are config, not code.
    """
    keys, min_picks = cfg["trip"]["group_keys"], cfg["trip"]["min_picks"]
    lo, hi = cfg["legs"]["min_seconds"], cfg["legs"]["max_seconds"]

    df = stream.sort_values(keys + ["ts"]).copy()
    df = df[df.groupby(keys)["ts"].transform("size") >= min_picks]

    g = df.groupby(keys, sort=False)
    df["trip_id"] = g.ngroup()
    df["elapsed_s"] = g["ts"].diff().dt.total_seconds()
    df["delta_path"] = g["path"].diff().abs()
    # Nullable string compare yields NA on each trip's first pick (dropped
    # below anyway) and on unparsed codes; count those as an aisle change.
    df["cross_aisle"] = df["aisle"].ne(g["aisle"].shift()).fillna(True).astype("int64")

    legs = df.dropna(subset=["elapsed_s", "delta_path"])
    legs = legs[legs["elapsed_s"].between(lo, hi)
                & (legs["delta_path"] <= cfg["legs"]["max_delta_path"])]
    return validate(legs, TRIP_LEGS, "trip_legs", strict=True)[list(TRIP_LEGS)].reset_index(drop=True)


def split_by_trip(legs: pd.DataFrame, holdout: float, seed: int):
    """Split on trips, never on legs.

    Legs in one trip share an operator, an hour and an aisle. A leg-level split
    puts correlated observations on both sides and flatters the holdout error.
    """
    rng = np.random.default_rng(seed)
    trips = legs["trip_id"].unique()
    held = set(rng.choice(trips, size=int(len(trips) * holdout), replace=False))
    mask = legs["trip_id"].isin(held)
    return legs[~mask].copy(), legs[mask].copy()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def _bin_medians(x: pd.Series, y: pd.Series, n_bins: int) -> pd.DataFrame:
    df = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)})
    df["bin"] = pd.qcut(df["x"], n_bins, duplicates="drop")
    return (df.groupby("bin", observed=True)
              .agg(n=("y", "size"), delta_lo=("x", "min"), delta_hi=("x", "max"),
                   delta_mid=("x", "median"), value=("y", "median"))
              .reset_index(drop=True))


class TravelModel:
    def __init__(self, coef: dict, distance: IsotonicRegression | None = None):
        self.coef = coef
        self.distance = distance

    @classmethod
    def fit(cls, legs: pd.DataFrame, n_bins: int = 40, isotonic_residual: bool = True):
        X = sm.add_constant(legs[_TERMS].astype(float), has_constant="add")
        params = sm.QuantReg(legs["elapsed_s"].to_numpy(float), X.to_numpy()).fit(q=0.5).params
        coef = {"fixed_s": float(params[0]), "per_case_s": float(params[1]),
                "per_path_unit_s": float(params[2]), "cross_aisle_s": float(params[3])}

        distance = None
        if isotonic_residual:
            base = (coef["fixed_s"] + coef["per_case_s"] * legs["qty"]
                    + coef["cross_aisle_s"] * legs["cross_aisle"])
            prof = _bin_medians(legs["delta_path"], legs["elapsed_s"] - base, n_bins)
            distance = IsotonicRegression(increasing=True, out_of_bounds="clip")
            distance.fit(prof["delta_mid"], prof["value"], sample_weight=prof["n"])
        return cls(coef, distance)

    def _travel(self, delta) -> np.ndarray:
        delta = np.asarray(delta, float)
        if self.distance is not None:
            return self.distance.predict(delta)
        return self.coef["per_path_unit_s"] * delta

    def predict(self, legs: pd.DataFrame) -> np.ndarray:
        c = self.coef
        return (c["fixed_s"] + c["per_case_s"] * legs["qty"].to_numpy(float)
                + c["cross_aisle_s"] * legs["cross_aisle"].to_numpy(float)
                + self._travel(legs["delta_path"]))

    def move_cost(self, delta_path: float, cross_aisle: bool) -> float:
        """Cost of a REPOSITIONING move: travel only, no per-case handling.

        This is what a re-slot pays. Moving a pallet between slots does not
        incur per-case pick time.
        """
        c = self.coef
        return float(c["fixed_s"] + self._travel([delta_path])[0]
                     + (c["cross_aisle_s"] if cross_aisle else 0.0))


# --------------------------------------------------------------------------
# Evaluation and reporting
# --------------------------------------------------------------------------

def evaluate(model: TravelModel, legs: pd.DataFrame, train_median: float,
             distance_only: TravelModel | None = None) -> dict:
    """Holdout error against baselines the model must beat.

    The constant baseline is the one that matters: if distance adds nothing
    over 'always predict the median leg', the report should say so.
    """
    obs = legs["elapsed_s"].to_numpy(float)

    def err(p):
        return {"mae": float(np.mean(np.abs(p - obs))),
                "medae": float(np.median(np.abs(p - obs)))}

    out = {"n_legs": int(len(legs)),
           "constant": err(np.full_like(obs, train_median)),
           "full_model": err(model.predict(legs)),
           "spearman": float(pd.Series(model.predict(legs)).corr(pd.Series(obs), method="spearman"))}
    if distance_only is not None:
        out["distance_only"] = err(distance_only.coef["fixed_s"]
                                   + distance_only._travel(legs["delta_path"]))
    return _tag(out)


def fit_distance_only(legs: pd.DataFrame, n_bins: int) -> TravelModel:
    """Distance-only reference: median + isotonic on delta_path alone.

    Reported so the gain from adding quantity is visible, not asserted.
    """
    med = float(legs["elapsed_s"].median())
    prof = _bin_medians(legs["delta_path"], legs["elapsed_s"] - med, n_bins)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(prof["delta_mid"], prof["value"], sample_weight=prof["n"])
    return TravelModel({"fixed_s": med, "per_case_s": 0.0,
                        "per_path_unit_s": 0.0, "cross_aisle_s": 0.0}, iso)


def decompose(model: TravelModel, legs: pd.DataFrame) -> dict:
    """Share of observed pick time per component.

    Sizes the project's ceiling: re-slotting can only move path travel and
    cross-aisle, so their combined share bounds what any slotting policy can
    address.
    """
    c, total = model.coef, float(legs["elapsed_s"].sum())
    parts = {"fixed_handling": c["fixed_s"] * len(legs),
             "case_handling": c["per_case_s"] * float(legs["qty"].sum()),
             "path_travel": float(model._travel(legs["delta_path"]).sum()),
             "cross_aisle": c["cross_aisle_s"] * float(legs["cross_aisle"].sum())}
    shares = {k: v / total for k, v in parts.items()}
    shares["unexplained"] = 1.0 - sum(shares.values())
    shares["addressable_by_reslotting"] = shares["path_travel"] + shares["cross_aisle"]
    return _tag(shares)


def trip_check(model: TravelModel, legs: pd.DataFrame) -> dict:
    """Predicted vs observed trip duration. Leg noise averages down over a
    trip, and the trip is the level the simulator cares about."""
    df = legs.assign(pred=model.predict(legs))
    t = df.groupby("trip_id").agg(pred_s=("pred", "sum"), obs_s=("elapsed_s", "sum"))
    err = (t["pred_s"] - t["obs_s"]).abs()
    return _tag({"n_trips": int(len(t)),
                 "median_obs_seconds": float(t["obs_s"].median()),
                 "median_abs_error_seconds": float(err.median()),
                 "median_abs_pct_error": float((err / t["obs_s"]).median()),
                 "spearman": float(t["pred_s"].corr(t["obs_s"], method="spearman"))})


def distance_profile(legs: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Observed median seconds by distance bin. A report figure."""
    return _bin_medians(legs["delta_path"], legs["elapsed_s"], n_bins).rename(
        columns={"value": "median_seconds"})


def cost_table(model: TravelModel, max_delta: float, step: int = 50) -> pd.DataFrame:
    """The handoff the simulator reads: path distance in, repositioning seconds
    out. A CSV, not a pickled model -- inspectable, diffable, and independent
    of the scikit-learn version."""
    grid = np.arange(0, max_delta + step, step, dtype=float)
    return pd.DataFrame({"delta_path": grid,
                         "seconds_same_aisle": [model.move_cost(d, False) for d in grid],
                         "seconds_cross_aisle": [model.move_cost(d, True) for d in grid],
                         "source": SOURCE})
