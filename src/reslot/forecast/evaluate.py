"""
Rolling-origin backtesting and metrics.

Two families of metric, and the second one is the one that matters here:

  Accuracy   MAE / RMSE / MASE -- standard, needed for the committee.
  Decision   peak-hit rate, valley-hit rate, lead time -- does the forecast
             flag the surge early enough for a re-slot to pay off? A forecast
             can be mediocre on MASE and still be excellent for this policy,
             because the dispatcher only needs the TIMING of regime changes.
             Reporting only MASE would understate what the system does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mase(actual: np.ndarray, pred: np.ndarray, period: int = 7) -> float:
    """MAE scaled by in-sample seasonal-naive MAE. <1 beats seasonal naive."""
    if len(actual) <= period:
        return np.nan
    scale = np.mean(np.abs(actual[period:] - actual[:-period]))
    return np.mean(np.abs(actual - pred)) / scale if scale > 0 else np.nan


def accuracy_metrics(joined: pd.DataFrame, period: int = 7) -> dict:
    a, p = joined["qty_eaches"].to_numpy(), joined["yhat"].to_numpy()
    return {"mae": float(np.mean(np.abs(a - p))),
            "rmse": float(np.sqrt(np.mean((a - p) ** 2))),
            "bias": float(np.mean(p - a)),
            "mase": float(mase(a, p, period))}


def flag_regimes(series: pd.Series, hi: float = 1.3, lo: float = 0.8,
                 window: int = 28) -> pd.Series:
    """Label each day peak / valley / normal relative to a trailing mean.

    This is the bridge to the dispatcher: it consumes regime labels, not raw
    forecasts. Same function runs on actuals and on yhat, so 'did we call the
    peak' is a like-for-like comparison. hi/lo are tunable and should be swept.
    """
    ref = series.rolling(window, min_periods=7).mean()
    ratio = series / ref
    return pd.Series(np.where(ratio >= hi, "peak",
                     np.where(ratio <= lo, "valley", "normal")),
                     index=series.index)


def decision_metrics(joined: pd.DataFrame, tolerance_days: int = 1,
                     history: pd.Series | None = None, window: int = 28) -> dict:
    """Did the forecast call the regime, within tolerance?

    history: trailing ACTUAL daily totals from the training window. Regimes are
    relative to a trailing mean, so a 14-day test slice evaluated on its own has
    no trailing context and every day looks 'normal'. Prepending the training
    tail is not optional -- without it this metric silently reads zero.
    """
    j = joined.sort_values("date")
    a = j.set_index("date")["qty_eaches"]
    p_ = j.set_index("date")["yhat"]
    if history is not None and len(history):
        tail = history.tail(window)
        a = pd.concat([tail, a])
        p_ = pd.concat([tail, p_])   # forecast inherits real history, as the
                                     # dispatcher would at decision time
    test_days = set(j["date"])
    truth = flag_regimes(a, window=window).loc[lambda s: s.index.isin(test_days)]
    pred = flag_regimes(p_, window=window).loc[lambda s: s.index.isin(test_days)]
    out = {}
    for regime in ("peak", "valley"):
        t_days = set(truth[truth == regime].index)
        p_days = set(pred[pred == regime].index)
        hit = sum(any(abs((t - p).days) <= tolerance_days for p in p_days) for t in t_days)
        out[f"{regime}_recall"] = hit / len(t_days) if t_days else np.nan
        fp = sum(not any(abs((p - t).days) <= tolerance_days for t in t_days) for p in p_days)
        out[f"{regime}_false_alarm"] = fp / len(p_days) if p_days else np.nan
    return out


def rolling_origin(panel: pd.DataFrame, model_factory, horizon: int = 14,
                   n_splits: int = 6, step: int = 14) -> pd.DataFrame:
    """Walk-forward evaluation. model_factory is a zero-arg callable so each
    fold gets a fresh, unfitted model -- no leakage across folds."""
    dates = np.sort(panel["date"].unique())
    results = []
    for k in range(n_splits):
        cut_idx = len(dates) - horizon - step * (n_splits - 1 - k)
        if cut_idx <= horizon:
            continue
        cut = dates[cut_idx - 1]
        train = panel[panel["date"] <= cut]
        test = panel[(panel["date"] > cut) &
                     (panel["date"] <= cut + pd.Timedelta(days=horizon))]
        fc = model_factory().fit_predict(train, horizon)
        j = test.merge(fc, on=["item_id", "date"], how="inner")
        if j.empty:
            continue
        agg = j.groupby("date")[["qty_eaches", "yhat"]].sum().reset_index()
        hist = train.groupby("date")["qty_eaches"].sum()
        results.append({"fold": k, "cutoff": cut, "n_obs": len(j),
                        **accuracy_metrics(j),
                        **decision_metrics(agg, history=hist)})
    return pd.DataFrame(results)
