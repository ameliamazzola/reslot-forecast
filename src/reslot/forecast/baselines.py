"""Honest baselines. These run first and they are not throwaways.

If SARIMA cannot beat seasonal-naive on MASE, that is a finding to report, not
a problem to hide -- and the anticipatory policy should then be driven by the
simplest model that works. Baseline honesty is the project's ground rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


class SeasonalNaive(Forecaster):
    """yhat(t) = y(t - period). The number everything else must beat."""
    name = "seasonal_naive"

    def __init__(self, period: int = 7):
        self.period = period

    def fit(self, panel):
        self._panel = panel.sort_values(["item_id", "date"])
        self._last_date = panel["date"].max()
        return self

    def predict(self, horizon):
        rows = []
        for item, g in self._panel.groupby("item_id"):
            hist = g.set_index("date")["qty_eaches"]
            for h in range(1, horizon + 1):
                d = self._last_date + pd.Timedelta(days=h)
                src = d - pd.Timedelta(days=self.period * (1 + (h - 1) // self.period))
                rows.append((item, d, float(hist.get(src, hist.tail(self.period).mean()))))
        return pd.DataFrame(rows, columns=["item_id", "date", "yhat"])


class MovingAverage(Forecaster):
    name = "moving_average"

    def __init__(self, window: int = 28):
        self.window = window

    def fit(self, panel):
        self._mu = (panel.sort_values("date")
                         .groupby("item_id")["qty_eaches"]
                         .apply(lambda s: s.tail(self.window).mean()))
        self._last_date = panel["date"].max()
        return self

    def predict(self, horizon):
        dates = [self._last_date + pd.Timedelta(days=h) for h in range(1, horizon + 1)]
        return pd.DataFrame(
            [(i, d, float(v)) for i, v in self._mu.items() for d in dates],
            columns=["item_id", "date", "yhat"])


class Croston(Forecaster):
    """For the intermittent C-class tail, where SARIMA is the wrong tool.
    Separate exponential smoothing of demand size and inter-arrival interval."""
    name = "croston"

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def fit(self, panel):
        out = {}
        for item, g in panel.sort_values("date").groupby("item_id"):
            y = g["qty_eaches"].to_numpy()
            nz = np.flatnonzero(y)
            if len(nz) == 0:
                out[item] = 0.0
                continue
            z, x, prev = y[nz[0]], max(nz[0], 1), nz[0]
            for i in nz[1:]:
                z += self.alpha * (y[i] - z)
                x += self.alpha * ((i - prev) - x)
                prev = i
            out[item] = float(z / max(x, 1e-9))
        self._rate, self._last_date = out, panel["date"].max()
        return self

    def predict(self, horizon):
        dates = [self._last_date + pd.Timedelta(days=h) for h in range(1, horizon + 1)]
        return pd.DataFrame([(i, d, v) for i, v in self._rate.items() for d in dates],
                            columns=["item_id", "date", "yhat"])
