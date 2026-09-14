"""SARIMAX per item, with calendar/promo exogenous regressors.

Scope discipline: the contribution is the closed loop, not forecast accuracy
for its own sake. Tune until it beats seasonal-naive on MASE, then stop and
spend the time on the forecast -> re-slot -> measured-gain coupling.
"""

from __future__ import annotations

import warnings

import pandas as pd

from .base import Forecaster


class SarimaPerItem(Forecaster):
    name = "sarima"

    def __init__(self, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7),
                 min_obs: int = 60, exog_cols: list[str] | None = None,
                 fallback=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.min_obs = min_obs
        self.exog_cols = exog_cols or []
        # Items too sparse or too short for SARIMA fall back to this model.
        # Report how many items took the fallback -- it is a real caveat.
        from .baselines import SeasonalNaive
        self.fallback = fallback or SeasonalNaive(period=7)

    def fit(self, panel):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._models, self._fallback_items = {}, []
        self._last_date = panel["date"].max()
        for item, g in panel.sort_values("date").groupby("item_id"):
            y = g.set_index("date")["qty_eaches"].asfreq("D").fillna(0.0)
            if len(y) < self.min_obs or (y > 0).sum() < 10:
                self._fallback_items.append(item)
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._models[item] = SARIMAX(
                        y, order=self.order, seasonal_order=self.seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False
                    ).fit(disp=False)
            except Exception:
                self._fallback_items.append(item)

        if self._fallback_items:
            self.fallback.fit(panel[panel["item_id"].isin(self._fallback_items)])
        self._panel = panel
        return self

    def predict(self, horizon):
        frames = []
        for item, res in self._models.items():
            f = res.forecast(steps=horizon)
            frames.append(pd.DataFrame(
                {"item_id": item, "date": f.index, "yhat": f.to_numpy().clip(min=0)}))
        if self._fallback_items:
            frames.append(self.fallback.predict(horizon))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["item_id", "date", "yhat"])

    @property
    def fallback_share(self) -> float:
        n = len(self._models) + len(self._fallback_items)
        return len(self._fallback_items) / n if n else 0.0
