"""Forecaster interface. Every model implements this and nothing else, so the
dispatcher can be handed any of them and the comparison stays apples-to-apples."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Forecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, panel: pd.DataFrame) -> "Forecaster":
        """panel: densified PANEL frame, training window only."""

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """Return columns: item_id, date, yhat. One row per item per future day."""

    def fit_predict(self, panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
        return self.fit(panel).predict(horizon)
