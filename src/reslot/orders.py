"""
Random order generator for the live simulation.

Orders are drawn from a *demand profile* -- per-SKU expected picks per day --
so the stream has the same SKU mix the forecast (or the held-out actuals)
says it should. Everything random goes through one seeded Generator, so a demo
run can be replayed exactly.

    arrivals per sim-minute  ~ Poisson(orders_per_day / shift_minutes * regime)
    lines per order          ~ min_lines + Poisson(mean - min_lines), capped
    first SKU                ~ Categorical(rate_i / sum rate)
    next SKU                 with prob `affinity`: same cluster as a SKU already
                             on the order (rate-weighted); otherwise global draw
    cases per line           ~ 1 + Poisson(cases_per_line_mean - 1)

`affinity` is what makes co-picked SKUs show up together, which is what
RackSmart's co-pick partner step and the cluster step exploit. Set it to 0 and
orders become independent draws -- a useful control in the demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Order:
    order_id: str
    minute: int                    # sim minute the order was released
    lines: list[tuple[str, int]]   # (sku, cases)

    @property
    def skus(self) -> list[str]:
        return [s for s, _ in self.lines]


@dataclass
class DemandProfile:
    rates: pd.Series                       # sku -> expected picks (lines) per day
    clusters: dict[str, int] = field(default_factory=dict)
    source: str = "unspecified"            # carried into every result, like panel.source

    def __post_init__(self):
        r = self.rates.astype(float).clip(lower=0)
        self.rates = r[r > 0]
        if self.rates.empty:
            raise ValueError("demand profile has no SKU with a positive rate")


class OrderGenerator:
    def __init__(self, profile: DemandProfile, orders_per_day: float, shift_minutes: int = 960,
                 lines_per_order_mean: float = 6.0, min_lines: int = 3, max_lines: int = 25,
                 cases_per_line_mean: float = 2.0, affinity: float = 0.5, seed: int = 42):
        self.profile = profile
        self.orders_per_day = float(orders_per_day)
        self.shift_minutes = int(shift_minutes)
        self.lpo_mean, self.min_lines, self.max_lines = lines_per_order_mean, min_lines, max_lines
        self.cpl_mean = cases_per_line_mean
        self.affinity = affinity
        self.rng = np.random.default_rng(seed)
        self._n = 0

        self.skus = np.array(profile.rates.index, dtype=object)
        w = profile.rates.values
        self.p = w / w.sum()
        self.cluster_of = {s: profile.clusters.get(s) for s in self.skus}
        self.by_cluster: dict = {}
        for s, wt in zip(self.skus, w):
            c = self.cluster_of[s]
            if c is not None and c >= 0:
                self.by_cluster.setdefault(c, ([], []))
                self.by_cluster[c][0].append(s)
                self.by_cluster[c][1].append(wt)
        self.by_cluster = {c: (np.array(s, dtype=object), np.array(wt) / np.sum(wt))
                           for c, (s, wt) in self.by_cluster.items() if len(s) > 1}

    # ------------------------------------------------------------------
    def set_affinity(self, a: float):
        self.affinity = float(a)

    def _n_lines(self) -> int:
        k = self.min_lines + self.rng.poisson(max(self.lpo_mean - self.min_lines, 0))
        return int(min(k, self.max_lines, len(self.skus)))

    def sample_order(self, minute: int) -> Order:
        k = self._n_lines()
        chosen: list[str] = []
        tries = 0
        while len(chosen) < k and tries < 50 * k:   # cap: a few SKUs can dominate the weights
            tries += 1
            s = None
            if chosen and self.affinity > 0 and self.rng.random() < self.affinity:
                anchor = chosen[self.rng.integers(len(chosen))]
                c = self.cluster_of.get(anchor)
                if c in self.by_cluster:
                    cs, cp = self.by_cluster[c]
                    s = cs[self.rng.choice(len(cs), p=cp)]
            if s is None or s in chosen:          # duplicate from the cluster -> global draw
                s = self.skus[self.rng.choice(len(self.skus), p=self.p)]
            if s not in chosen:
                chosen.append(s)
        cases = 1 + self.rng.poisson(max(self.cpl_mean - 1, 0), size=len(chosen))
        self._n += 1
        return Order(f"SIM{self._n:06d}", minute, list(zip(chosen, cases.astype(int).tolist())))

    def arrivals(self, minute: int, multiplier: float = 1.0) -> list[Order]:
        """Orders released in one sim minute."""
        lam = self.orders_per_day / self.shift_minutes * multiplier
        return [self.sample_order(minute) for _ in range(self.rng.poisson(lam))]
