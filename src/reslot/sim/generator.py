"""Order-generator profile, empirical mode, SKU -> face layouts, and the
generated-vs-real comparison. The generator itself is the team's, in
src/reslot/orders.py (OrderGenerator), used unchanged; its Order class is
imported here as GenOrder (src/reslot/sim/orders.py has no Order class of its
own -- the engine's order is workload.Tour).

Definitions (shared by the profile fit and the comparison, applied the same way
to real and generated orders):

  order          one WMS order number (ORD_NO) with at least one in-scope case
                 pick (event_class pick, zone WZPCK, UOM CS) in the window; the
                 generator's Order. Released at its first in-scope pick (proxy;
                 the export has no release time).
  line           one distinct SKU on an order (the generator draws distinct SKUs)
  cases per line the cases of that SKU on that order (qty summed), rounded to a
                 whole case; lines with < 1 case are left out
  hour / day     clock hour (0-23) and weekday (Mon = 0) of the order's release

Excluded SKUs (excluded_skus; the affinity pipeline's filter, src/affinity/
build_matrix.py steps 1 and 1b): SKUs picked from the VMS zone (WZVMS) in the
window unless their description matches the keep pattern (ARRID), and SKUs whose
description matches the discontinued vitamin brands (VITFUS|CRITTER), wherever
stored. A missing description counts as no match, as in the pipeline. Their lines
are dropped from every order before anything is counted; an order left with no
line is dropped.

Profile (fit_profile), from the planning window only, after the exclusion:
  orders_per_day, lines-per-order and cases-per-line histograms, hour-of-day
  and day-of-week shares of orders, SKU rates (lines per day), and clusters
  (Louvain on order co-occurrence: pair support >= min_support, Jaccard >=
  min_jaccard -- the affinity pipeline's settings, recomputed on the planning
  month alone, not read from sku_clusters.csv).

Empirical mode (EmpiricalOrderGenerator, a subclass; OrderGenerator's default
behaviour is untouched): lines per order and cases per line are drawn from the
profile's histograms, and the arrival rate in each minute is
    orders_per_day x 7 x day_share[weekday] x hour_share[hour] / 60,
with hour_share (generator.hour_profile):
    split   weekday hours (Mon-Fri orders) or weekend hours (Sat-Sun orders),
            whichever the minute falls on -- the fix agreed for the weekend score
    single  one hour-of-day profile over all days (the first version, kept so the
            before/after is reproducible)
Both are fitted on the planning window only.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pandas as pd

from ..orders import DemandProfile, OrderGenerator
from ..orders import Order as GenOrder

SCHEMA = "racksmart-generator-profile/1"
MODE_LABELS = {
    "default": "team generator as written ({h:02d}:00 + {m} min shift)",
    "empirical": "empirical: planning-window line, case, day-of-week and {hours} hour-of-day profile",
}
HOUR_PROFILES = {"split": "weekday / weekend", "single": "single"}


def mode_label(mode: str, default_cfg: dict | None = None, hour_profile: str = "split") -> str:
    d = default_cfg or {}
    return MODE_LABELS[mode].format(h=int(d.get("shift_start_hour", 6)), m=int(d.get("shift_minutes", 960)),
                                    hours=HOUR_PROFILES[hour_profile])
DEFINITIONS = {
    "order": "ORD_NO with >= 1 in-scope case pick in the window; released at its first in-scope pick (proxy)",
    "line": "distinct SKU on an order",
    "cases_per_line": "qty of that SKU on that order, summed and rounded to whole cases; < 1 case left out",
    "hour_of_day / day_of_week": "of the order's release; Monday = 0",
    "sku_rate": "lines per day over the window",
}


# --------------------------------------------------------------------------
# real orders in a window
# --------------------------------------------------------------------------

def order_lines(events: pd.DataFrame, scope: dict, start, end) -> pd.DataFrame:
    """One row per (order, SKU): cases, first pick ts, and the order's release."""
    s = events[(events["event_class"] == scope["event_class"]) & (events["from_zone"] == scope["zone"])
               & (events["uom"] == scope["uom"]) & (events["ts"] >= pd.Timestamp(start))
               & (events["ts"] < pd.Timestamp(end)) & events["order_id"].notna() & events["item_id"].notna()]
    if s.empty:
        return pd.DataFrame(columns=["order_id", "sku", "cases", "first_ts", "released"])
    g = (s.assign(sku=s["item_id"].astype(str).str.strip(), q=pd.to_numeric(s["qty"], errors="coerce").fillna(0))
         .groupby(["order_id", "sku"], sort=True).agg(cases=("q", "sum"), first_ts=("ts", "min")).reset_index())
    g["cases"] = g["cases"].round().astype(int)
    g = g[g["cases"] >= 1]
    g["released"] = g.groupby("order_id")["first_ts"].transform("min")
    return g.reset_index(drop=True)


def excluded_skus(events: pd.DataFrame, start, end, exclude: dict | None) -> tuple[set, dict]:
    """The affinity pipeline's VMS + vitamin-brand SKU filter, over every zone's
    picks in [start, end). exclude: {vms_zone, vms_keep_pattern, vitamin_pattern}."""
    if not exclude:
        return set(), {"rule": "none"}
    w = events[(events["event_class"] == "pick") & (events["ts"] >= pd.Timestamp(start))
               & (events["ts"] < pd.Timestamp(end)) & events["item_id"].notna()]
    item = w["item_id"].astype(str).str.strip()
    descr = w["item_descr"] if "item_descr" in w else pd.Series(pd.NA, index=w.index, dtype="string")
    keep = descr.str.contains(exclude["vms_keep_pattern"], case=False, na=False)
    vms = set(item[(w["from_zone"] == exclude["vms_zone"]) & ~keep])
    vit = set(item[descr.str.contains(exclude["vitamin_pattern"], case=False, na=False)])
    no_descr = set(item[(w["from_zone"] == exclude["vms_zone"]) & descr.isna()])
    return vms | vit, {"rule": "affinity pipeline steps 1 + 1b (src/affinity/build_matrix.py)", **dict(exclude),
                       "vms_skus": len(vms), "vitamin_brand_skus": len(vit), "excluded_skus": len(vms | vit),
                       "vms_skus_without_description": len(no_descr)}


def drop_skus(lines: pd.DataFrame, skus: set) -> pd.DataFrame:
    """Drop the lines of excluded SKUs; re-derive each order's release from what is left."""
    if not skus:
        return lines
    out = lines[~lines["sku"].isin(skus)].copy()
    out["released"] = out.groupby("order_id")["first_ts"].transform("min")
    return out.reset_index(drop=True)


def _hist(values) -> dict:
    c = Counter(int(v) for v in values)
    return {int(k): int(c[k]) for k in sorted(c)}


def order_stats(lines: pd.DataFrame, start, end) -> dict:
    """The statistics both the profile and the comparison use."""
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    orders = lines.groupby("order_id").agg(released=("released", "first"), n_lines=("sku", "size"))
    hours = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="h", inclusive="left")
    per_hour = orders.groupby(orders["released"].dt.floor("h")).size().reindex(hours, fill_value=0)
    weekend = per_hour.index.dayofweek >= 5
    sku_lines = lines.groupby("sku").size().sort_values(ascending=False)
    return {"days": int(days), "orders": int(len(orders)), "lines": int(len(lines)),
            "orders_per_day": float(len(orders) / days) if days else 0.0,
            "orders_per_hour_hist": _hist(per_hour.to_numpy()),
            "orders_per_hour_hist_weekday": _hist(per_hour[~weekend].to_numpy()),
            "orders_per_hour_hist_weekend": _hist(per_hour[weekend].to_numpy()),
            "lines_per_order_hist": _hist(orders["n_lines"]),
            "cases_per_line_hist": _hist(lines["cases"]),
            "share_1_2_line_orders": float((orders["n_lines"] <= 2).mean()) if len(orders) else 0.0,
            "hour_of_day_share": _shares(orders["released"].dt.hour, 24),
            "hour_of_day_share_weekday": _shares(orders.loc[orders["released"].dt.dayofweek < 5, "released"].dt.hour, 24),
            "hour_of_day_share_weekend": _shares(orders.loc[orders["released"].dt.dayofweek >= 5, "released"].dt.hour, 24),
            "orders_weekday": int((orders["released"].dt.dayofweek < 5).sum()),
            "orders_weekend": int((orders["released"].dt.dayofweek >= 5).sum()),
            "day_of_week_share": _shares(orders["released"].dt.dayofweek, 7),
            "sku_lines": {str(k): int(v) for k, v in sku_lines.items()}}


def _shares(values: pd.Series, n: int) -> list:
    c = np.bincount(values.astype(int).to_numpy(), minlength=n)[:n].astype(float)
    return (c / c.sum()).tolist() if c.sum() else [1.0 / n] * n


# --------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------

def fit_clusters(lines: pd.DataFrame, min_support: float, min_jaccard: float, resolution: float, seed: int) -> dict:
    """SKU -> cluster: Louvain on the order co-occurrence graph. Edges: SKU pairs on
    >= min_support of orders with Jaccard >= min_jaccard (fpgrowth with max_len 2
    finds the same pairs). SKUs with no edge get no cluster."""
    import networkx as nx
    baskets = lines.groupby("order_id")["sku"].apply(lambda s: sorted(set(s)))
    n = len(baskets)
    if n == 0:
        return {}
    single = Counter(s for b in baskets for s in b)
    keep = {s for s, c in single.items() if c / n >= min_support}          # a frequent pair needs frequent items
    pairs = Counter(p for b in baskets for p in itertools.combinations([s for s in b if s in keep], 2))
    G = nx.Graph()
    for (a, b), c in pairs.items():
        if c / n < min_support:
            continue
        j = c / (single[a] + single[b] - c)
        if j >= min_jaccard:
            G.add_edge(a, b, weight=j)
    if G.number_of_nodes() == 0:
        return {}
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    comms = sorted((sorted(c) for c in comms), key=lambda c: (-len(c), c[0]))
    return {s: k for k, c in enumerate(comms) for s in c}


def fit_profile(events: pd.DataFrame, scope: dict, start, end, cluster_cfg: dict,
                exclude: dict | None = None) -> dict:
    """The generator profile from [start, end) only, VMS / vitamin-brand SKUs excluded."""
    skip, skip_info = excluded_skus(events, start, end, exclude)
    all_lines = order_lines(events, scope, start, end)
    lines = drop_skus(all_lines, skip)
    skip_info.update(lines_dropped=int(len(all_lines) - len(lines)),
                     orders_dropped=int(all_lines["order_id"].nunique() - lines["order_id"].nunique()))
    if lines.empty:
        raise ValueError(f"no in-scope case-pick orders between {start} and {end}")
    st = order_stats(lines, start, end)
    clusters = fit_clusters(lines, **{k: cluster_cfg[k] for k in ("min_support", "min_jaccard", "resolution", "seed")})
    return {"schema": SCHEMA, "window": [str(pd.Timestamp(start)), str(pd.Timestamp(end))], "scope": dict(scope),
            "definitions": DEFINITIONS, "days": st["days"], "orders": st["orders"],
            "orders_per_day": st["orders_per_day"],
            "lines_per_order_hist": st["lines_per_order_hist"], "cases_per_line_hist": st["cases_per_line_hist"],
            "hour_of_day_share": st["hour_of_day_share"], "day_of_week_share": st["day_of_week_share"],
            "hour_of_day_share_weekday": st["hour_of_day_share_weekday"],
            "hour_of_day_share_weekend": st["hour_of_day_share_weekend"],
            "orders_weekday": st["orders_weekday"], "orders_weekend": st["orders_weekend"],
            "sku_rates": {s: n / st["days"] for s, n in st["sku_lines"].items()},
            "clusters": clusters, "cluster_settings": dict(cluster_cfg),
            "n_clusters": len(set(clusters.values())), "skus_with_cluster": len(clusters),
            "excluded_skus": sorted(skip), "exclusion": skip_info}


def _int_keys(h: dict) -> dict:
    return {int(k): int(v) for k, v in h.items()}


def demand_profile(profile: dict, skus=None) -> DemandProfile:
    rates = pd.Series(profile["sku_rates"], dtype=float)
    if skus is not None:
        rates = rates[rates.index.isin(list(skus))]
    clusters = {s: int(c) for s, c in profile["clusters"].items() if s in rates.index}
    return DemandProfile(rates, clusters, source=f"planning-window profile {profile['window'][0]} .. {profile['window'][1]}")


# --------------------------------------------------------------------------
# empirical mode
# --------------------------------------------------------------------------

class EmpiricalOrderGenerator(OrderGenerator):
    """OrderGenerator with lines per order and cases per line drawn from fitted
    histograms. SKU choice (rates, clusters, affinity) is the parent's. A line
    count the SKU draw can't fill (fewer distinct SKUs found than drawn) is
    redrawn, so every emitted count is in the histogram's support."""

    def __init__(self, profile: DemandProfile, orders_per_day: float, lines_hist: dict, cases_hist: dict,
                 affinity: float = 0.5, seed: int = 42):
        lh, ch = _int_keys(lines_hist), _int_keys(cases_hist)
        n_skus = len(profile.rates)
        lh = {k: v for k, v in lh.items() if 1 <= k <= n_skus and v > 0}
        ch = {k: v for k, v in ch.items() if k >= 1 and v > 0}
        if not lh or not ch:
            raise ValueError("empirical mode needs non-empty line and case histograms")
        super().__init__(profile, orders_per_day, shift_minutes=1440, min_lines=min(lh), max_lines=max(lh),
                         affinity=affinity, seed=seed)
        self.l_vals = np.array(sorted(lh))
        self.l_p = np.array([lh[k] for k in self.l_vals], float) / sum(lh.values())
        self.c_vals = np.array(sorted(ch))
        self.c_p = np.array([ch[k] for k in self.c_vals], float) / sum(ch.values())
        self.line_support, self.case_support = set(int(v) for v in self.l_vals), set(int(v) for v in self.c_vals)

    def _n_lines(self) -> int:
        return int(self.rng.choice(self.l_vals, p=self.l_p))

    def sample_order(self, minute: int) -> GenOrder:
        for _ in range(1000):
            o = super().sample_order(minute)
            if len(o.lines) in self.line_support:
                break
            self._n -= 1                                     # redraw under the same order number
        else:
            raise RuntimeError("could not draw an order with a line count in the fitted support")
        cases = self.rng.choice(self.c_vals, p=self.c_p, size=len(o.lines))
        return GenOrder(o.order_id, minute, list(zip(o.skus, [int(c) for c in cases])))


def minute_multiplier(gen: OrderGenerator, ts: pd.Timestamp, hour_share, day_share: list) -> float:
    """Multiplier for OrderGenerator.arrivals so that the minute's expected orders are
    orders_per_day x 7 x day_share x hour_share / 60. hour_share: one 24-list, or
    {"weekday": [...], "weekend": [...]} chosen by the minute's day."""
    if isinstance(hour_share, dict):
        hour_share = hour_share["weekend" if ts.dayofweek >= 5 else "weekday"]
    per_minute = gen.orders_per_day * 7 * day_share[ts.dayofweek] * hour_share[ts.hour] / 60
    return per_minute / (gen.orders_per_day / gen.shift_minutes)


# --------------------------------------------------------------------------
# SKU -> face
# --------------------------------------------------------------------------

def synthetic_layout(skus, faces: list, seed: int) -> dict:
    """Seeded random layout: each SKU on its own usable face."""
    skus, faces = sorted(skus), sorted(faces)
    if len(skus) > len(faces):
        raise ValueError(f"{len(skus)} SKUs but only {len(faces)} usable faces for a synthetic layout")
    pick = np.random.default_rng(seed).choice(len(faces), size=len(skus), replace=False)
    return {s: faces[i] for s, i in zip(skus, pick)}


def real_layout(face_items: pd.Series, faces: list) -> dict:
    """SKU -> face from the last item picked at each face before the run start
    (racksmart.face_items). An item last seen at several faces takes the first
    face in location order. Faces not usable are left out."""
    usable = set(faces)
    out = {}
    for face, item in face_items.sort_index().items():
        if face in usable:
            out.setdefault(str(item).strip(), face)
    return out


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def tvd(p: dict, q: dict) -> float:
    """Total variation distance between two histograms (counts or shares)."""
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    if not sp or not sq:
        return 1.0
    return 0.5 * sum(abs(p.get(k, 0) / sp - q.get(k, 0) / sq) for k in keys)


def _bin(h: dict, width: int, cap: int | None = None) -> dict:
    out = Counter()
    for k, v in h.items():
        k = int(k)
        b = (min(k, cap) if cap is not None else k) // width
        out[b] += v
    return dict(out)


def concentration(sku_lines: dict) -> dict:
    """SKU rank -> share of lines (identity dropped)."""
    v = sorted(sku_lines.values(), reverse=True)
    return {i: c for i, c in enumerate(v)}


def compare_stats(gen: dict, real: dict, cfg: dict) -> dict:
    """TVD per statistic, generated vs real; pass/fail only where a threshold is set."""
    b = cfg.get("bins") or {}
    w = b.get("orders_per_hour", 5)
    scores = {
        "orders_per_hour": tvd(_bin(gen["orders_per_hour_hist"], w), _bin(real["orders_per_hour_hist"], w)),
        "orders_per_hour_weekday": tvd(_bin(gen["orders_per_hour_hist_weekday"], w),
                                       _bin(real["orders_per_hour_hist_weekday"], w)),
        "orders_per_hour_weekend": tvd(_bin(gen["orders_per_hour_hist_weekend"], w),
                                       _bin(real["orders_per_hour_hist_weekend"], w)),
        "lines_per_order": tvd(_bin(gen["lines_per_order_hist"], 1, b.get("lines_cap", 30)),
                               _bin(real["lines_per_order_hist"], 1, b.get("lines_cap", 30))),
        "cases_per_line": tvd(_bin(gen["cases_per_line_hist"], 1, b.get("cases_cap", 50)),
                              _bin(real["cases_per_line_hist"], 1, b.get("cases_cap", 50))),
        # a two-bin distribution: TVD = |difference of the shares|
        "share_1_2_line_orders": abs(gen["share_1_2_line_orders"] - real["share_1_2_line_orders"]),
        "sku_concentration": tvd(concentration(gen["sku_lines"]), concentration(real["sku_lines"])),
    }
    th = cfg.get("thresholds") or {}
    out = {}
    for k, v in scores.items():
        t = th.get(k)
        out[k] = {"tvd": float(v), "threshold": t, "pass": None if t is None else bool(v <= t)}
    return out


# --------------------------------------------------------------------------
# config + profile -> GeneratorSource
# --------------------------------------------------------------------------

def generator_lines(source) -> pd.DataFrame:
    """The generated orders as order_lines rows (same columns as real ones)."""
    rows = [{"order_id": o.order_id, "sku": s, "cases": int(c),
             "released": source.start + pd.Timedelta(minutes=o.minute)} for o in source.generated for s, c in o.lines]
    df = pd.DataFrame(rows, columns=["order_id", "sku", "cases", "released"])
    df["first_ts"] = df["released"]
    return df


def make_generator_source(gcfg: dict, scope: dict, net, profile: dict, start, days: int):
    """Build GeneratorSource from config `generator` and a fitted profile.

    sku_to_face: 'synthetic' -- seeded random layout on the usable faces (FACE_RULE);
                 'real' -- the last item picked at each face before the run start
                 (profile face_items, from the planning picks).
    SKUs with no face are left out of the demand profile; their share of the
    profile's lines is recorded."""
    from .orders import GeneratorSource
    from .racksmart import structural_faces

    if pd.Timestamp(profile["window"][1]) > pd.Timestamp(start):
        raise ValueError(f"profile window ends {profile['window'][1]}, after the run start {start}")
    faces = structural_faces(net, scope)
    rates = pd.Series(profile["sku_rates"], dtype=float).sort_values(ascending=False, kind="stable")
    how = gcfg.get("sku_to_face", "synthetic")
    if how == "synthetic":
        top = list(rates.index[: len(faces)])                     # more SKUs than faces: the busiest ones
        sku_face = synthetic_layout(top, faces, int(gcfg["layout_seed"]))
    elif how == "real":
        if not profile.get("face_items"):
            raise ValueError("sku_to_face: real needs face_items in the profile (fit_generator_profile.py)")
        sku_face = real_layout(pd.Series(profile["face_items"]), faces)
    else:
        raise ValueError(f"generator.sku_to_face must be 'synthetic' or 'real', got {how!r}")
    mapped = [s for s in rates.index if s in sku_face]
    if not mapped:
        raise ValueError("no profile SKU has a face")
    dp = demand_profile(profile, mapped)
    opd = float(gcfg.get("orders_per_day") or profile["orders_per_day"])
    mode = gcfg.get("mode", "empirical")
    if mode == "default":
        d = gcfg.get("default") or {}
        gen = OrderGenerator(dp, opd, shift_minutes=int(d.get("shift_minutes", 960)),
                             lines_per_order_mean=float(d.get("lines_per_order_mean", 6.0)),
                             min_lines=int(d.get("min_lines", 3)), max_lines=int(d.get("max_lines", 25)),
                             cases_per_line_mean=float(d.get("cases_per_line_mean", 2.0)),
                             affinity=float(gcfg.get("affinity", 0.5)), seed=int(gcfg["seed"]))
        shift_start = int(d.get("shift_start_hour", 6))
    elif mode == "empirical":
        gen = EmpiricalOrderGenerator(dp, opd, profile["lines_per_order_hist"], profile["cases_per_line_hist"],
                                      affinity=float(gcfg.get("affinity", 0.5)), seed=int(gcfg["seed"]))
        shift_start = 0
    else:
        raise ValueError(f"generator.mode must be 'default' or 'empirical', got {mode!r}")
    hour_profile = gcfg.get("hour_profile", "split")
    if hour_profile == "split":
        if "hour_of_day_share_weekday" not in profile:
            raise ValueError("hour_profile: split needs weekday / weekend hour shares in the profile; "
                             "rerun fit_generator_profile.py (or set hour_profile: single)")
        hour_share = {"weekday": profile["hour_of_day_share_weekday"], "weekend": profile["hour_of_day_share_weekend"]}
    elif hour_profile == "single":
        hour_share = profile["hour_of_day_share"]
    else:
        raise ValueError(f"generator.hour_profile must be 'split' or 'single', got {hour_profile!r}")
    info = {"sku_to_face": how, "layout_seed": gcfg.get("layout_seed") if how == "synthetic" else None,
            "hour_profile": hour_profile if mode == "empirical" else None,
            "skus_in_profile": int(len(rates)), "skus_with_face": len(mapped),
            "share_of_profile_lines_with_face": float(rates[mapped].sum() / rates.sum()),
            "usable_faces": len(faces), "orders_per_day": opd, "seed": int(gcfg["seed"])}
    return GeneratorSource(gen, {s: sku_face[s] for s in mapped}, pd.Timestamp(start), days, mode,
                           hour_share=hour_share, day_share=profile["day_of_week_share"],
                           shift_start_hour=shift_start, layout=info,
                           label=mode_label(mode, gcfg.get("default"), hour_profile))


def noise_floor(gcfg: dict, scope: dict, net, profile: dict, start, end, pairs: int, seed_base: int) -> dict:
    """The generator against itself: `pairs` pairs of generator seeds (seed_base + 1,
    + 2), (+ 3, + 4), ... over the same window and config, scored with the same
    statistics as generated vs real. The TVD two draws of one process show by chance:
    a threshold at or below it can't be met by any generator."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = (end - start).days

    def stats(seed):
        src = make_generator_source({**gcfg, "seed": seed}, scope, net, profile, start, days)
        for _ in src.orders():
            pass
        return order_stats(generator_lines(src), start, end)

    runs, seeds = [], []
    for i in range(int(pairs)):
        a, b = seed_base + 2 * i + 1, seed_base + 2 * i + 2
        seeds.append([a, b])
        runs.append(compare_stats(stats(a), stats(b), {"bins": (gcfg.get("compare") or {}).get("bins")}))
    out = {}
    for k in runs[0]:
        v = [r[k]["tvd"] for r in runs]
        out[k] = {"mean": float(np.mean(v)), "max": float(np.max(v)), "values": v}
    return {"seed_pairs": seeds, "scores": out}
