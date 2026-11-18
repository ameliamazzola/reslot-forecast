import numpy as np
from .engine.algorithm import World
from .engine.plus import PlannerPlus, limit_swaps

SETTINGS = dict(lookback=1, K=30, margin=2.0, top=20, passes=2, cap=100, cluster=3.0, refill=True, stable="auto")
MAX_CLUSTER = 20

class _ClusterPlanner(PlannerPlus):
    weight = 0.0; mates = None

    def set_clusters(self, lc, weight):
        self.weight = weight; by = {}
        for l, c in lc.items(): by.setdefault(c, []).append(l)
        self.mates = {l: sorted([x for x in by[c] if x != l], key=lambda x: -self.vis.get(x, 0))[:MAX_CLUSTER] for l, c in lc.items()}
        counts = [c for v in self.aff.values() for c, _ in v]
        self.typical = float(np.median(counts)) if counts else 1.0

    def aff_partners(self, hot, moved):
        v = list(self.aff.get(hot, []))
        if self.weight and self.mates and hot in self.mates:
            have = {b for _, b in v}
            v += [(self.weight * self.typical, b) for b in self.mates[hot] if b not in have]
        if not v: return []
        idx = np.array([b for _, b in v]); w = np.array([c for c, _ in v], float)
        score = self.Dn[:, idx] @ w
        vh = self.vis.get(hot, 0)
        cand = [l for l in np.argsort(score)[:400] if l != hot and l not in moved and 0 < self.vis.get(l, 0) < vh]
        return [int(l) for l in cand]

def plan(tours, D, depot, cases, golden, dollars_per_unit, cost_per_case, groups=None, loc_cluster=None, refill=None, window_months=1, sizes=None, settings=None):
    p = dict(SETTINGS, **(settings or {}))
    W = World(D, depot, cases, golden, dollars_per_unit / max(window_months, 1), cost_per_case)
    sizes = sizes or [len(tours)]
    if loc_cluster and p["cluster"]:
        pl = _ClusterPlanner(W, tours, part_sizes=sizes, stable=p["stable"]); pl.build_affinity(top=p["top"]); pl.set_clusters(loc_cluster, p["cluster"])
    else:
        pl = PlannerPlus(W, tours, part_sizes=sizes, stable=p["stable"]); pl.build_affinity(top=p["top"])
    pl.K, pl.margin = p["K"], p["margin"]
    if refill and p["refill"]: pl.refill = refill
    order = [l for l in pl.order_detour if l in pl.aff]
    same = (lambda hot, q: groups[q] == groups[hot]) if groups is not None else (lambda hot, q: True)
    parts = lambda hot, mv: [q for q in pl.aff_partners(hot, mv) if same(hot, q)]
    moved, _, _ = pl.greedy(order, True, partners=parts)
    for _ in range(p["passes"] - 1):
        m2, _, _ = pl.greedy(order, True, partners=parts, moved0=moved)
        if m2 == moved: break
        moved = m2
    return limit_swaps(W, tours, moved, p["cap"])
