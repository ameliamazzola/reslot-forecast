from collections import defaultdict
import numpy as np
import re
from .algorithm import Planner, World, replay, move_cost

REPEAT_MIN = 0.60

class PlannerPlus(Planner):
    def __init__(self, W, train, part_sizes=None, stable=False, **kw):
        super().__init__(W, train, **kw)
        self.repeat = None
        if stable == "auto" and part_sizes and len(part_sizes) > 1:
            starts = np.cumsum([0] + list(part_sizes)); sets = []
            for a, b in zip(starts[:-1], starts[1:]):
                c = defaultdict(int)
                for t in train[a:b]:
                    for l in set(t): c[l] += 1
                k = max(1, int(len(c) * 0.07)); sets.append(set(sorted(c, key=lambda l: -c[l])[:k]))
            self.repeat = float(np.mean([len(x & y) / max(len(x), 1) for x, y in zip(sets[:-1], sets[1:])]))
            stable = self.repeat < REPEAT_MIN
        self.stable = bool(stable) and bool(part_sizes) and len(part_sizes) > 1
        if self.stable:
            self.pid = []; 
            for i, n in enumerate(part_sizes): self.pid += [i] * n
            self.pid += [len(part_sizes) - 1] * (len(train) - len(self.pid)); self.np = len(part_sizes)

    def try_swap(self, hot, partner, moved, cur_len):
        saved, cost, w, al = super().try_swap(hot, partner, moved, cur_len)
        rf = getattr(self, "refill", None)
        if rf:
            if rf[0] == "prod":
                _, rate, orow, ratio = rf; d = 0.0
                for a, b in ((hot, partner), (partner, hot)):
                    if a in orow: d += rate[a] * (orow[a][b] - orow[a][a])
                cost += 2 * self.W.dpu * ratio * d
            else:
                rate, row, ratio = rf
                cost += 2 * self.W.dpu * ratio * (rate[hot] * (row[partner] - row[hot]) + rate[partner] * (row[hot] - row[partner]))
        cr = getattr(self, "crowd", None)
        if cr:
            aisle, kappa = cr
            def loads(mv):
                ld = {}
                for l, v in self.vis.items():
                    a = aisle[mv.get(l, l)]; ld[a] = ld.get(a, 0) + v
                return ld
            la, lb = loads(moved), loads(w)
            cost += kappa * (sum(v * v for v in lb.values()) - sum(v * v for v in la.values()))
        if self.stable:
            per = [0.0] * self.np
            for i, v in al.items(): per[self.pid[i]] += (cur_len[i] - v) * self.W.dpu
            saved = min(saved, self.np * min(per))
        return saved, cost, w, al
    def nearer_quiet(self, hot, moved):
        dh, vh = self.dep_dist[hot], self.vis.get(hot, 0)
        sc = []
        for l, v in self.vis.items():
            if l == hot or l in moved: continue
            dd, dv = dh - self.dep_dist[l], vh - v
            if dd > 0 and dv > 0: sc.append((dd * dv, l))
        sc.sort(reverse=True)
        return [l for _, l in sc]

    def plus(self, order, K=10, passes=3, ring=True):
        old = self.K; self.K = K
        moved, n, c = self.greedy(order, True)
        for _ in range(passes - 1 if ring else 0):
            m2, n2, c2 = self.greedy(order, True, partners=self.nearer_quiet, moved0=moved)
            if n2 == 0 and m2 == moved: break
            moved, n, c = m2, n + n2 - 0, c + c2
        self.K = old
        return moved, n, c

    def cluster_two_way(self, moved0, loc_cluster, order):
        vis_in = defaultdict(list)
        for l, c in loc_cluster.items(): vis_in[c].append(self.vis.get(l, 0))
        ranked = sorted(vis_in, key=lambda c: -np.mean(vis_in[c]))
        pool = sorted(loc_cluster, key=lambda l: self.dep_dist[l]); band, pos = {}, 0
        for c in ranked:
            for l in pool[pos:pos + len(vis_in[c])]: band[l] = c
            pos += len(vis_in[c])
        mis = [l for l in order if l in loc_cluster and band[l] != loc_cluster[l]]
        def match(hot, moved):
            tgt, here = loc_cluster[hot], band[hot]
            both = [l for l in loc_cluster if l not in moved and band[l] == tgt and loc_cluster[l] == here]
            one = [l for l in loc_cluster if l not in moved and band[l] == tgt and loc_cluster[l] != tgt and loc_cluster[l] != here]
            f = lambda l: (self.vis.get(l, 0), self.cases(l))
            return sorted(both, key=f) + sorted(one, key=f)
        return self.greedy(mis, True, partners=match, moved0=moved0)

    def rs_plus(self): return self.plus(self.order_detour, K=10, passes=3)

    def build_affinity(self, top=20, min_count=2):
        from itertools import combinations
        from collections import Counter
        cnt = Counter()
        for t in self.train:
            u = sorted(set(t))
            if len(u) > 40: u = u[:40]
            for a, b in combinations(u, 2): cnt[(a, b)] += 1
        nb = defaultdict(list)
        for (a, b), c in cnt.items():
            if c >= min_count: nb[a].append((c, b)); nb[b].append((c, a))
        self.aff = {a: sorted(v, reverse=True)[:top] for a, v in nb.items()}
        self.Dn = np.asarray(self.W.D)

        self.pull = {a: sum(c * self.Dn[a][b] for c, b in v) for a, v in self.aff.items()}

    def aff_partners(self, hot, moved):
        v = self.aff.get(hot)
        if not v: return []
        idx = np.array([b for _, b in v]); w = np.array([c for c, _ in v], float)
        score = self.Dn[:, idx] @ w
        vh = self.vis.get(hot, 0)
        cand = [l for l in np.argsort(score)[:400] if l != hot and l not in moved and 0 < self.vis.get(l, 0) < vh]
        return [int(l) for l in cand]

    def affinity_v2(self, K=10, passes=2):
        self.build_affinity(); order = sorted(self.pull, key=lambda a: -self.pull[a])
        old = self.K; self.K = K; moved, n, c = self.greedy(order, True, partners=self.aff_partners)
        for _ in range(passes - 1):
            m2, n2, c2 = self.greedy(order, True, partners=self.aff_partners, moved0=moved)
            if m2 == moved: break
            moved, n, c = m2, n + n2, c + c2
        self.K = old; return moved, n, c

    def family_grouping(self, n_swaps, min_jac=0.10, fam_size=8, cost_test=False):
        from itertools import combinations
        from collections import Counter
        cnt = Counter()
        for t in self.train:
            u = sorted(set(t))[:40]
            for a, b in combinations(u, 2): cnt[(a, b)] += 1
        nb = defaultdict(list)
        for (a, b), c in cnt.items():
            j = c / (self.vis.get(a, 0) + self.vis.get(b, 0) - c)
            if j >= min_jac: nb[a].append((j, b)); nb[b].append((j, a))
        seen, fams = set(), []
        for s in sorted(self.vis, key=lambda l: -self.vis[l]):
            if s in seen: continue
            f = [s]; seen.add(s)
            for j, b in sorted(nb.get(s, []), reverse=True):
                if len(f) >= fam_size: break
                if b not in seen and b in self.vis: f.append(b); seen.add(b)
            fams.append(f)
        fams.sort(key=lambda f: -sum(self.vis[l] for l in f))
        slots = sorted(range(self.W.n), key=lambda l: self.dep_dist[l]); target, k = {}, 0
        for f in fams:
            for l in sorted(f, key=lambda l: -self.vis[l]):
                target[l] = slots[k]; k += 1
        order = [l for f in fams for l in sorted(f, key=lambda l: -self.vis[l])]
        parts = lambda hot, moved: [target[hot]] if target[hot] != hot and target[hot] not in moved else []
        old = self.K; moved, n, c = self.greedy(order, cost_test, n_force=n_swaps, partners=parts); self.K = old
        return moved, n, c

def limit_swaps(W, tours, moved, cap):
    keep, seen = {}, set()
    for a, b in moved.items():
        if a in seen: continue
        if cap and len(seen) // 2 >= cap: break
        seen.add(a); seen.add(b); keep[a] = b; keep[b] = a
    return keep

def racksmart_sweep(inp, train, sizes, caps, window=1, demand_scale=1.0, cost_mult=1.0, guard=True, prep=None):
    Wp = World(inp.D, inp.depot, inp.cases, inp.golden, inp.dollars_per_unit * demand_scale / window, inp.cost_per_case * cost_mult)
    We = World(inp.D, inp.depot, inp.cases, inp.golden, inp.dollars_per_unit * demand_scale, inp.cost_per_case * cost_mult)
    faces = inp.extra.get("faces")
    grp = [1 if re.match(r"[A-Z]+", f).group(0) in ("GG", "HH") else 0 for f in faces] if faces else None
    def plan(tr, szs):
        pl = PlannerPlus(Wp, tr, part_sizes=szs, stable="auto"); pl.build_affinity(); pl.K = 10
        if prep: prep(pl)
        order = [l for l in pl.order_detour if l in pl.aff]
        parts = (lambda hot, mv, _b=pl.aff_partners: [p for p in _b(hot, mv) if grp[p] == grp[hot]]) if grp else pl.aff_partners
        moved, n, _ = pl.greedy(order, True, partners=parts)
        m2, n2, _ = pl.greedy(order, True, partners=parts, moved0=moved)
        if m2 != moved: moved = m2
        return moved, pl.repeat
    moved, rep = plan(train, sizes)
    do_guard = guard and len(sizes) > 1
    if do_guard:
        k = sizes[-1]; older, latest = train[:-k], train[-k:]
        m_old, _ = plan(older, sizes[:-1])
    out = {}
    for cap in caps:
        mv = limit_swaps(Wp, train, moved, cap); blocked = 0
        if do_guard:
            mo = limit_swaps(Wp, older, m_old, cap)
            s_old, _, _ = replay(We, latest, mo); c_old = move_cost(We, mo)
            proj = s_old * We.dpu * ((len(train) / window) / max(len(latest), 1)) - c_old
            if proj <= 0: mv, blocked = {}, 1
        out[cap] = (mv, len(mv) // 2, blocked, rep)
    return out
