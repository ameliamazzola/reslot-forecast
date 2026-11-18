import os
import numpy as np
from collections import defaultdict

class World:
    def __init__(self, D, depot, cases, golden, dollars_per_unit, cost_per_case):
        self.D = [list(map(float, r)) for r in np.asarray(D)]
        self.depot = depot
        self.cases = cases
        self.golden = list(golden)
        self.dpu = dollars_per_unit
        self.cpc = cost_per_case
        self.n = len(cases)

    def tour_len(self, seq):
        if not seq: return 0.0
        D, dep = self.D, self.depot
        s = D[dep][seq[0]] + D[seq[-1]][dep]
        for a, b in zip(seq, seq[1:]): s += D[a][b]
        return s

    def replan(self, seq, m):
        if not any(l in m for l in seq): return seq
        D, dep = self.D, self.depot
        cur = [l for l in seq if l not in m]
        for l in seq:
            if l in m:
                x = m[l]
                if x in cur: continue
                best_c, best_i, prev = None, 0, dep
                for i in range(len(cur) + 1):
                    nxt = cur[i] if i < len(cur) else dep
                    c = D[prev][x] + D[x][nxt] - D[prev][nxt]
                    if best_c is None or c < best_c: best_c, best_i = c, i
                    prev = nxt
                cur = cur[:best_i] + [x] + cur[best_i:]
        return cur

    def detour_ranking(self, tours):
        D, dep = self.D, self.depot
        det, vis = defaultdict(float), defaultdict(int)
        for t in tours:
            s = [dep] + list(t) + [dep]
            for i in range(1, len(s) - 1):
                x = D[s[i-1]][s[i]] + D[s[i]][s[i+1]] - D[s[i-1]][s[i+1]]
                det[s[i]] += max(x, 0.0); vis[s[i]] += 1
        order = sorted(det, key=lambda l: -det[l])
        return order, dict(vis), dict(det)

class Planner:
    def __init__(self, W, train, K=5, cap=150):
        self.W, self.train, self.K, self.cap = W, train, K, cap
        self.order_detour, self.vis, self.det = W.detour_ranking(train)
        self.through = defaultdict(list)
        for i, t in enumerate(train):
            for l in set(t): self.through[l].append(i)
        self.dep_dist = [W.D[W.depot][l] for l in range(W.n)]

    def cases(self, l): return self.W.cases[l]

    def quietest_golden(self, hot, moved):
        if hot in self.gset: return []
        c = [l for l in self.W.golden if l not in moved and (os.environ.get('VISITED_ONLY','1') == '0' or l in self.vis)]
        c.sort(key=lambda l: (self.vis.get(l, 0), self.cases(l), self.dep_dist[l]))
        return c

    @property
    def gset(self):
        if not hasattr(self, "_g"): self._g = set(self.W.golden)
        return self._g

    def try_swap(self, hot, partner, moved, cur_len):
        W = self.W
        touched = set(self.through.get(hot, [])) | set(self.through.get(partner, []))
        w = {**moved, hot: partner, partner: hot}
        before = sum(cur_len[i] for i in touched)
        after_lens = {i: W.tour_len(W.replan(self.train[i], w)) for i in touched}
        saved = (before - sum(after_lens.values())) * W.dpu
        cost = W.cpc * (self.cases(hot) + self.cases(partner))
        return saved, cost, w, after_lens

    def greedy(self, order, cost_test=True, n_force=None, partners=None, moved0=None):
        W = self.W
        partners = partners or self.quietest_golden
        moved = dict(moved0 or {}); cur_len = [W.tour_len(W.replan(t, moved)) if moved else W.tour_len(t) for t in self.train]
        n, tcost = 0, 0.0
        limit = self.cap if cost_test else n_force
        for hot in order:
            if n >= limit: break
            if hot in moved: continue
            parts = partners(hot, moved)[: (self.K if cost_test else 1)]
            best = None
            for p in parts:
                saved, cost, w, al = self.try_swap(hot, p, moved, cur_len)
                if cost_test:
                    if saved >= cost * getattr(self, 'margin', 1.0) and (best is None or saved - cost > best[0] - best[1]): best = (saved, cost, w, al)
                else:
                    best = (saved, cost, w, al)
            if best:
                _, cost, moved, al = best
                for i, v in al.items(): cur_len[i] = v
                n += 1; tcost += cost
        return moved, n, tcost

    def cluster_step(self, moved0, loc_cluster, order):
        vis_in = defaultdict(list)
        for l, c in loc_cluster.items(): vis_in[c].append(self.vis.get(l, 0))
        ranked = sorted(vis_in, key=lambda c: -np.mean(vis_in[c]))
        pool = sorted(loc_cluster, key=lambda l: self.dep_dist[l]); block, pos = {}, 0
        for c in ranked:
            block[c] = set(pool[pos:pos + len(vis_in[c])]); pos += len(vis_in[c])
        mis = [l for l in order if l in loc_cluster and l not in block[loc_cluster[l]]]
        def own_block_spots(hot, moved):
            c = loc_cluster[hot]
            return sorted((l for l in block[c] if loc_cluster.get(l) != c and l not in moved), key=lambda l: (self.vis.get(l, 0), self.cases(l)))
        return self.greedy(mis, True, partners=own_block_spots, moved0=moved0)

    def plan(self, policy, n_ref=None):
        if policy == "detour+cost":
            return self.greedy(self.order_detour, True)
        if policy == "velocity+cost":
            order = sorted(self.vis, key=lambda l: -self.vis[l])
            return self.greedy(order, True)
        if policy == "classic_abc":
            order = sorted(self.vis, key=lambda l: -self.vis[l])
            return self.greedy(order, False, n_force=n_ref)
        raise ValueError(policy)

def replay(W, test, moved):
    tot_before = sum(W.tour_len(t) for t in test)
    if not moved: return 0.0, tot_before, 0.0
    saved = 0.0
    for t in test:
        if any(l in moved for l in t):
            saved += W.tour_len(t) - W.tour_len(W.replan(t, moved))
    return saved, tot_before, saved / tot_before if tot_before else 0.0

def move_cost(W, moved):
    seen, c = set(), 0.0
    for a, b in moved.items():
        if a in seen: continue
        seen.add(a); seen.add(b); c += W.cpc * (W.cases[a] + W.cases[b])
    return c
