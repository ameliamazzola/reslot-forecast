# RackSmart

Re-slotting algorithm: picks which items to swap and where, then keeps only swaps that pay for themselves.

```python
from reslot.racksmart import plan
swaps = plan(tours, D, depot, cases, golden, dollars_per_unit, cost_per_case, groups=None, loc_cluster=None, refill=None, window_months=1)
```

`swaps` is `{location: location}`: each pair swaps contents, in the order accepted, cut to 100 swaps (`SETTINGS["cap"]`).

Inputs are described at the top of `racksmart.py`. Final tuned settings are in `SETTINGS` (same values as `settings.json`): look-back 1 month, K 30, margin 2.0, 2 passes, clusters weight 3.0, refill term on, cap 100.

How it works: marginal detour picks WHO moves; co-pick distance (plus cluster-mates) picks WHERE; a swap is kept only if walking saved is at least 2.0 x its cost (move cost + refill change); two passes; swaps stay inside `groups` (aisle fence, pallet height).

Needs Python 3 and numpy.
