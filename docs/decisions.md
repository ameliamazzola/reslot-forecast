# Decisions

Append-only. Each entry: what was decided, why, and what would reverse it.
Data facts live in `data-notes.md`; this file records choices.

---

## 2026-09-14 — WORK_PATH is the travel coordinate

**Decision.** Use the WMS `WORK_PATH` field as a one-dimensional travel
coordinate rather than waiting on x/y/z or CAD-derived geometry.

**Why.** C&D confirmed the WMS does not populate coordinates, and the location
master bears that out. `WORK_PATH` is populated on every row and near-unique
across pick faces. Across 3,118 order-trips of 5+ picks, median |Spearman|
between chronological pick order and `WORK_PATH` is 0.928; 66% of trips exceed
0.8. Pickers traverse in path order.

**Reverses if.** The CAD layout gives metric coordinates, or the analyst says
`WORK_PATH` is maintained per zone rather than building-wide. `WORK_PATH` is
ordinal — it orders slots, it doesn't measure feet. Metric geometry is still
worth having.

---

## 2026-09-14 — Travel model includes quantity and aisle changes

**Decision.** Model pick-to-pick time as
`fixed + per_case*qty + cross_aisle + distance(delta_path)`, not distance alone.

**Why.** Cases picked predicts elapsed time better than path distance does
(Spearman 0.52 vs 0.32). Holdout MAE: constant 81.2s, distance-only 76.0s,
full model 62.7s. Omitting quantity attributes handling time to distance and
inflates the apparent payoff of re-slotting.

**Consequence.** Path travel plus aisle changes are 8.8% of observed case-pick
time. An aisle change costs ~33s; in-aisle distance is nearly free. The policy
lever on the pick face is reducing aisle changes per trip (co-locating SKUs that
ship together), not shortening distance to a golden zone. And 8.8% is the
ceiling any slotting policy can address there.

**Reverses if.** Reserve pallet pulls decompose differently — plausible and
untested. Run the same decomposition there before settling where the policy
aims.

---

## 2026-09-14 — No Chebyshev travel model for C&D data

**Decision.** Chebyshev (simultaneous-axis AS/RS crane) is not used for the C&D
operation.

**Why.** This DC is operators on equipment moving through an aisle network,
not a crane. Travel runs along aisles.

**Reverses if.** A synthetic AS/RS becomes the primary subject again.

---

## 2026-09-14 — Slot inventory comes from the master, not transactions

**Decision.** Build the slot model from `SLOTS` (location master).

**Why.** 1,543 pick faces exist in `WZPCK`; 1,002 saw a case pick in 30 days.
The other 541 are the free capacity re-slotting moves into, and they appear
only in the master.

---

## 2026-09-14 — Trip-level holdout split

**Decision.** Split travel-model train/holdout by trip, never by leg.

**Why.** Legs in one trip share an operator, an hour and an aisle. A leg-level
split leaks correlated observations across the split and flatters the error.
Guarded by `test_split_no_trip_leak`.

---

## 2026-09-14 — Derived travel outputs are not committed

**Decision.** `travel_cost.csv` and `travel_profile.csv` go to
`data/processed/travel/` (gitignored). Each teammate regenerates them from the
Drive extracts.

**Why.** They're derived from C&D data and permissions aren't settled. Same
rule as the rest of `data/processed/`.

**Reverses if.** C&D confirms aggregates are fine to commit. Then commit the
cost table as the fixed sim↔data interface.
