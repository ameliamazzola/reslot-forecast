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

---

## 2026-09-28 — Metric travel network from the layout drawing

**Decision.** Build travel geometry from the York DC layout drawing: forklift
paths and stop points drawn on dedicated layers, turned into a graph and a
stop-to-stop distance table in feet (`travel/network.py`). This is the
"reverses if" of the WORK_PATH entry above, for geometry: WORK_PATH stays the
calibration history until the fit is rerun on network feet.

**Why.** WORK_PATH is ordinal. It can't say how far two slots are apart, so it
can't price a re-slot or tell a golden zone from a far one.

**How it's built.** Lines connect only where they share an endpoint or a stop
circle; crossings are not auto-joined, so every turn is drawn explicitly.
Hand-traced routes match the table (placeholder→FF bays 1–4 133.7 ft, →FF
charger end 705.3 ft, FF→GG across either cross-aisle 41.1 ft).

**Reverses if.** A site walk finds a drawn path that trucks can't use, or an
opening that isn't drawn.

---

## 2026-09-28 — Location → stop rule (crosswalk)

**Decision.** Map each WMS location to a drawn stop by rule
(`travel/crosswalk.py`, anchors in `configs/travel.yaml`): aisle code → path x
(FF rightmost), bay 1 at the dock, 4 bays per stop (one pallet per bay, odd
east / even west), each third of the aisle anchored to its own drawn y.
Tunnel bays 79–84 and 185–190 (upper levels only) map to the cross-aisle
junction under them. Level doesn't change the stop.

**Why.** No coordinates in the WMS. Evidence the rule holds: all 30 drawn
column rack sections land exactly on bay numbers the WMS skips; no stop gets
more than 4 floor locations; floor spot-check by Anthony agreed.

**Scope.** Case picking on FF–QQ. 97.1% of CS picks in the 30-day report map to
a stop (64,763 / 66,699); the rest are RR, ZZ, LP, W — not drawn. Reserve
pallet pulls (single-letter aisles, double-deep) are out of scope.

**Reverses if.** A spot-check disagrees, or the analyst's location-code
definitions contradict the side/level reading.

---

## 2026-09-28 — Trip start/drop-off is a placeholder

**Decision.** One placeholder node (`PD_Placeholder` layer) on a 35 ft spur
off the top cross-aisle, between LL and KK, until staging-destination data
arrives. `network.pd_mode: placeholder` is written to every manifest.

**Why.** `TO_LOCATION` on case picks is the picker's equipment (V-xxxx), not a
floor location, so the transactions can't say where pallets go.

**Consequence.** Pick-to-pick legs are unaffected. Trip totals that include the
first/last leg are not citable until `pd_mode: real`. Run a sensitivity check
across candidate drop-off points before reporting any policy ranking.
