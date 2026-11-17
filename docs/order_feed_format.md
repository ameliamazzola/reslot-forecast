# Order feed format

The message a real order feed would send to the simulator's live mode, and how
the engine turns it into the work it simulates. Today only replay is
implemented (`ReplaySource`, from the transactions export); `LiveFeedSource`
in `src/reslot/sim/orders.py` is a stub written against this document.

## Transport

JSON Lines: one message per line, UTF-8, over a socket or an appended file
that the source tails. Messages are processed in the order received.

## The order message (`racksmart-order/1`)

```json
{
  "schema": "racksmart-order/1",
  "type": "order.release",
  "order_id": "8006904399",
  "released_at": "2026-10-06T08:14:27-04:00",
  "customer_id": "WALMART DC 6030A",
  "lines": [
    {"line_no": 1, "item_id": "20511059", "location": "KK119A", "qty": 4, "uom": "CS"},
    {"line_no": 2, "item_id": "20505021", "location": "KK123A", "qty": 1, "uom": "CS"}
  ]
}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `schema` | string | yes | `racksmart-order/1`. Anything else is refused. |
| `type` | string | yes | `order.release`: the WMS has released this order for picking. No other types in v1 (no amendments or cancellations). |
| `order_id` | string | yes | WMS order number, as text (suffixes such as `CR`/`LR` occur). A repeated `order_id` is ignored after the first. |
| `released_at` | ISO 8601 timestamp with UTC offset | yes | When the WMS released the work. Must not be earlier than the previous message's `released_at`; an out-of-order message is refused. |
| `customer_id` | string | no | Kept for reference; the simulator does not use it. |
| `lines` | array | yes, ≥ 1 | The order's pick lines. |
| `lines[].line_no` | integer | yes | Line number within the order. |
| `lines[].item_id` | string | yes | SKU, as text (leading zeros matter). |
| `lines[].location` | string | yes | The WMS location the line is picked from, e.g. `KK119A`. Must map through `location_stop_crosswalk.csv`. |
| `lines[].qty` | number | yes | Quantity in `uom`. |
| `lines[].uom` | string | yes | `CS` (cases). v1 simulates case picks only; other UOMs are dropped and counted. |

The operator is not part of the message: in live mode the engine's operator
pool assigns one.

## How a message becomes simulated work

1. **Validate** the message (schema, required fields, `released_at` order).
2. **Place each line** with the same rules as replay (`sim/workload.drop_reasons`):
   a line is kept only if its location is in the crosswalk with status `ok`, is
   a case-pick face in scope (`WZPCK`, slot type `F`), and its stop is in the
   distance table. Every dropped line is logged with its reason, never guessed.
3. **Build one tour** (`workload.Tour`) per order: `arrival_s` = `released_at`
   in seconds from the run start, the kept lines as `Pick`s in message order,
   `user_id` empty.
4. **Release it** to the engine at `released_at`. Policies see it only from
   then on (the no-lookahead rule; `orders.ArrivedOrders`).

## How replay fills the same fields

Replay (`ReplaySource`, `mode: replay`) builds the same tours from the
transactions export, with two differences that every replay manifest records
under `source`:

- **`released_at` is a proxy.** The export has no release time. A tour is
  released at its **first kept pick scan**: later than the true release by the
  drive to the first pick (and any wait before it).
- **A tour is order × operator,** not an order. An order picked by two
  operators is two tours, as it happened; the observed operator is kept in
  `user_id`.

## Modes and citation

| Mode | Source | Citable |
|---|---|---|
| `replay` | transactions export over a window | yes, from a clean commit (`git_dirty: false`) |
| `live` | generator or live feed, real-time clock | never (`mode: live` in its manifest) |
