"""
Slotting policies the simulator runs.

A policy answers one question during a run: the product the WMS picked from
`location` -- where does it sit at sim time t? And it reports every
relocation move it made, so the engine can charge the forklift time for it.
A policy that re-slots for free is not a fair comparison (docs/decisions.md
2026-11-02, static policy entry).

v1 has one policy:

  static   the location master's slotting as-is, no moves (current practice;
           the reference every other policy is compared against).

one_shot (start-of-week swaps) and reactive ABC come later and must report
their moves through moves().
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Move:
    """One relocation. The engine prices it; the policy only records it."""
    t: float                  # sim seconds when the move happens
    from_location: str
    to_location: str
    cases: float              # cases moved (move cost scales with it)


class Policy:
    name = "base"

    def location_for(self, location: str, t: float) -> str:
        """Where the product picked from `location` sits at time t."""
        raise NotImplementedError

    def moves(self) -> list[Move]:
        """Every relocation made so far, in time order."""
        raise NotImplementedError

    def route(self, locations: tuple, t: float) -> list[int]:
        """Order in which a tour's picks are visited, as positions into
        `locations` (the picks' observed locations, in scan order). Must be a
        permutation. Default: the observed order."""
        return list(range(len(locations)))


class StaticPolicy(Policy):
    """No re-slots: every product stays where the WMS picked it from."""
    name = "static"

    def location_for(self, location: str, t: float) -> str:
        return location

    def moves(self) -> list[Move]:
        return []


POLICIES = {StaticPolicy.name: StaticPolicy}


def make_policy(name: str) -> Policy:
    try:
        return POLICIES[name]()
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; available: {sorted(POLICIES)}") from None
