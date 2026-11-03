"""
Read-only access to the travel network outputs for the simulator.

scripts/build_travel_network.py turns the layout drawing into two files in
data/processed/travel/ (network.distance_matrix_ft, crosswalk.build_crosswalk):

    stop_distance_ft.csv          stop x stop shortest-path feet
    location_stop_crosswalk.csv   WMS location -> stop_id, aisle, status, ...

This module loads them, checks they fit together, and answers lookups. It does
no distance math: feet(a, b) is a table lookup, and an unknown stop raises
rather than returning a guess. Rebuilding from the DXF is not an option here
-- it takes minutes, and the drawing is the build script's job.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DISTANCE_FILE = "stop_distance_ft.csv"
CROSSWALK_FILE = "location_stop_crosswalk.csv"
CROSSWALK_COLUMNS = ("location", "aisle", "zone", "slot_type", "status", "stop_id")


class ArtifactError(ValueError):
    pass


def load_distance_table(path: str | Path) -> pd.DataFrame:
    """stop_distance_ft.csv as written by distance_matrix_ft: square, same
    stops on both axes in the same order, zero diagonal, no gaps."""
    D = pd.read_csv(path, index_col=0)
    D.index = D.index.astype(str)
    D.columns = D.columns.astype(str)
    if list(D.index) != list(D.columns):
        raise ArtifactError(f"{path}: rows and columns are not the same stops in the same order")
    if D.index.duplicated().any():
        raise ArtifactError(f"{path}: duplicate stop ids")
    M = D.to_numpy(float)
    if np.isnan(M).any():
        raise ArtifactError(f"{path}: missing distances (disconnected network?)")
    if not np.allclose(np.diag(M), 0.0):
        raise ArtifactError(f"{path}: non-zero diagonal")
    return D


def load_crosswalk(path: str | Path) -> pd.DataFrame:
    """location_stop_crosswalk.csv as written by build_crosswalk. Read as
    strings: location codes and levels must never be parsed as numbers."""
    xw = pd.read_csv(path, dtype=str)
    missing = [c for c in CROSSWALK_COLUMNS if c not in xw.columns]
    if missing:
        raise ArtifactError(f"{path}: missing columns {missing}")
    dupes = int(xw["location"].duplicated().sum())
    if dupes:
        raise ArtifactError(f"{path}: {dupes} duplicate locations")
    return xw


class TravelNetwork:
    """The distance table and crosswalk together, checked against each other."""

    def __init__(self, distances: pd.DataFrame, crosswalk: pd.DataFrame,
                 sources: dict | None = None):
        self.distances = distances
        self._pos = {s: i for i, s in enumerate(distances.index)}
        self._M = distances.to_numpy(float)
        xw = crosswalk.set_index("location", drop=False)
        xw["in_distance_table"] = xw["stop_id"].isin(self._pos)
        self.crosswalk = xw
        self.sources = dict(sources or {})

    # -- lookups ----------------------------------------------------------
    def has_stop(self, stop_id: str) -> bool:
        return stop_id in self._pos

    def feet(self, a: str, b: str) -> float:
        """Shortest-path feet between two stops, from the table. KeyError on
        an unknown stop."""
        try:
            return float(self._M[self._pos[a], self._pos[b]])
        except KeyError as exc:
            raise KeyError(f"stop {exc.args[0]!r} is not in the distance table") from None

    def usable_faces(self, zone: str, slot_type: str) -> pd.DataFrame:
        """Locations a pick can be placed on: crosswalk status ok, a pick face
        in scope, and a stop the distance table knows. Indexed by location,
        columns stop_id and aisle."""
        x = self.crosswalk
        ok = ((x["status"] == "ok") & (x["zone"] == zone) & (x["slot_type"] == slot_type)
              & x["in_distance_table"])
        return x.loc[ok, ["stop_id", "aisle"]]

    # -- checks -----------------------------------------------------------
    def require_stops(self, *stop_ids: str) -> None:
        """Fail before a run, not inside it, if a configured stop (dispatch)
        is not on the network."""
        missing = [s for s in stop_ids if s not in self._pos]
        if missing:
            raise ArtifactError(f"stops not in the distance table: {missing}")

    def summary(self) -> dict:
        """What the manifest records about the network it ran on."""
        x = self.crosswalk
        has_stop = x["stop_id"].notna()
        orphan = has_stop & ~x["in_distance_table"]
        return {
            **self.sources,
            "n_stops": int(len(self._pos)),
            "n_locations": int(len(x)),
            "crosswalk_status": {str(k): int(v) for k, v in x["status"].value_counts().items()},
            "crosswalk_stops_not_in_distance_table": int(orphan.sum()),
            "examples_not_in_distance_table": x.loc[orphan, "location"].head(10).tolist(),
        }


def load_travel(directory: str | Path) -> TravelNetwork:
    """Load both files from the build script's output directory."""
    d = Path(directory)
    dist_path, xw_path = d / DISTANCE_FILE, d / CROSSWALK_FILE
    for p in (dist_path, xw_path):
        if not p.exists():
            raise ArtifactError(f"{p} not found: run scripts/build_travel_network.py first")
    return TravelNetwork(load_distance_table(dist_path), load_crosswalk(xw_path),
                         sources={"distance_table": str(dist_path), "crosswalk": str(xw_path)})
