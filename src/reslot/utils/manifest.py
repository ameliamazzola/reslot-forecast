"""Every run writes a manifest. No number goes in the report without one.

Established week one, not retrofitted: config hash + git SHA + seed + input
file hashes, so any figure can be traced back to the exact code and data that
produced it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _dirty() -> bool:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        return False


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_manifest(out_dir, config: dict, inputs: list | None = None, extra: dict | None = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_blob = json.dumps(config, sort_keys=True, default=str)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "git_dirty": _dirty(),      # True here means the run is NOT citable
        "config_hash": hashlib.sha256(cfg_blob.encode()).hexdigest()[:16],
        "config": config,
        "seed": config.get("seed"),
        "python": platform.python_version(),
        "inputs": [{"path": str(p), "sha256_16": file_hash(p)} for p in (inputs or [])],
        **(extra or {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest
