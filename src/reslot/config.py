"""YAML config loading. Policy switching and seeds live in configs/, never in code."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if "seed" not in cfg:
        raise ValueError("config must set a seed -- reproducibility is not optional")
    return cfg
