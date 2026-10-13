from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser().resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    for section in ("experiment", "data", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing config section: {section}")
    exp, data, output = config["experiment"], config["data"], config["output"]
    if tuple(exp.get("input_size", ())) != (256, 256):
        raise ValueError("Prepared shard format currently requires input_size [256, 256].")
    if exp.get("depth_mode") not in ("metric", "relative"):
        raise ValueError("depth_mode must be metric or relative")
    if int(exp.get("batch_size", 0)) < 1 or int(exp.get("epochs", 0)) < 1:
        raise ValueError("batch_size and epochs must be positive")
    data["shard_dir"] = _expand_path(data["shard_dir"])
    data["cache_dir"] = _expand_path(data["cache_dir"])
    output["run_dir"] = _expand_path(output["run_dir"])
    return config
