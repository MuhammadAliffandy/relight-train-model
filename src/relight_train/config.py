from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    for section in ("experiment", "data", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing config section: {section}")
    exp, data, output = config["experiment"], config["data"], config["output"]
    input_size = tuple(int(size) for size in exp.get("input_size", ()))
    if len(input_size) != 2 or any(size < 32 or size % 32 for size in input_size):
        raise ValueError("input_size must be [height, width] with both dimensions positive multiples of 32.")
    exp["input_size"] = list(input_size)
    if exp.get("depth_mode") not in ("metric", "relative"):
        raise ValueError("depth_mode must be metric or relative")
    if int(exp.get("batch_size", 0)) < 1 or int(exp.get("epochs", 0)) < 1:
        raise ValueError("batch_size and epochs must be positive")
    data["source_root"] = _expand_path(data["source_root"])
    data["inventory_dir"] = _expand_path(data["inventory_dir"])
    output["run_dir"] = _expand_path(output["run_dir"])
    if exp.get("init_checkpoint"):
        exp["init_checkpoint"] = _expand_path(exp["init_checkpoint"])
    return config
