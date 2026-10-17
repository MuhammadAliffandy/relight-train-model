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
    if exp.get("decoder_refinement", "none") not in ("none", "stride2", "fullres"):
        raise ValueError("decoder_refinement must be none, stride2, or fullres")
    if exp.get("guarded_refinement", False) and exp.get("decoder_refinement") != "fullres":
        raise ValueError("guarded_refinement requires fullres decoder_refinement")
    if float(exp.get("auxiliary_loss_weight", 0.0)) < 0:
        raise ValueError("auxiliary_loss_weight must be nonnegative")
    if float(exp.get("residual_depth_limit", 0.05)) <= 0:
        raise ValueError("residual_depth_limit must be positive")
    if not 0 < float(exp.get("residual_normal_limit", 0.15)) < 1:
        raise ValueError("residual_normal_limit must be between 0 and 1")
    if int(exp.get("refinement_width", 16)) < 1:
        raise ValueError("refinement_width must be positive")
    if int(exp.get("fullres_width", 24)) < 1 or int(exp.get("rgb_full_width", 8)) < 1:
        raise ValueError("fullres_width and rgb_full_width must be positive")
    if exp.get("init_mode", "full") not in ("full", "encoder_only"):
        raise ValueError("init_mode must be full or encoder_only")
    if int(exp.get("batch_size", 0)) < 1 or int(exp.get("epochs", 0)) < 1:
        raise ValueError("batch_size and epochs must be positive")
    data["source_root"] = _expand_path(data["source_root"])
    data["inventory_dir"] = _expand_path(data["inventory_dir"])
    output["run_dir"] = _expand_path(output["run_dir"])
    if exp.get("init_checkpoint"):
        exp["init_checkpoint"] = _expand_path(exp["init_checkpoint"])
    return config
