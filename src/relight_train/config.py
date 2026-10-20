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
    for key in ("crop_depth_shape_weight", "depth_gradient_weight", "normal_edge_weight"):
        if float(exp.get(key, 0.0)) < 0:
            raise ValueError(f"{key} must be nonnegative")
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
    early_stopping = exp.get("early_stopping")
    if early_stopping is not None:
        if not isinstance(early_stopping, dict):
            raise ValueError("early_stopping must be a mapping")
        if int(early_stopping.get("patience", 0)) < 1:
            raise ValueError("early_stopping.patience must be positive")
        minimum_epochs = int(early_stopping.get("min_epochs", 1))
        if not 1 <= minimum_epochs <= int(exp["epochs"]):
            raise ValueError("early_stopping.min_epochs must be between 1 and epochs")
        if float(early_stopping.get("min_delta", 0.0)) < 0:
            raise ValueError("early_stopping.min_delta must be nonnegative")
    domain_augmentation = exp.get("domain_augmentation")
    if domain_augmentation is not None:
        if not isinstance(domain_augmentation, dict):
            raise ValueError("domain_augmentation must be a mapping")
        for key in ("background_probability", "crop_probability", "head_crop_fraction", "jpeg_probability"):
            value = float(domain_augmentation.get(key, 0))
            if not 0 <= value <= 1:
                raise ValueError(f"domain_augmentation.{key} must be between 0 and 1")
        jpeg_quality = domain_augmentation.get("jpeg_quality", [75, 95])
        if not isinstance(jpeg_quality, (list, tuple)) or len(jpeg_quality) != 2 or not 1 <= int(jpeg_quality[0]) <= int(jpeg_quality[1]) <= 100:
            raise ValueError("domain_augmentation.jpeg_quality must be [minimum, maximum] within 1-100")
        if int(domain_augmentation.get("minimum_person_height", 64)) < 1:
            raise ValueError("domain_augmentation.minimum_person_height must be positive")
        bands = domain_augmentation.get("crop_zoom_bands")
        if bands is not None:
            if not isinstance(bands, list) or not bands:
                raise ValueError("domain_augmentation.crop_zoom_bands must be a nonempty list")
            for band in bands:
                if not isinstance(band, (list, tuple)) or len(band) != 3:
                    raise ValueError("Each crop_zoom_band must be [minimum_zoom, maximum_zoom, weight]")
                low, high, weight = map(float, band)
                if not 1 < low <= high or weight <= 0:
                    raise ValueError("Crop zoom bands require 1 < minimum <= maximum and positive weight")
            if not 24 <= int(domain_augmentation.get("minimum_crop_height", 96)) <= input_size[0]:
                raise ValueError("minimum_crop_height must be between 24 and input height")
    crop_validation = exp.get("crop_validation")
    if crop_validation is not None:
        if not isinstance(crop_validation, dict):
            raise ValueError("crop_validation must be a mapping")
        zooms = crop_validation.get("zooms", [])
        if not isinstance(zooms, list) or not zooms or any(float(zoom) <= 1 for zoom in zooms):
            raise ValueError("crop_validation.zooms must contain zoom values greater than 1")
        if int(crop_validation.get("samples", 0)) < 1:
            raise ValueError("crop_validation.samples must be positive")
        for key in ("normal_score_weight", "silog_score_weight"):
            if float(crop_validation.get(key, 0)) < 0:
                raise ValueError(f"crop_validation.{key} must be nonnegative")
    data["source_root"] = _expand_path(data["source_root"])
    data["inventory_dir"] = _expand_path(data["inventory_dir"])
    output["run_dir"] = _expand_path(output["run_dir"])
    if exp.get("init_checkpoint"):
        exp["init_checkpoint"] = _expand_path(exp["init_checkpoint"])
    real_data = config.get("real_data")
    if real_data is not None:
        if not isinstance(real_data, dict):
            raise ValueError("real_data must be a mapping")
        for key in ("manifest", "pseudo_depth_root", "validation_manifest", "test_manifest"):
            if key not in real_data:
                if key in ("validation_manifest", "test_manifest"):
                    continue
                raise ValueError(f"real_data.{key} is required")
            real_data[key] = _expand_path(real_data[key])
        if int(real_data.get("batch_size", 0)) < 1:
            raise ValueError("real_data.batch_size must be positive")
        if int(real_data.get("batches_per_synth_batch", 0)) < 1:
            raise ValueError("real_data.batches_per_synth_batch must be positive")
        if float(real_data.get("distillation_weight", -1)) < 0:
            raise ValueError("real_data.distillation_weight must be nonnegative")
        if not 0 <= float(real_data.get("crop_probability", 0.0)) <= 1:
            raise ValueError("real_data.crop_probability must be between zero and one")
        crop_zoom = real_data.get("crop_zoom_range", [1.25, 2.0])
        if (
            not isinstance(crop_zoom, (list, tuple))
            or len(crop_zoom) != 2
            or not 1 < float(crop_zoom[0]) <= float(crop_zoom[1])
        ):
            raise ValueError("real_data.crop_zoom_range must be [minimum, maximum] with 1 < minimum <= maximum")
    return config
