#!/usr/bin/env python3
"""Cache compact MoGe-3 neck features for offline FastViT distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from generate_moge3_pseudo_depth_dgx import make_triton_dtype_compatible, sha256_file


def feature_path(root: Path, sample_id: str) -> Path:
    return root / "features" / f"{sample_id}.npz"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/celebamask_hq/moge3_base_pilot/manifest.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/moge-3-vitl/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/celebamask_hq/moge3_base_features_v10"))
    parser.add_argument("--expected-images", type=int, default=30000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--feature-size", type=int, default=48)
    parser.add_argument("--projection-seed", type=int, default=173)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.expected_images < 1 or args.limit < 0 or args.channels < 1 or args.feature_size < 8 or args.log_every < 1:
        parser.error("Invalid image count, feature shape, or log interval")
    if not args.manifest.is_file() or not args.checkpoint.is_file():
        parser.error("MoGe manifest or checkpoint is missing")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != args.expected_images:
        raise ValueError(f"Expected {args.expected_images} labeled images; found {len(rows)}")
    teacher_info = json.loads((args.manifest.parent / "teacher_metadata.json").read_text(encoding="utf-8"))
    if teacher_info.get("teacher") != "Ruicheng/moge-3-vitl" or teacher_info.get("refine_steps") != 0:
        raise ValueError("Feature cache must use the same MoGe-3 base teacher as the depth labels")
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != teacher_info.get("checkpoint_sha256"):
        raise ValueError("MoGe checkpoint differs from the depth-label teacher")
    seen = set()
    for row in rows:
        if row["sample_id"] in seen:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        seen.add(row["sample_id"])
        if not Path(row["image"]).is_file() or not Path(row["pseudo_label"]).is_file():
            raise FileNotFoundError(f"Missing image or depth label: {row['sample_id']}")
    metadata = {
        "format": "moge3-neck-highest-resolution-random-projection-v1",
        "teacher": "Ruicheng/moge-3-vitl",
        "teacher_checkpoint_sha256": checkpoint_sha,
        "depth_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "resolution_level": int(teacher_info["resolution_level"]),
        "refine_steps": 0,
        "fp16": bool(teacher_info["fp16"]),
        "channels": args.channels,
        "feature_size": args.feature_size,
        "projection_seed": args.projection_seed,
        "source_images": len(rows),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise RuntimeError("Existing feature cache has a different teacher or projection")
    else:
        atomic_json(metadata_path, metadata)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MoGe feature extraction")
    make_triton_dtype_compatible()
    from moge.model.v3 import MoGeModel

    model = MoGeModel.from_pretrained(str(args.checkpoint.resolve())).to("cuda").eval()
    model.requires_grad_(False)
    captured: list[torch.Tensor] = []

    def capture_neck(_module, _inputs, output):
        tensors = [item for item in output if isinstance(item, torch.Tensor)] if isinstance(output, (list, tuple)) else [output]
        spatial = [item for item in tensors if isinstance(item, torch.Tensor) and item.ndim == 4]
        if not spatial:
            raise RuntimeError("MoGe neck did not produce a spatial feature map")
        captured.append(max(spatial, key=lambda item: item.shape[-2] * item.shape[-1]).detach())

    handle = model.neck.register_forward_hook(capture_neck)
    selected = rows[:args.limit] if args.limit else rows
    start = time.monotonic()
    reused = new = 0
    projector = None
    try:
        for index, row in enumerate(selected, 1):
            destination = feature_path(args.output, row["sample_id"])
            if destination.is_file():
                reused += 1
                continue
            with Image.open(row["image"]) as source:
                rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
            if rgb.shape[0] != rgb.shape[1]:
                raise ValueError(f"V10 feature cache expects square CelebAMask-HQ photos: {row['sample_id']}")
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).to("cuda", dtype=torch.float32).div_(255.0)
            captured.clear()
            with torch.inference_mode():
                model.infer(tensor, resolution_level=metadata["resolution_level"], refine_steps=0, use_fp16=metadata["fp16"])
                if len(captured) != 1:
                    raise RuntimeError(f"Expected one MoGe neck call for {row['sample_id']}; saw {len(captured)}")
                feature = captured.pop().float()
                if projector is None:
                    generator = torch.Generator(device="cpu").manual_seed(args.projection_seed)
                    projector = torch.randn(args.channels, feature.shape[1], generator=generator).to("cuda")
                    projector = F.normalize(projector, p=2, dim=1).view(args.channels, feature.shape[1], 1, 1)
                elif feature.shape[1] != projector.shape[1]:
                    raise RuntimeError("MoGe neck channel count changed within the cache")
                feature = F.interpolate(feature, size=(args.feature_size, args.feature_size), mode="bilinear", align_corners=False)
                compact = F.conv2d(feature, projector)
                compact = F.normalize(compact, p=2, dim=1, eps=1e-6)[0]
            array = compact.half().cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite teacher feature: {row['sample_id']}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            with temporary.open("wb") as target:
                np.savez_compressed(target, feature=array)
            os.replace(temporary, destination)
            new += 1
            if index % args.log_every == 0 or index == len(selected):
                rate = new / max(time.monotonic() - start, 1e-6)
                print(f"Feature {index}/{len(selected)} reused={reused} new={new} {rate:.2f} new images/s", flush=True)
    finally:
        handle.remove()
    print(f"READY: {reused + new}/{len(selected)} cached features in {args.output}", flush=True)


if __name__ == "__main__":
    main()
