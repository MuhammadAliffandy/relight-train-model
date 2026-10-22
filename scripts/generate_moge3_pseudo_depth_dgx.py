#!/usr/bin/env python3
"""Generate resumable MoGe-3 depth pseudo-labels for real portrait photos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TEACHER = "Ruicheng/moge-3-vitl"


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = max(0, round(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_groups(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    groups: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id, group_id = str(row["sample_id"]), str(row["group_id"])
            if sample_id in groups:
                raise ValueError(f"Duplicate sample_id in group manifest: {sample_id}")
            groups[sample_id] = group_id
    return groups


def split_for(group_id: str, seed: int) -> str:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{group_id}".encode()).digest()[:8], "big") % 1000
    return "train" if value < 900 else "validation" if value < 950 else "test"


def discover_images(root: Path, groups: dict[str, str] | None, seed: int) -> list[dict]:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    rows = []
    for path in paths:
        sample_id = path.relative_to(root).as_posix()
        if groups is not None and sample_id not in groups:
            raise ValueError(f"Missing group_id for {sample_id}")
        group_id = groups[sample_id] if groups is not None else sample_id
        stat = path.stat()
        rows.append({
            "sample_id": sample_id,
            "image": str(path.resolve()),
            "group_id": group_id,
            "split": split_for(group_id, seed),
            "source_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
        })
    return rows


def label_path(output: Path, sample_id: str) -> Path:
    return output / "labels" / f"{sample_id}.npz"


def label_record(row: dict, path: Path) -> dict:
    return {
        "sample_id": row["sample_id"],
        "image": row["image"],
        "group_id": row["group_id"],
        "split": row["split"],
        "pseudo_label": str(path.resolve()),
        "label_source": TEACHER,
        "depth_key": "depth_m",
        "valid_key": "valid_mask",
        "depth_units": "model-estimated metres",
    }


def infer_label(model, image_path: Path, resolution_level: int, refine_steps: int, fp16: bool):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).to("cuda", dtype=torch.float32).div_(255.0)
    with torch.inference_mode():
        result = model.infer(
            tensor,
            resolution_level=resolution_level,
            refine_steps=refine_steps,
            use_fp16=fp16,
        )
    depth = result["depth"].float().cpu().numpy().astype(np.float32, copy=False)
    valid = result["mask"].bool().cpu().numpy()
    intrinsics = result["intrinsics"].float().cpu().numpy().astype(np.float32, copy=False)
    if depth.shape != rgb.shape[:2] or valid.shape != depth.shape or intrinsics.shape != (3, 3):
        raise ValueError(f"Unexpected MoGe output shape for {image_path}: {depth.shape}, {valid.shape}, {intrinsics.shape}")
    valid &= np.isfinite(depth) & (depth > 0)
    if valid.mean() < 0.01 or not np.isfinite(intrinsics).all():
        raise ValueError(f"Invalid MoGe depth or intrinsics for {image_path}")
    if float(np.std(depth[valid])) < 1e-6:
        raise ValueError(f"Nearly constant MoGe depth for {image_path}")
    depth[~valid] = 0.0
    return depth, valid.astype(np.uint8), intrinsics


def save_label(path: Path, depth: np.ndarray, valid: np.ndarray, intrinsics: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as target:
            np.savez_compressed(target, depth_m=depth, valid_mask=valid, intrinsics=intrinsics)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/celebamask_hq/images"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/moge-3-vitl/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/celebamask_hq/moge3_pseudo"))
    parser.add_argument("--groups", type=Path, help="Optional JSONL: sample_id (relative image path) and group_id (identity/session)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-images", type=int, default=30000)
    parser.add_argument("--limit", type=int, default=0, help="Process the first N images; 0 means all")
    parser.add_argument("--resolution-level", type=int, default=9, choices=range(10))
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--fp16", action="store_true", help="Faster mixed-precision inference; off by default")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()
    if args.expected_images < 0 or args.limit < 0 or args.refine_steps < 0 or args.log_every < 1 or args.max_errors < 1:
        parser.error("expected-images, limit, refine-steps must be nonnegative; log-every and max-errors positive")
    if not args.images.is_dir() or not args.checkpoint.is_file():
        parser.error(f"Missing images directory or checkpoint: {args.images}, {args.checkpoint}")
    if args.groups is not None and not args.groups.is_file():
        parser.error(f"Missing groups manifest: {args.groups}")
    if not torch.cuda.is_available():
        raise RuntimeError("MoGe-3 pseudo-label generation requires a CUDA GPU")
    try:
        from moge.model.v3 import MoGeModel
    except ImportError as error:
        raise RuntimeError("MoGe-3 is not installed in this Python environment") from error

    rows = discover_images(args.images, load_groups(args.groups), args.seed)
    if args.expected_images and len(rows) != args.expected_images:
        raise RuntimeError(f"Expected {args.expected_images} images; found {len(rows)} under {args.images}")
    args.output.mkdir(parents=True, exist_ok=True)
    source_digest = hashlib.sha256("\n".join(json.dumps(row, sort_keys=True) for row in rows).encode()).hexdigest()
    config = {
        "teacher": TEACHER,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_manifest_sha256": source_digest,
        "source_images": len(rows),
        "groups_manifest": str(args.groups.resolve()) if args.groups else None,
        "split_scope": "group_id" if args.groups else "sample_id; identities may overlap across splits",
        "seed": args.seed,
        "resolution_level": args.resolution_level,
        "refine_steps": args.refine_steps,
        "fp16": args.fp16,
        "label_format": "NPZ: depth_m float32, valid_mask uint8, intrinsics float32 normalized 3x3",
        "depth_semantics": "MoGe-3 estimated metric depth; smaller positive values are closer; pseudo-label, not measured ground truth",
        "invalid_depth": "zero with valid_mask=0; excluded from all losses",
    }
    config_path = args.output / "teacher_metadata.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config:
            raise RuntimeError(f"Output contains labels from a different source or teacher configuration: {config_path}")
    else:
        write_json_atomic(config_path, config)
    write_jsonl_atomic(args.output / "source_manifest.jsonl", rows)

    selected = rows[:args.limit] if args.limit else rows
    existing = sum(label_path(args.output, row["sample_id"]).is_file() for row in selected)
    print(f"MoGe-3 teacher on {torch.cuda.get_device_name(0)} | images={len(rows)} selected={len(selected)} existing={existing}", flush=True)
    if args.groups is None:
        print("WARNING: no identity groups supplied; train/validation/test split is sample-level only.", flush=True)
    model = MoGeModel.from_pretrained(str(args.checkpoint.resolve())).to("cuda").eval()
    model.requires_grad_(False)
    print("Teacher loaded; generating labels.", flush=True)

    started = time.monotonic()
    last_report = started
    new = errors = 0
    with (args.output / "errors.jsonl").open("a", encoding="utf-8") as error_log:
        for index, row in enumerate(selected, 1):
            destination = label_path(args.output, row["sample_id"])
            if not destination.is_file():
                try:
                    depth, valid, intrinsics = infer_label(
                        model, Path(row["image"]), args.resolution_level, args.refine_steps, args.fp16
                    )
                    save_label(destination, depth, valid, intrinsics)
                    new += 1
                except (OSError, ValueError) as error:
                    errors += 1
                    record = {"sample_id": row["sample_id"], "error": repr(error)}
                    error_log.write(json.dumps(record) + "\n")
                    error_log.flush()
                    print(f"ERROR {row['sample_id']}: {error}", flush=True)
                    if errors >= args.max_errors:
                        print(f"Stopping after {errors} errors. Rerun to resume completed labels.", flush=True)
                        break
            now = time.monotonic()
            if index == len(selected) or index % args.log_every == 0 or now - last_report >= 30:
                rate = new / max(now - started, 1e-6)
                remaining = max(0, len(selected) - existing - new)
                print(
                    f"Label {index}/{len(selected)} ({index / len(selected):.1%}) | reused={existing} "
                    f"new={new} errors={errors} | {rate:.2f} new images/s | ETA {duration(remaining / rate if rate else None)}",
                    flush=True,
                )
                last_report = now

    completed = [label_record(row, label_path(args.output, row["sample_id"])) for row in selected
                 if label_path(args.output, row["sample_id"]).is_file()]
    write_jsonl_atomic(args.output / "manifest.jsonl", completed)
    print(f"READY: {len(completed)}/{len(selected)} labels; manifest={args.output / 'manifest.jsonl'}; errors={errors}", flush=True)
    if len(completed) != len(selected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
