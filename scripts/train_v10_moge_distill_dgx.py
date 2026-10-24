#!/usr/bin/env python3
"""Train a fresh FastViT depth student against MoGe-3 depth and cached features."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from relight_train.model import PortraitDepthStudent
from train_moge3_depth_dgx import MoGeDepthDataset, depth_loss, load_manifest, save_checkpoint


class StudentDataset(MoGeDepthDataset):
    def __init__(self, rows, height, width, feature_root: Path | None):
        super().__init__(rows, height, width)
        self.feature_root = feature_root

    def __getitem__(self, index):
        rgb, depth, valid = super().__getitem__(index)
        if self.feature_root is None:
            return rgb, depth, valid
        row = self.rows[index]
        path = self.feature_root / "features" / f"{row['sample_id']}.npz"
        with np.load(path, allow_pickle=False) as source:
            feature = torch.from_numpy(source["feature"].astype(np.float32))
        if feature.ndim != 3 or not torch.isfinite(feature).all():
            raise ValueError(f"Invalid cached teacher feature: {path}")
        with Image.open(row["image"]) as image:
            original_width, original_height = image.size
        scale = min(self.width / original_width, self.height / original_height)
        fitted_width = min(self.width, max(1, round(original_width * scale)))
        fitted_height = min(self.height, max(1, round(original_height * scale)))
        grid_height, grid_width = self.height // 16, self.width // 16
        content_width = max(1, min(grid_width, round(fitted_width / 16)))
        content_height = max(1, min(grid_height, round(fitted_height / 16)))
        left = (grid_width - content_width) // 2
        top = (grid_height - content_height) // 2
        aligned = torch.zeros((feature.shape[0], grid_height, grid_width), dtype=torch.float32)
        aligned[:, top:top + content_height, left:left + content_width] = F.interpolate(
            feature.unsqueeze(0), size=(content_height, content_width), mode="bilinear", align_corners=False
        )[0]
        feature_mask = torch.zeros((1, grid_height, grid_width), dtype=torch.bool)
        feature_mask[:, top:top + content_height, left:left + content_width] = True
        return rgb, depth, valid, aligned, feature_mask


def feature_loss(student_feature, teacher_feature, feature_mask, depth_valid, adapter):
    student = F.interpolate(student_feature.float(), size=teacher_feature.shape[-2:], mode="area")
    student = F.normalize(adapter(student), dim=1, eps=1e-6)
    teacher = F.normalize(teacher_feature.float(), dim=1, eps=1e-6)
    valid = feature_mask.bool() & (F.interpolate(depth_valid.float(), size=teacher.shape[-2:], mode="area") > 0.9)
    if not valid.any():
        raise ValueError("No valid pixels for feature distillation")
    cosine = (student * teacher).sum(dim=1, keepdim=True)
    return (1 - cosine)[valid].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/celebamask_hq/moge3_base_pilot/manifest.jsonl"))
    parser.add_argument("--feature-root", type=Path, default=Path("data/celebamask_hq/moge3_base_features_v10"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=30000)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--snapshot-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-lr", type=float, default=3e-6)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--feature-weight", type=float, default=0.02)
    parser.add_argument("--feature-warmup-epochs", type=int, default=5)
    parser.add_argument("--gradient-weight", type=float, default=0.2)
    parser.add_argument("--metric-weight", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    args = parser.parse_args()
    if (args.height % 32 or args.width % 32 or min(args.height, args.width) < 32 or
            min(args.expected_images, args.batch_size, args.epochs, args.log_every,
                args.snapshot_interval, args.feature_warmup_epochs) < 1 or
            min(args.workers, args.max_train_samples, args.max_validation_samples) < 0 or
            min(args.encoder_lr, args.head_lr) <= 0 or
            min(args.feature_weight, args.gradient_weight, args.metric_weight) < 0):
        parser.error("Invalid image size, count, learning rate, or loss weight")
    if not torch.cuda.is_available():
        raise RuntimeError("V10 DGX training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    splits, manifest_sha, teacher_info = load_manifest(args.manifest, args.expected_images)
    train_rows = splits["train"][:args.max_train_samples] if args.max_train_samples else splits["train"]
    val_rows = splits["validation"][:args.max_validation_samples] if args.max_validation_samples else splits["validation"]
    feature_root = args.feature_root if args.feature_weight > 0 else None
    feature_info = None
    if feature_root is not None:
        feature_info = json.loads((feature_root / "metadata.json").read_text(encoding="utf-8"))
        if (feature_info.get("format") != "moge3-neck-highest-resolution-random-projection-v1" or
                feature_info.get("teacher_checkpoint_sha256") != teacher_info["checkpoint_sha256"] or
                feature_info.get("depth_manifest_sha256") != manifest_sha or
                feature_info.get("refine_steps") != 0 or
                feature_info.get("resolution_level") != teacher_info["resolution_level"] or
                feature_info.get("fp16") != teacher_info["fp16"]):
            raise ValueError("Cached features do not match the MoGe depth teacher and manifest")
        missing = [row["sample_id"] for row in train_rows
                   if not (feature_root / "features" / f"{row['sample_id']}.npz").is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} teacher features missing; examples: {missing[:3]}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.run_dir / "last_checkpoint.pt"
    metadata = {
        "manifest_sha256": manifest_sha,
        "teacher_checkpoint_sha256": teacher_info["checkpoint_sha256"],
        "feature_metadata_sha256": hashlib.sha256((feature_root / "metadata.json").read_bytes()).hexdigest() if feature_root else None,
        "init": "timm fastvit_t8.apple_in1k ImageNet pretrained; fresh FPN and depth decoder; no V9 checkpoint",
        "input_size": [args.height, args.width],
        "preprocessing": "fit_pad_gray_0.5; raw RGB 0..1 at API, ImageNet normalization inside encoder",
        "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
        "feature_weight": args.feature_weight, "feature_warmup_epochs": args.feature_warmup_epochs,
        "gradient_weight": args.gradient_weight,
        "metric_weight": args.metric_weight, "train_count": len(train_rows),
        "validation_count": len(val_rows), "held_out_test_count": len(splits["test"]),
    }
    metadata_path = args.run_dir / "training_metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise RuntimeError("Existing run has different training settings")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    model = PortraitDepthStudent(pretrained=not last_path.exists(), fpn_width=72, fpn_fusion="concat_all").to("cuda")
    adapter = nn.Conv2d(72, int(feature_info["channels"]), 1).to("cuda") if feature_info else None
    groups = [
        {"params": model.encoder.parameters(), "lr": args.encoder_lr},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("encoder.")], "lr": args.head_lr},
    ]
    if adapter is not None:
        groups.append({"params": adapter.parameters(), "lr": args.head_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    best = float("inf")
    start_epoch = 1
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("manifest_sha256") != manifest_sha or "optimizer" not in saved:
            raise RuntimeError("Incompatible V10 resume checkpoint")
        model.load_state_dict(saved["model"], strict=True)
        if adapter is not None:
            adapter.load_state_dict(saved["distillation_adapter"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        best = float(saved["best_validation_loss"])
        print(f"Resuming after epoch {start_epoch - 1}; best={best:.5f}", flush=True)
    options = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": True}
    if args.workers:
        options["persistent_workers"] = True
    train_generator = torch.Generator()
    train_loader = DataLoader(StudentDataset(train_rows, args.height, args.width, feature_root),
                              shuffle=True, generator=train_generator, **options)
    val_loader = DataLoader(StudentDataset(val_rows, args.height, args.width, None), shuffle=False, **options)
    experiment = {
        "name": args.run_dir.name,
        "model_type": "portrait_depth_student",
        "backbone": "fastvit_t8.apple_in1k",
        "fpn_width": 72, "fpn_fusion": "concat_all", "detail_width": 24,
        "input_size": [args.height, args.width], "preprocessing": "fit_pad_gray_0.5",
        "depth_training_target": "MoGe-3 base metric pseudo-depth + neck feature distillation" if adapter else "MoGe-3 base metric pseudo-depth",
        "teacher_manifest_sha256": manifest_sha,
        "teacher_feature_weight": args.feature_weight,
        "normal_training": False,
    }
    print(f"V10 student: train={len(train_rows)} val={len(val_rows)} test={len(splits['test'])} "
          f"epochs={args.epochs} feature_weight={args.feature_weight} start={start_epoch}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.monotonic()
        train_generator.manual_seed(args.seed + epoch)
        model.train()
        if adapter is not None:
            adapter.train()
        totals = []
        for step, batch in enumerate(train_loader, 1):
            rgb, depth, valid = (item.to("cuda", non_blocking=True) for item in batch[:3])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted, coarse, fused = model(rgb, return_feature=True)
            main, shape, metric, gradient = depth_loss(predicted, depth, valid, args.gradient_weight, args.metric_weight)
            auxiliary, _, _, _ = depth_loss(coarse, depth, valid, args.gradient_weight, args.metric_weight)
            kd = predicted.new_zeros(())
            if adapter is not None:
                teacher_feature, feature_mask = (item.to("cuda", non_blocking=True) for item in batch[3:])
                kd = feature_loss(fused, teacher_feature, feature_mask, valid, adapter)
            kd_weight = args.feature_weight * min(1.0, epoch / args.feature_warmup_epochs)
            loss = main + 0.3 * auxiliary + kd_weight * kd
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}")
            loss.backward()
            optimizer.step()
            totals.append(float(loss.detach()))
            if step == 1 or step % args.log_every == 0 or step == len(train_loader):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                      f"loss={np.mean(totals):.5f} shape={float(shape):.5f} metric={float(metric):.5f} "
                      f"gradient={float(gradient):.5f} feature={float(kd):.5f} kd_weight={kd_weight:.3f} "
                      f"{step * args.batch_size / elapsed:.1f} images/s", flush=True)
        model.eval()
        scores = []
        with torch.inference_mode():
            for rgb, depth, valid in val_loader:
                rgb, depth, valid = (item.to("cuda", non_blocking=True) for item in (rgb, depth, valid))
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predicted = model(rgb)
                score, _, _, _ = depth_loss(predicted, depth, valid, args.gradient_weight, args.metric_weight)
                scores.append(float(score))
        score = float(np.mean(scores))
        if not np.isfinite(score):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}")
        is_best = score < best
        best = min(best, score)
        payload = {
            "model": model.state_dict(),
            "distillation_adapter": adapter.state_dict() if adapter else None,
            "optimizer": optimizer.state_dict(), "experiment": experiment,
            "epoch": epoch, "validation_loss": score,
            "best_validation_loss": best, "manifest_sha256": manifest_sha,
        }
        if is_best:
            save_checkpoint(args.run_dir / "best_checkpoint.pt", payload)
            save_checkpoint(args.run_dir / "best_deploy.pt", {
                "model": model.state_dict(), "experiment": experiment,
                "epoch": epoch, "validation_loss": score,
            })
        if epoch % args.snapshot_interval == 0:
            save_checkpoint(args.run_dir / f"checkpoint_epoch_{epoch:03d}.pt", payload)
        save_checkpoint(last_path, payload)
        minutes = (time.monotonic() - started) / 60
        with (args.run_dir / "history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps({"epoch": epoch, "train_loss": float(np.mean(totals)),
                                      "validation_loss": score, "best_validation_loss": best,
                                      "minutes": minutes}) + "\n")
        print(f"Epoch {epoch}: train={np.mean(totals):.5f} val={score:.5f} "
              f"best={best:.5f} minutes={minutes:.1f}", flush=True)
    print(f"READY: best={best:.5f}; checkpoint={args.run_dir / 'best_checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
