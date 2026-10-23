"""Fine-tune V9.4 depth with MoGe-3 pseudo-labels on DGX."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from relight_train.model import PortraitGeometryModel


class MoGeDepthDataset(Dataset):
    def __init__(self, rows: list[dict], size: int):
        self.rows = rows
        self.size = size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["image"]) as source:
            image = source.convert("RGB")
            original_size = image.size
            rgb = np.asarray(
                image.resize((self.size, self.size), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ).copy() / 255.0
        with np.load(row["pseudo_label"], allow_pickle=False) as label:
            depth = label["depth_m"].astype(np.float32, copy=False)
            valid = label["valid_mask"].astype(bool, copy=False)
        if depth.shape != original_size[::-1] or valid.shape != depth.shape:
            raise ValueError(f"Image/depth shape mismatch: {row['sample_id']}")
        valid = valid & np.isfinite(depth) & (depth > 0)
        weighted = np.asarray(
            Image.fromarray(np.where(valid, depth, 0).astype(np.float32)).resize(
                (self.size, self.size), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        coverage = np.asarray(
            Image.fromarray(valid.astype(np.float32)).resize(
                (self.size, self.size), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        resized_valid = coverage > 0.99
        resized_depth = np.divide(weighted, np.maximum(coverage, 1e-6))
        resized_depth[~resized_valid] = 0
        if np.count_nonzero(resized_valid) < 64:
            raise ValueError(f"Too few valid depth pixels: {row['sample_id']}")
        return (
            torch.from_numpy(rgb).permute(2, 0, 1),
            torch.from_numpy(resized_depth.copy())[None],
            torch.from_numpy(resized_valid.copy())[None],
        )


def load_manifest(path: Path, expected_images: int):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_images:
        raise ValueError(f"Expected {expected_images} MoGe labels, found {len(rows)} in {path}")
    teacher = json.loads((path.parent / "teacher_metadata.json").read_text(encoding="utf-8"))
    if teacher.get("teacher") != "Ruicheng/moge-3-vitl" or teacher.get("refine_steps") != 0:
        raise ValueError("Manifest must contain MoGe-3 ViT-L labels with refine_steps=0")
    if teacher.get("source_images") != expected_images:
        raise ValueError("Teacher source count does not match the requested label count")
    seen = set()
    splits = {"train": [], "validation": [], "test": []}
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if row.get("label_source") != teacher["teacher"] or row.get("split") not in splits:
            raise ValueError(f"Unexpected teacher or split for {sample_id}")
        if not Path(row["image"]).is_file() or not Path(row["pseudo_label"]).is_file():
            raise FileNotFoundError(f"Missing image or MoGe label: {sample_id}")
        splits[row["split"]].append(row)
    if not splits["train"] or not splits["validation"] or not splits["test"]:
        raise ValueError("Manifest must have nonempty train, validation, and test splits")
    return splits, hashlib.sha256(path.read_bytes()).hexdigest(), teacher


def depth_loss(predicted, teacher, mask, gradient_weight: float, metric_weight: float):
    predicted_log = predicted.float().clamp_min(1e-5).log()
    teacher_log = teacher.float().clamp_min(1e-5).log()
    shape_terms = []
    metric_terms = []
    gradient_terms = []
    for index in range(predicted.shape[0]):
        valid = mask[index, 0].bool() & torch.isfinite(teacher_log[index, 0])
        if int(valid.sum()) < 64:
            raise ValueError("A training image has too few valid MoGe depth pixels")
        p = predicted_log[index, 0]
        t = teacher_log[index, 0]
        offset = (t[valid].median() - p[valid].median()).detach()
        shape_terms.append(F.smooth_l1_loss((p + offset)[valid], t[valid], beta=0.1))
        metric_terms.append(F.smooth_l1_loss(p[valid].mean(), t[valid].mean(), beta=0.1))
        valid_x = valid[:, 1:] & valid[:, :-1]
        valid_y = valid[1:, :] & valid[:-1, :]
        gradient_x = F.smooth_l1_loss(
            (p[:, 1:] - p[:, :-1])[valid_x],
            (t[:, 1:] - t[:, :-1])[valid_x],
            beta=0.05,
        ) if valid_x.any() else p.new_zeros(())
        gradient_y = F.smooth_l1_loss(
            (p[1:, :] - p[:-1, :])[valid_y],
            (t[1:, :] - t[:-1, :])[valid_y],
            beta=0.05,
        ) if valid_y.any() else p.new_zeros(())
        gradient_terms.append(0.5 * (gradient_x + gradient_y))
    shape = torch.stack(shape_terms).mean()
    metric = torch.stack(metric_terms).mean()
    gradient = torch.stack(gradient_terms).mean()
    total = shape + metric_weight * metric + gradient_weight * gradient
    return total, shape.detach(), metric.detach(), gradient.detach()


def make_model(source: dict):
    exp = source["experiment"]
    if exp.get("backbone") != "fastvit_t8.apple_in1k" or exp.get("fpn_fusion") != "concat_all":
        raise ValueError("Initialization must be the FastViT T8 V9.4 concat_all checkpoint")
    model = PortraitGeometryModel(
        backbone=exp["backbone"], pretrained=False,
        fpn_width=int(exp["fpn_width"]), fpn_fusion=exp["fpn_fusion"],
        decoder_refinement=exp.get("decoder_refinement", "none"),
        refinement_width=int(exp.get("refinement_width", 16)),
        fullres_width=int(exp.get("fullres_width", 24)),
        rgb_full_width=int(exp.get("rgb_full_width", 8)),
        guarded_refinement=bool(exp.get("guarded_refinement", False)),
        residual_depth_limit=float(exp.get("residual_depth_limit", 0.05)),
        residual_normal_limit=float(exp.get("residual_normal_limit", 0.15)),
        grayscale_refinement=bool(exp.get("grayscale_refinement", False)),
    )
    model.load_state_dict(source["model"], strict=True)
    model.normal_head.requires_grad_(False)
    if hasattr(model, "refinement"):
        model.refinement.normal_delta.requires_grad_(False)
    return model, exp


def save_checkpoint(path: Path, payload: dict):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--expected-images", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--snapshot-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=5e-6)
    parser.add_argument("--gradient-weight", type=float, default=0.2)
    parser.add_argument("--metric-weight", type=float, default=0.1)
    args = parser.parse_args()
    if (args.size < 32 or args.size % 32 or args.expected_images < 1 or
            min(args.batch_size, args.epochs, args.snapshot_interval, args.log_every) < 1 or
            args.workers < 0 or min(args.encoder_lr, args.head_lr) <= 0 or
            min(args.gradient_weight, args.metric_weight) < 0):
        parser.error("Invalid image size, count, rate, or loss weight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DGX training")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    splits, manifest_sha, teacher = load_manifest(args.manifest, args.expected_images)
    source = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model, source_exp = make_model(source)
    model.to("cuda")
    del source
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": args.encoder_lr},
        {"params": [p for name, p in model.named_parameters()
                    if not name.startswith("encoder.") and p.requires_grad], "lr": args.head_lr},
    ], weight_decay=1e-4)
    options = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": True}
    if args.workers:
        options["persistent_workers"] = True
    train_loader = DataLoader(MoGeDepthDataset(splits["train"], args.size), shuffle=True, **options)
    val_loader = DataLoader(MoGeDepthDataset(splits["validation"], args.size), shuffle=False, **options)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "manifest_sha256": manifest_sha,
        "init_checkpoint": str(args.init_checkpoint.resolve()),
        "teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
        "teacher_refine_steps": 0,
        "size": args.size, "epochs": args.epochs, "batch_size": args.batch_size,
        "seed": args.seed, "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
        "gradient_weight": args.gradient_weight, "metric_weight": args.metric_weight,
        "train_count": len(splits["train"]), "validation_count": len(splits["validation"]),
        "test_count": len(splits["test"]),
    }
    metadata_path = args.run_dir / "training_metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise RuntimeError("Existing run has different metadata; choose another run directory")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    experiment = dict(
        source_exp, name=args.run_dir.name, init_checkpoint=str(args.init_checkpoint.resolve()),
        input_size=[args.size, args.size], depth_training_target="MoGe-3 base metric pseudo-depth",
        normal_training=False, normal_head_frozen=True,
        teacher_refine_steps=0, teacher_manifest_sha256=manifest_sha,
    )
    best = float("inf")
    start_epoch = 1
    last_path = args.run_dir / "last_checkpoint.pt"
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("manifest_sha256") != manifest_sha or "optimizer" not in saved:
            raise RuntimeError("Cannot resume incompatible or incomplete checkpoint")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        best = float(saved["best_validation_loss"])
        print(f"Resuming from completed epoch {start_epoch - 1}; best={best:.5f}", flush=True)
    print(
        f"MoGe depth fine-tuning: train={len(splits['train'])} "
        f"validation={len(splits['validation'])} held_out_test={len(splits['test'])} "
        f"epochs={args.epochs} start={start_epoch} normal_head=frozen",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.monotonic()
        model.train()
        model.normal_head.eval()
        totals = []
        for step, (rgb, depth, valid) in enumerate(train_loader, 1):
            rgb = rgb.to("cuda", non_blocking=True)
            depth = depth.to("cuda", non_blocking=True)
            valid = valid.to("cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted, _, coarse, _ = model(rgb, return_coarse=True)
            main_loss, shape, metric, gradient = depth_loss(
                predicted, depth, valid, args.gradient_weight, args.metric_weight
            )
            coarse_loss, _, _, _ = depth_loss(
                coarse, depth, valid, args.gradient_weight, args.metric_weight
            )
            loss = main_loss + 0.3 * coarse_loss
            loss.backward()
            optimizer.step()
            totals.append(float(loss.detach()))
            if step % args.log_every == 0 or step == 1 or step == len(train_loader):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                    f"loss={np.mean(totals):.5f} shape={float(shape):.5f} "
                    f"metric={float(metric):.5f} gradient={float(gradient):.5f} "
                    f"{step * args.batch_size / elapsed:.1f} images/s",
                    flush=True,
                )
        model.eval()
        validation = []
        with torch.inference_mode():
            for rgb, depth, valid in val_loader:
                rgb = rgb.to("cuda", non_blocking=True)
                depth = depth.to("cuda", non_blocking=True)
                valid = valid.to("cuda", non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predicted, _ = model(rgb)
                score, _, _, _ = depth_loss(
                    predicted, depth, valid, args.gradient_weight, args.metric_weight
                )
                validation.append(float(score))
        score = float(np.mean(validation))
        if not np.isfinite(score):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}")
        is_best = score < best
        best = min(best, score)
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "experiment": experiment, "epoch": epoch, "validation_loss": score,
            "best_validation_loss": best, "manifest_sha256": manifest_sha,
        }
        if is_best:
            save_checkpoint(args.run_dir / "best_checkpoint.pt", payload)
        if epoch % args.snapshot_interval == 0:
            save_checkpoint(args.run_dir / f"checkpoint_epoch_{epoch:03d}.pt", payload)
        save_checkpoint(last_path, payload)
        minutes = (time.monotonic() - started) / 60
        with (args.run_dir / "history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps({
                "epoch": epoch, "train_loss": float(np.mean(totals)),
                "validation_loss": score, "best_validation_loss": best,
                "minutes": minutes,
            }) + "\n")
        print(
            f"Epoch {epoch}: train={np.mean(totals):.5f} val={score:.5f} "
            f"best={best:.5f} minutes={minutes:.1f}", flush=True,
        )
    print(f"READY: best={best:.5f}; checkpoint={args.run_dir / 'best_checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
