"""Fine-tune V9 depth on Face-Depth-3D relative labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from relight_train.model import PortraitGeometryModel
from relight_train.objectives import pseudo_depth_distillation_loss


class FaceDepthDataset(Dataset):
    def __init__(self, rows, size):
        self.rows = rows
        self.size = size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image"]) as source:
            image = source.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
            rgb = np.asarray(image, dtype=np.float32).copy() / 255.0
        with Image.open(row["inferno_index"]) as source:
            label = source.convert("L").resize((self.size, self.size), Image.Resampling.BILINEAR)
            proximity = np.asarray(label, dtype=np.float32).copy() / 255.0
        valid = proximity > 0.03
        return (
            torch.from_numpy(rgb).permute(2, 0, 1),
            torch.from_numpy(proximity[None]),
            torch.from_numpy(valid[None]),
        )


def split_rows(path, seed):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    groups = sorted({row["group_id"] for row in rows})
    random.Random(seed).shuffle(groups)
    validation_groups = set(groups[:max(1, round(len(groups) * 0.1))])
    train = [row for row in rows if row["group_id"] not in validation_groups]
    validation = [row for row in rows if row["group_id"] in validation_groups]
    if not train or not validation:
        raise ValueError("Group split produced an empty partition")
    for row in rows:
        if not Path(row["image"]).is_file() or not Path(row["inferno_index"]).is_file():
            raise FileNotFoundError(f"Missing image or label for {row['sample_id']}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return train, validation, digest


def edge_aware_smoothness_loss(predicted_depth, rgb, valid_mask):
    """Smooth relative proximity inside valid regions while retaining RGB edges."""
    inverse_depth = predicted_depth.float().clamp_min(1e-5).reciprocal()
    losses = []
    for index in range(predicted_depth.shape[0]):
        valid = valid_mask[index, 0].bool()
        if int(valid.sum()) < 64:
            continue
        values = inverse_depth[index, 0]
        sampled_valid = valid[::4, ::4]
        reference = values.detach()[::4, ::4][sampled_valid]
        if reference.numel() < 64:
            reference = values.detach()[valid]
        low, high = torch.quantile(reference, values.new_tensor([0.05, 0.95]))
        normalized = ((values - low) / (high - low).clamp_min(1e-4)).clamp(0, 1)

        valid_x = valid[:, 1:] & valid[:, :-1]
        valid_y = valid[1:, :] & valid[:-1, :]
        image = rgb[index].float()
        image_dx = (image[:, :, 1:] - image[:, :, :-1]).abs().mean(0)
        image_dy = (image[:, 1:, :] - image[:, :-1, :]).abs().mean(0)
        weight_x = torch.exp(-10.0 * image_dx)
        weight_y = torch.exp(-10.0 * image_dy)
        depth_dx = (normalized[:, 1:] - normalized[:, :-1]).abs()
        depth_dy = (normalized[1:, :] - normalized[:-1, :]).abs()
        loss_x = (depth_dx * weight_x * valid_x).sum() / valid_x.sum().clamp_min(1)
        loss_y = (depth_dy * weight_y * valid_y).sum() / valid_y.sum().clamp_min(1)
        losses.append(0.5 * (loss_x + loss_y))
    if not losses:
        raise ValueError("A batch has no valid pixels for edge-aware smoothness")
    return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/face_depth_3d/manifest.jsonl"))
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fusion", choices=("concat", "concat_all"), required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--min-delta", type=float, default=0.00005)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--snapshot-interval", type=int, default=25)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    args = parser.parse_args()
    if args.size % 32 or args.size < 32 or min(args.batch_size, args.epochs, args.patience) < 1:
        parser.error("size must be a positive multiple of 32; batch-size, epochs, patience must be positive")
    if (args.workers < 0 or args.min_delta < 0 or args.smoothness_weight < 0
            or args.snapshot_interval < 1 or args.log_every < 1):
        parser.error("workers, min-delta, and smoothness-weight must be nonnegative; snapshot-interval and log-every must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DGX training")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_rows, val_rows, manifest_sha = split_rows(args.manifest, args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    source = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    exp = source["experiment"]
    model = PortraitGeometryModel(
        backbone=exp["backbone"], pretrained=False,
        fpn_width=int(exp["fpn_width"]), fpn_fusion=args.fusion,
        decoder_refinement=exp.get("decoder_refinement", "none"),
        refinement_width=int(exp.get("refinement_width", 16)),
        fullres_width=int(exp.get("fullres_width", 24)),
        rgb_full_width=int(exp.get("rgb_full_width", 8)),
        guarded_refinement=bool(exp.get("guarded_refinement", False)),
        residual_depth_limit=float(exp.get("residual_depth_limit", 0.05)),
        residual_normal_limit=float(exp.get("residual_normal_limit", 0.15)),
        grayscale_refinement=bool(exp.get("grayscale_refinement", False)),
    )
    source_fusion = exp.get("fpn_fusion")
    if source_fusion not in {"concat", "concat_all"} or exp.get("depth_mode") != "metric":
        raise ValueError("Initialization must be a metric-depth V9 checkpoint with concat or concat_all FPN")
    if args.fusion != source_fusion and not (source_fusion == "concat" and args.fusion == "concat_all"):
        raise ValueError(f"Cannot transfer FPN fusion {source_fusion!r} into {args.fusion!r} safely")
    loaded = model.load_state_dict(source["model"], strict=False)
    expected_missing = (
        {"neck.all_stage_fusion.weight", "neck.all_stage_fusion.bias"}
        if source_fusion == "concat" and args.fusion == "concat_all" else set()
    )
    if set(loaded.missing_keys) != expected_missing or loaded.unexpected_keys:
        raise RuntimeError(f"Checkpoint transfer mismatch: {loaded}")
    for parameter in model.normal_head.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    del source
    optimizer = torch.optim.AdamW([
        {"params": [parameter for parameter in model.encoder.parameters() if parameter.requires_grad], "lr": args.encoder_lr},
        {"params": [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.") and parameter.requires_grad], "lr": args.head_lr},
    ], weight_decay=1e-4)
    loader_options = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": True}
    train_loader = DataLoader(FaceDepthDataset(train_rows, args.size), shuffle=True, **loader_options)
    val_loader = DataLoader(FaceDepthDataset(val_rows, args.size), shuffle=False, **loader_options)
    experiment = dict(exp, name=args.run_dir.name, init_checkpoint=str(args.init_checkpoint),
                      input_size=[args.size, args.size], fpn_fusion=args.fusion,
                      edge_aware_smoothness_weight=args.smoothness_weight,
                      depth_training_target="Face-Depth-3D inferno index relative proximity",
                      normal_training=False)
    metadata = {"manifest_sha256": manifest_sha, "train_count": len(train_rows),
                "validation_count": len(val_rows), "seed": args.seed, "size": args.size,
                "fusion": args.fusion, "init_checkpoint": str(args.init_checkpoint),
                "epochs": args.epochs, "patience": args.patience, "min_delta": args.min_delta,
                "smoothness_weight": args.smoothness_weight,
                "snapshot_interval": args.snapshot_interval, "log_every": args.log_every}
    (args.run_dir / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    best = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        model.normal_head.eval()
        totals = []
        print(f"Epoch {epoch}/{args.epochs} creating data iterator", flush=True)
        train_iterator = iter(train_loader)
        for step in range(1, len(train_loader) + 1):
            should_log = step % args.log_every == 0 or step == 1 or step == len(train_loader)
            if should_log:
                print(f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} fetching batch", flush=True)
            rgb, proximity, valid = next(train_iterator)
            if should_log:
                print(f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} computing", flush=True)
            rgb, proximity, valid = rgb.to(device), proximity.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predicted, _, coarse, _ = model(rgb, return_coarse=True)
                depth_loss = pseudo_depth_distillation_loss(predicted, proximity, valid)
                coarse_loss = pseudo_depth_distillation_loss(coarse, proximity, valid)
                smooth_loss = edge_aware_smoothness_loss(predicted, rgb, valid)
                loss = depth_loss + 0.3 * coarse_loss + args.smoothness_weight * smooth_loss
            loss.backward()
            optimizer.step()
            totals.append(float(loss.detach()))
            if should_log:
                elapsed = max(time.monotonic() - started, 1e-6)
                epoch_eta = (len(train_loader) - step) * elapsed / step / 60
                gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
                print(f"Epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} "
                      f"loss={np.mean(totals):.4f} smooth={float(smooth_loss.detach()):.4f} "
                      f"{step * args.batch_size / elapsed:.1f} images/s "
                      f"epoch_eta={epoch_eta:.1f}m gpu_alloc={gpu_memory:.1f}GiB", flush=True)
        model.eval()
        validation = []
        validation_smoothness = []
        with torch.inference_mode():
            for rgb, proximity, valid in val_loader:
                rgb, proximity, valid = rgb.to(device), proximity.to(device), valid.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predicted, _ = model(rgb)
                    validation.append(float(pseudo_depth_distillation_loss(predicted, proximity, valid)))
                    validation_smoothness.append(float(edge_aware_smoothness_loss(predicted, rgb, valid)))
        score = float(np.mean(validation))
        smooth_score = float(np.mean(validation_smoothness))
        epoch_minutes = (time.monotonic() - started) / 60
        print(f"Epoch {epoch}: train={np.mean(totals):.5f} val={score:.5f} "
              f"val_smooth={smooth_score:.5f} minutes={epoch_minutes:.1f}", flush=True)
        with (args.run_dir / "history.jsonl").open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps({
                "epoch": epoch,
                "train_loss": float(np.mean(totals)),
                "validation_relative_loss": score,
                "validation_smoothness_loss": smooth_score,
                "minutes": epoch_minutes,
            }) + "\n")
        payload = {"model": model.state_dict(), "experiment": experiment, "epoch": epoch,
                   "validation_relative_loss": score, "validation_smoothness_loss": smooth_score,
                   "manifest_sha256": manifest_sha}
        temporary = args.run_dir / "last_checkpoint.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(args.run_dir / "last_checkpoint.pt")
        if epoch % args.snapshot_interval == 0:
            snapshot = args.run_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            temporary = snapshot.with_suffix(snapshot.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(snapshot)
        if score < best - args.min_delta:
            best = score
            stale = 0
            temporary = args.run_dir / "best_checkpoint.pt.tmp"
            torch.save(payload, temporary)
            temporary.replace(args.run_dir / "best_checkpoint.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}; best validation loss={best:.5f}", flush=True)
                break
    print(f"READY: best={best:.5f}; checkpoint={args.run_dir / 'best_checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
