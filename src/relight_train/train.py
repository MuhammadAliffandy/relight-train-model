from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from relight_train.config import load_config
from relight_train.data import PreparedShardDataset, load_packed_manifest, shard_sample_order
from relight_train.model import PortraitGeometryModel
from relight_train.objectives import GeometryObjective, evaluate


def atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def remote_backup(local_path: Path, remote_dir: str):
    if not remote_dir:
        return
    try:
        subprocess.run(["rclone", "copyto", str(local_path), f"{remote_dir.rstrip('/')}/{local_path.name}"], check=True)
        print(f"Backed up {local_path.name} to {remote_dir}", flush=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"WARNING: local file is safe, but rclone backup failed for {local_path.name}: {error}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train SynthHuman depth + surface-normal model")
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    args = parser.parse_args()
    config = load_config(args.config)
    exp, data_cfg, output_cfg = config["experiment"], config["data"], config["output"]
    run_dir: Path = output_cfg["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "epoch_outputs").mkdir(exist_ok=True)
    shutil.copy2(args.config, run_dir / "config.yaml")

    seed = int(exp["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "device": str(device)}, indent=2), flush=True)
    if device.type == "cuda":
        print(json.dumps({"cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu_count": torch.cuda.device_count(), "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "gpu_memory_gib": [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2) for i in range(torch.cuda.device_count())]}, indent=2), flush=True)
    else:
        raise RuntimeError("CUDA is unavailable. Activate the DGX Conda environment and check PyTorch CUDA support.")

    manifest, records, partitions = load_packed_manifest(data_cfg["shard_dir"], int(data_cfg["expected_samples"]), int(data_cfg["expected_seed"]))
    part_sizes = {key: len(value) for key, value in partitions.items()}
    print(f"Prepared dataset OK: {sum(part_sizes.values()):,} samples; split={part_sizes}; shards={len(records)}; selection={manifest['config']['selection_sha256'][:16]}", flush=True)
    datasets = {
        key: PreparedShardDataset(value, records, data_cfg["shard_dir"], data_cfg["cache_dir"], data_cfg["cache_gib"], data_cfg["min_free_gib"], augment=(key == "train"))
        for key, value in partitions.items()
    }
    loaders = {key: DataLoader(value, batch_size=int(exp["batch_size"]), shuffle=False, num_workers=0, pin_memory=True, drop_last=False) for key, value in datasets.items()}
    sample = datasets["train"][0]
    image, depth, normal, mask = sample
    valid_depth = depth[0][mask[0] > 0]
    print(f"Sample audit: RGB={tuple(image.shape)} depth_m={tuple(depth.shape)} normals={tuple(normal.shape)} mask={tuple(mask.shape)} depth_p01/p50/p99={torch.quantile(valid_depth, torch.tensor([.01,.5,.99])).tolist()} mask_fraction={float(mask.mean()):.4f}", flush=True)
    if any(size == 0 for size in part_sizes.values()):
        raise RuntimeError(f"All train/val/test partitions must be non-empty: {part_sizes}")

    model = PortraitGeometryModel(exp["backbone"], bool(exp["pretrained"]), int(exp["fpn_width"])).to(device)
    objective = GeometryObjective(exp["depth_mode"]).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with torch.inference_mode():
        result = model(torch.zeros(2, 3, *exp["input_size"], device=device))
    if [tuple(item.shape) for item in result] != [(2, 1, 256, 256), (2, 3, 256, 256)]:
        raise RuntimeError("Model output shape audit failed")
    print(f"Model={exp['backbone']} parameters={parameter_count / 1e6:.2f}M FPN={exp['fpn_width']} targets=depth+normals", flush=True)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(objective.parameters()), lr=float(exp["learning_rate"]), weight_decay=float(exp["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(exp["amp"]))
    checkpoint = run_dir / "best_checkpoint.pt"
    last_checkpoint = run_dir / "last_checkpoint.pt"
    history_path = run_dir / "history.json"
    progress_path = run_dir / "progress.json"
    history = []
    start_epoch = resume_batch = 0
    resume_order = None
    resume_train_totals = []
    resume_task_totals = {}
    resume_train_seconds = 0.0
    best_score = float("inf")
    if last_checkpoint.is_file():
        state = torch.load(last_checkpoint, map_location=device, weights_only=False)
        if state.get("selection_sha256") != manifest["config"]["selection_sha256"]:
            raise RuntimeError("Checkpoint belongs to a different prepared data selection; use a new run_dir or move old checkpoints.")
        model.load_state_dict(state["model"])
        objective.load_state_dict(state["objective"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]
        resume_batch = state.get("batch_in_epoch", 0)
        resume_order = state.get("epoch_order")
        best_score = state.get("best_score", float("inf"))
        history = state.get("history", [])
        resume_train_totals = state.get("partial_train_totals", [])
        resume_task_totals = state.get("partial_task_totals", {})
        resume_train_seconds = state.get("train_seconds_elapsed", 0.0)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resuming at epoch {start_epoch + 1}, after {resume_batch} completed batches.", flush=True)

    train_steps = math.ceil(part_sizes["train"] / int(exp["batch_size"]))
    total_steps = train_steps * int(exp["epochs"])
    print(f"Run plan: epochs={exp['epochs']} batch={exp['batch_size']} steps/epoch={train_steps:,} max_steps={total_steps:,} output={run_dir}", flush=True)

    def state_payload(epoch, batch, order, train_values, task_values, elapsed):
        return {"epoch": epoch, "batch_in_epoch": batch, "epoch_order": order, "partial_train_totals": train_values, "partial_task_totals": task_values, "train_seconds_elapsed": elapsed, "model": model.state_dict(), "objective": objective.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "best_score": best_score, "history": history, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "selection_sha256": manifest["config"]["selection_sha256"], "experiment": exp}

    try:
        for epoch in range(start_epoch, int(exp["epochs"])):
            torch.cuda.synchronize()
            epoch_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            model.train()
            objective.train()
            if epoch == start_epoch and resume_batch:
                train_totals = list(resume_train_totals)
                task_totals = {key: list(value) for key, value in resume_task_totals.items()}
                epoch_order = resume_order
            else:
                train_totals = []
                task_totals = {"depth_loss": [], "normal_loss": []}
                epoch_order = shard_sample_order(datasets["train"], seed + epoch)
                resume_batch = 0
            loader = DataLoader(datasets["train"], batch_size=int(exp["batch_size"]), sampler=epoch_order[resume_batch * int(exp["batch_size"]):], num_workers=0, pin_memory=True, drop_last=False)
            prior_seconds = resume_train_seconds if epoch == start_epoch else 0.0
            last_checkpoint_time = time.monotonic()
            last_batch_end = time.perf_counter()
            data_wait = 0.0
            progress = tqdm(loader, total=train_steps, initial=resume_batch, desc=f"Epoch {epoch + 1}/{exp['epochs']}", unit="batch", dynamic_ncols=True)
            for batch_index, (x, d, n, m) in enumerate(progress, start=resume_batch):
                current_wait = time.perf_counter() - last_batch_end
                data_wait += current_wait
                x, d, n, m = [item.to(device, non_blocking=True) for item in (x, d, n, m)]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(exp["amp"])):
                    predicted_depth, predicted_normal = model(x)
                    loss, parts = objective(predicted_depth, d, predicted_normal, n, m)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, batch={batch_index + 1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(objective.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
                train_totals.append(float(loss.detach()))
                for key, value in parts.items():
                    task_totals[key].append(float(value))
                completed = batch_index + 1
                memory = torch.cuda.memory_allocated(device) / 2**30
                progress.set_postfix(iteration=f"{completed}/{train_steps}", loss=f"{float(loss.detach()):.4f}", depth=f"{float(parts['depth_loss']):.4f}", normal=f"{float(parts['normal_loss']):.4f}", gpu_mem=f"{memory:.1f}GiB", shards=datasets["train"].stats["copies"], data_s=f"{current_wait:.2f}", refresh=False)
                if completed % int(exp["progress_json_interval_steps"]) == 0:
                    atomic_json(progress_path, {"epoch": epoch + 1, "epochs_total": exp["epochs"], "batches_complete": completed, "batches_total": train_steps, "latest_train_loss": train_totals[-1], "last_update_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                if completed % int(exp["log_interval_steps"]) == 0:
                    print(f"epoch {epoch + 1:03d}/{exp['epochs']} batch {completed}/{train_steps} loss={train_totals[-1]:.4f} depth={float(parts['depth_loss']):.4f} normal={float(parts['normal_loss']):.4f} data_wait={current_wait:.2f}s gpu_mem={memory:.1f}GiB", flush=True)
                if time.monotonic() - last_checkpoint_time >= int(exp["checkpoint_interval_minutes"]) * 60:
                    torch.cuda.synchronize()
                    save_checkpoint(last_checkpoint, state_payload(epoch, completed, epoch_order, train_totals, task_totals, prior_seconds + time.perf_counter() - epoch_start))
                    remote_backup(last_checkpoint, output_cfg.get("backup_remote", ""))
                    last_checkpoint_time = time.monotonic()
                last_batch_end = time.perf_counter()

            torch.cuda.synchronize()
            train_seconds = prior_seconds + time.perf_counter() - epoch_start
            val_start = time.perf_counter()
            val_metrics = evaluate(model, loaders["val"], device)
            torch.cuda.synchronize()
            val_seconds = time.perf_counter() - val_start
            preview_x, preview_d, preview_n, preview_m = next(iter(loaders["val"]))
            model.eval()
            with torch.inference_mode():
                preview_depth, preview_normal = model(preview_x[:1].to(device))
            preview_path = run_dir / "epoch_outputs" / f"epoch_{epoch + 1:03d}_sample.npz"
            np.savez_compressed(preview_path, depth_pred_m=preview_depth[0, 0].float().cpu().numpy(), normal_pred_xyz=preview_normal[0].float().cpu().numpy(), depth_gt_m=preview_d[0, 0].numpy(), normal_gt_xyz=preview_n[0].numpy(), valid_mask=preview_m[0, 0].numpy())
            epoch_seconds = time.perf_counter() - epoch_start
            sample_rate = part_sizes["train"] / max(train_seconds, 1e-6)
            eta = max(int(exp["epochs"]) - epoch - 1, 0) * epoch_seconds
            score = val_metrics["depth_rmse_m"] / 0.1 + val_metrics["normal_mean_deg"] / 30.0
            scheduler.step(score)
            improved = score < best_score
            if improved:
                best_score = score
            peak_memory = torch.cuda.max_memory_allocated(device) / 2**30
            record = {"epoch": epoch + 1, "session_data_wait_seconds": data_wait, "peak_gpu_memory_gib": peak_memory, "train_loss": float(np.mean(train_totals)), **{key: float(np.mean(value)) for key, value in task_totals.items()}, "val": val_metrics, "selection_score": score, "train_seconds": train_seconds, "validation_seconds": val_seconds, "epoch_seconds": epoch_seconds, "train_samples_per_second": sample_rate, "eta_seconds": eta, "lr": optimizer.param_groups[0]["lr"], "task_sigmas": torch.exp(objective.log_vars.detach()).cpu().tolist(), "steps_per_epoch": train_steps, "train_samples": part_sizes["train"], "validation_samples": part_sizes["val"], "preview_matrix": str(preview_path)}
            history.append(record)
            atomic_json(history_path, history)
            completed = state_payload(epoch + 1, 0, None, [], {}, 0.0)
            save_checkpoint(last_checkpoint, completed)
            remote_backup(last_checkpoint, output_cfg.get("backup_remote", ""))
            if improved:
                completed["best_epoch"] = epoch + 1
                save_checkpoint(checkpoint, completed)
                remote_backup(checkpoint, output_cfg.get("backup_remote", ""))
            remote_backup(history_path, output_cfg.get("backup_remote", ""))
            atomic_json(progress_path, {"epoch": epoch + 1, "epochs_total": exp["epochs"], "status": "epoch_complete", "train_seconds": train_seconds, "validation_seconds": val_seconds, "train_samples_per_second": sample_rate, "eta_seconds": eta, "preview_matrix": str(preview_path)})
            resume_batch = 0
            resume_order = None
            resume_train_totals = []
            resume_task_totals = {}
            resume_train_seconds = 0.0
            print(f"epoch {epoch + 1:03d}/{exp['epochs']} iterations={train_steps:,} train={record['train_loss']:.4f} val depth(RMSE m)={val_metrics['depth_rmse_m']:.4f} normal(mean°)={val_metrics['normal_mean_deg']:.2f} score={score:.4f} train={train_seconds / 60:.1f}m val={val_seconds / 60:.1f}m rate={sample_rate:.1f} samples/s ETA={eta / 60:.1f}m", flush=True)

        if not checkpoint.is_file() and last_checkpoint.is_file():
            shutil.copy2(last_checkpoint, checkpoint)
        best_state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model"])
        test_metrics = evaluate(model, loaders["test"], device)
        atomic_json(run_dir / "test_metrics.json", test_metrics)
        remote_backup(run_dir / "test_metrics.json", output_cfg.get("backup_remote", ""))
        print("Held-out test metrics:", json.dumps(test_metrics, indent=2), flush=True)
        print(f"Training complete. Best epoch={best_state.get('best_epoch', best_state['epoch'])}; checkpoint={checkpoint}", flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()
