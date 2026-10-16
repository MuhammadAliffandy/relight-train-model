from __future__ import annotations

import argparse
import hashlib
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
from relight_train.data import RawSynthHumanDataset, build_raw_manifest
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
        torch.backends.cudnn.benchmark = str(exp.get("decoder_refinement", "none")) != "fullres"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "device": str(device)}, indent=2), flush=True)
    if device.type == "cuda":
        print(json.dumps({"cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu_count": torch.cuda.device_count(), "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "gpu_memory_gib": [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2) for i in range(torch.cuda.device_count())]}, indent=2), flush=True)
    else:
        raise RuntimeError("CUDA is unavailable. Activate the DGX Conda environment and check PyTorch CUDA support.")

    rows, partitions, inventory = build_raw_manifest(data_cfg["rclone_source"], data_cfg["source_root"], run_dir, int(data_cfg["expected_samples"]), seed, data_cfg["inventory_dir"])
    selection_sha256 = hashlib.sha256("\n".join(sorted(row["subject_id"] for row in rows)).encode("utf-8")).hexdigest()
    part_sizes = {key: len(value) for key, value in partitions.items()}
    print(f"Raw dataset manifest ready: {sum(part_sizes.values()):,} samples; split={part_sizes}; inventory={inventory}", flush=True)
    datasets = {
        key: RawSynthHumanDataset(value, exp["input_size"], augment=(key == "train"))
        for key, value in partitions.items()
    }
    workers = int(exp["workers"])
    loaders = {key: DataLoader(value, batch_size=int(exp["batch_size"]), shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=(workers > 0), drop_last=False) for key, value in datasets.items()}
    sample = datasets["train"][0]
    image, depth, normal, mask = sample
    valid_depth = depth[0][mask[0] > 0]
    print(f"Sample audit: RGB={tuple(image.shape)} depth_m={tuple(depth.shape)} normals={tuple(normal.shape)} mask={tuple(mask.shape)} depth_p01/p50/p99={torch.quantile(valid_depth, torch.tensor([.01,.5,.99])).tolist()} mask_fraction={float(mask.mean()):.4f}", flush=True)
    if any(size == 0 for size in part_sizes.values()):
        raise RuntimeError(f"All train/val/test partitions must be non-empty: {part_sizes}")

    fpn_fusion = str(exp.get("fpn_fusion", "add"))
    decoder_refinement = str(exp.get("decoder_refinement", "none"))
    refinement_width = int(exp.get("refinement_width", 16))
    fullres_width = int(exp.get("fullres_width", 24))
    rgb_full_width = int(exp.get("rgb_full_width", 8))
    model = PortraitGeometryModel(
        exp["backbone"],
        bool(exp["pretrained"]),
        int(exp["fpn_width"]),
        fpn_fusion,
        decoder_refinement,
        refinement_width,
        fullres_width,
        rgb_full_width,
    ).to(device)
    objective = GeometryObjective(exp["depth_mode"]).to(device)
    init_checkpoint = exp.get("init_checkpoint")
    if init_checkpoint and not (run_dir / "last_checkpoint.pt").is_file():
        init_checkpoint = Path(init_checkpoint)
        if not init_checkpoint.is_file():
            raise FileNotFoundError(f"Fine-tune initialization checkpoint not found: {init_checkpoint}")
        initial_state = torch.load(init_checkpoint, map_location=device, weights_only=False)
        source_experiment = initial_state.get("experiment", {})
        init_mode = str(exp.get("init_mode", "full"))
        if init_mode == "encoder_only":
            if source_experiment.get("backbone") != exp.get("backbone"):
                raise RuntimeError(
                    "Encoder-only initialization requires the same backbone: "
                    f"{source_experiment.get('backbone')} vs {exp.get('backbone')}"
                )
            if source_experiment.get("depth_mode") != exp.get("depth_mode"):
                raise RuntimeError(
                    "Encoder-only initialization requires the same depth_mode: "
                    f"{source_experiment.get('depth_mode')} vs {exp.get('depth_mode')}"
                )
            current_state = model.state_dict()
            source_model_state = initial_state["model"]
            encoder_weights = {
                key: value
                for key, value in source_model_state.items()
                if key.startswith("encoder.")
                and key in current_state
                and current_state[key].shape == value.shape
            }
            expected_encoder = {key for key in current_state if key.startswith("encoder.")}
            if set(encoder_weights) != expected_encoder:
                missing_encoder = sorted(expected_encoder - set(encoder_weights))
                raise RuntimeError(
                    "Could not transfer the complete encoder from initialization checkpoint; "
                    f"missing keys: {missing_encoder[:8]}"
                )
            model.load_state_dict(encoder_weights, strict=False)
            if "objective" in initial_state:
                objective.load_state_dict(initial_state["objective"])
            print(
                "Transferred FastViT encoder and task objective only; "
                "FPN, heads, and full-resolution decoder start fresh.",
                flush=True,
            )
            source_fusion = str(source_experiment.get("fpn_fusion", "add"))
            source_refinement = str(source_experiment.get("decoder_refinement", "none"))
        else:
            compatibility = ("backbone", "fpn_width", "depth_mode")
            mismatches = {
                key: (source_experiment.get(key), exp.get(key))
                for key in compatibility
                if source_experiment.get(key) != exp.get(key)
            }
            source_fusion = str(source_experiment.get("fpn_fusion", "add"))
            if source_fusion != fpn_fusion and not (source_fusion == "add" and fpn_fusion == "concat"):
                mismatches["fpn_fusion"] = (source_fusion, fpn_fusion)
            source_refinement = str(source_experiment.get("decoder_refinement", "none"))
            if source_refinement != decoder_refinement and not (
                source_refinement == "none" and decoder_refinement == "stride2"
            ):
                mismatches["decoder_refinement"] = (source_refinement, decoder_refinement)
            if mismatches:
                raise RuntimeError(f"Initialization checkpoint architecture/config mismatch: {mismatches}")
            if source_fusion == fpn_fusion and source_refinement == decoder_refinement:
                model.load_state_dict(initial_state["model"])
            else:
                loaded = model.load_state_dict(initial_state["model"], strict=False)
                expected_missing = set()
                if source_fusion != fpn_fusion:
                    expected_missing.update(
                        key for key in model.state_dict() if key.startswith("neck.fusions.")
                    )
                if source_refinement != decoder_refinement:
                    expected_missing.update(
                        key for key in model.state_dict() if key.startswith("refinement.")
                    )
                if set(loaded.missing_keys) != expected_missing or loaded.unexpected_keys:
                    raise RuntimeError(
                        "Unexpected checkpoint transfer while initializing the new architecture: "
                        f"missing={loaded.missing_keys}, unexpected={loaded.unexpected_keys}"
                    )
                if source_fusion != fpn_fusion:
                    print("Transferred additive FPN weights; new concat layers start as exact addition.", flush=True)
                if source_refinement != decoder_refinement:
                    print("Transferred V2 weights; stride-2 refinement starts with zero output corrections.", flush=True)
            if "objective" in initial_state:
                objective.load_state_dict(initial_state["objective"])
        resolution_note = "same input resolution" if source_experiment.get("input_size") == exp["input_size"] else f"resolution {source_experiment.get('input_size')} -> {exp['input_size']}"
        transfer_note = (
            "encoder-only warm start; FPN and decoder initialize fresh"
            if str(exp.get("init_mode", "full")) == "encoder_only"
            else "full compatible-model warm start"
        )
        print(f"Initialized from {init_checkpoint}; {transfer_note}; {resolution_note}; optimizer and scheduler start fresh.", flush=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    audit_batch_size = 1 if decoder_refinement == "fullres" else 2
    print(f"Running model output-shape audit (batch={audit_batch_size}, refinement={decoder_refinement})...", flush=True)
    with torch.inference_mode():
        audit_input = torch.zeros(audit_batch_size, 3, *exp["input_size"], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(exp["amp"])):
            result = model(audit_input)
        torch.cuda.synchronize(device)
    height, width = (int(size) for size in exp["input_size"])
    if [tuple(item.shape) for item in result] != [
        (audit_batch_size, 1, height, width),
        (audit_batch_size, 3, height, width),
    ]:
        raise RuntimeError("Model output shape audit failed")
    print("Model output-shape audit passed.", flush=True)
    print(f"Model={exp['backbone']} parameters={parameter_count / 1e6:.2f}M FPN={exp['fpn_width']} fusion={fpn_fusion} refinement={decoder_refinement}:stride2={refinement_width}:fullres={fullres_width} targets=depth+normals", flush=True)

    encoder_lr = exp.get("encoder_learning_rate")
    if encoder_lr is None:
        optimizer_parameters = list(model.parameters()) + list(objective.parameters())
    else:
        encoder_parameter_ids = {id(parameter) for parameter in model.encoder.parameters()}
        decoder_parameters = [
            parameter for parameter in model.parameters()
            if id(parameter) not in encoder_parameter_ids
        ]
        optimizer_parameters = [
            {"params": list(model.encoder.parameters()), "lr": float(encoder_lr)},
            {"params": decoder_parameters + list(objective.parameters()), "lr": float(exp["learning_rate"])},
        ]
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=float(exp["learning_rate"]), weight_decay=float(exp["weight_decay"]))
    scheduler_name = str(exp.get("scheduler", "plateau")).lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(exp["epochs"]),
            eta_min=float(exp.get("min_learning_rate", 1e-6)),
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    else:
        raise ValueError(f"Unsupported scheduler {scheduler_name!r}; expected 'cosine' or 'plateau'.")
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
        if state.get("selection_sha256") != selection_sha256:
            raise RuntimeError("Checkpoint belongs to a different raw data selection; use a new run_dir or move old checkpoints.")
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
        return {"epoch": epoch, "batch_in_epoch": batch, "epoch_order": order, "partial_train_totals": train_values, "partial_task_totals": task_values, "train_seconds_elapsed": elapsed, "model": model.state_dict(), "objective": objective.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "best_score": best_score, "history": history, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "selection_sha256": selection_sha256, "experiment": exp}

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
                epoch_order = [(index, (seed + epoch * 1_000_003 + index) % (2**32)) for index in range(len(datasets["train"]))]
                random.Random(seed + epoch).shuffle(epoch_order)
                resume_batch = 0
            loader = DataLoader(datasets["train"], batch_size=int(exp["batch_size"]), sampler=epoch_order[resume_batch * int(exp["batch_size"]):], num_workers=workers, pin_memory=True, persistent_workers=(workers > 0), drop_last=False)
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
                progress.set_postfix(iteration=f"{completed}/{train_steps}", loss=f"{float(loss.detach()):.4f}", depth=f"{float(parts['depth_loss']):.4f}", normal=f"{float(parts['normal_loss']):.4f}", gpu_mem=f"{memory:.1f}GiB", data_s=f"{current_wait:.2f}", refresh=False)
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
            if scheduler_name == "cosine":
                scheduler.step()
            else:
                scheduler.step(score)
            improved = score < best_score
            if improved:
                best_score = score
            peak_memory = torch.cuda.max_memory_allocated(device) / 2**30
            record = {"epoch": epoch + 1, "session_data_wait_seconds": data_wait, "peak_gpu_memory_gib": peak_memory, "train_loss": float(np.mean(train_totals)), **{key: float(np.mean(value)) for key, value in task_totals.items()}, "val": val_metrics, "selection_score": score, "train_seconds": train_seconds, "validation_seconds": val_seconds, "epoch_seconds": epoch_seconds, "train_samples_per_second": sample_rate, "eta_seconds": eta, "lr": optimizer.param_groups[-1]["lr"], "encoder_lr": optimizer.param_groups[0]["lr"], "task_sigmas": torch.exp(objective.log_vars.detach()).cpu().tolist(), "steps_per_epoch": train_steps, "train_samples": part_sizes["train"], "validation_samples": part_sizes["val"], "preview_matrix": str(preview_path)}
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
