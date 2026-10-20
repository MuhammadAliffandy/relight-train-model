from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


FILE_PATTERN = re.compile(r"^(rgb|alpha|depth|normal)_(\d+)\.(png|exr)$")
REQUIRED = {"rgb", "alpha", "depth", "normal"}


def _atomic_json(path: Path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_raw_manifest(remote_path: str, source_root: Path, output_dir: Path, sample_count: int, seed: int, inventory_dir: Path, val_fraction=0.15, test_fraction=0.15):
    """List modality filenames with rclone, then save a deterministic raw-file manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "raw_dataset_manifest.jsonl"
    report_path = output_dir / "raw_dataset_inventory.json"
    metadata_path = output_dir / "raw_dataset_manifest.meta.json"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    inventory_paths = inventory_dir / "relative_file_paths.txt"
    inventory_meta_path = inventory_dir / "inventory.meta.json"
    expected_inventory_meta = {"remote_path": remote_path, "source_root": str(source_root), "inventory_version": 1}
    expected_meta = {"remote_path": remote_path, "source_root": str(source_root), "samples": sample_count, "seed": seed, "val_fraction": val_fraction, "test_fraction": test_fraction, "manifest_version": 1}
    if manifest_path.is_file() and metadata_path.is_file():
        saved_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved_meta == expected_meta:
            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) == sample_count:
                parts = {name: [row for row in rows if row["partition"] == name] for name in ("train", "val", "test")}
                report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
                print(f"Reusing indexed raw manifest: {len(rows):,} samples ({manifest_path})", flush=True)
                return rows, parts, report

    if not source_root.is_dir():
        raise FileNotFoundError(f"rclone mount source folder is not accessible: {source_root}")
    groups = {}
    duplicate_ids = set()
    scanned = 0
    started = time.perf_counter()

    def add_path(relative, file_id):
        nonlocal scanned
        relative = relative.strip().lstrip("/")
        if not relative:
            return
        scanned += 1
        filename = Path(relative).name
        match = FILE_PATTERN.fullmatch(filename)
        if not match:
            return
        kind, sample_id, extension = match.groups()
        if extension != ("png" if kind in ("rgb", "alpha") else "exr"):
            return
        sample_id = int(sample_id)
        group = groups.setdefault(sample_id, {})
        object_ids = group.setdefault("_object_ids", {})
        if kind in group and (group[kind] != relative or object_ids.get(kind) != file_id):
            duplicate_ids.add(sample_id)
        else:
            group[kind] = relative
            object_ids[kind] = file_id

    cache_valid = False
    if inventory_paths.is_file() and inventory_meta_path.is_file():
        cache_valid = json.loads(inventory_meta_path.read_text(encoding="utf-8")) == expected_inventory_meta
    if cache_valid:
        print(f"Loading cached raw filename inventory: {inventory_paths}", flush=True)
        with inventory_paths.open(encoding="utf-8") as handle:
            for line in handle:
                relative, file_id = line.rstrip("\n").split("\t", 1)
                add_path(relative, file_id)
    else:
        command = [os.environ.get("RELIGHT_RCLONE_BIN", "rclone"), "lsf", "--recursive", "--files-only", "--format", "pi", "--separator", "\t", remote_path]
        print("Indexing filenames through rclone (metadata only; no image contents downloaded yet):", " ".join(command), flush=True)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, bufsize=1)
        temporary_inventory = inventory_paths.with_suffix(".txt.tmp")
        assert process.stdout is not None
        with temporary_inventory.open("w", encoding="utf-8") as output:
            for line in process.stdout:
                relative, file_id = line.rstrip("\n").split("\t", 1)
                relative = relative.strip().lstrip("/")
                if not relative:
                    continue
                output.write(relative + "\t" + file_id + "\n")
                add_path(relative, file_id)
                if scanned % 100_000 == 0:
                    elapsed_minutes = (time.perf_counter() - started) / 60
                    print(f"Indexed {scanned:,} Drive entries; candidate samples={len(groups):,}; elapsed={elapsed_minutes:.1f} min", flush=True)
        return_code = process.wait()
        if return_code:
            temporary_inventory.unlink(missing_ok=True)
            raise RuntimeError(f"rclone filename inventory failed with exit code {return_code}")
        temporary_inventory.replace(inventory_paths)
        _atomic_json(inventory_meta_path, expected_inventory_meta)
    complete_ids = sorted(sample_id for sample_id, group in groups.items() if REQUIRED <= group.keys() and sample_id not in duplicate_ids)
    incomplete_count = sum(1 for group in groups.values() if not REQUIRED <= group.keys())
    if len(complete_ids) < sample_count:
        raise ValueError(f"Only {len(complete_ids):,} complete unambiguous RGB/alpha/depth/normal groups found; need {sample_count:,}. Check the rclone remote and source folder.")

    selected_ids = sorted(random.Random(seed).sample(complete_ids, sample_count))
    rows = []
    for sample_id in selected_ids:
        group = groups[sample_id]
        rows.append({"image": str(source_root / group["rgb"]), "mask": str(source_root / group["alpha"]), "depth": str(source_root / group["depth"]), "normal": str(source_root / group["normal"]), "subject_id": f"sample_{sample_id:07d}", "label_source": "SynthHuman/DAViD renderer", "depth_units": "m", "depth_scale_to_m": 0.01, "normal_convention": "camera_xyz_signed"})
    split_ids = sorted(row["subject_id"] for row in rows)
    random.Random(seed).shuffle(split_ids)
    n_test = max(1, round(len(split_ids) * test_fraction))
    n_val = max(1, round(len(split_ids) * val_fraction))
    test_ids = set(split_ids[:n_test])
    val_ids = set(split_ids[n_test:n_test + n_val])
    for row in rows:
        row["partition"] = "test" if row["subject_id"] in test_ids else "val" if row["subject_id"] in val_ids else "train"
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(manifest_path)
    metadata_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    report = {"remote_path": remote_path, "source_root": str(source_root), "entries_scanned": scanned, "candidate_sample_ids": len(groups), "complete_samples": len(complete_ids), "incomplete_samples": incomplete_count, "ambiguous_sample_ids": len(duplicate_ids), "selected_samples": sample_count, "seed": seed, "inventory_minutes": (time.perf_counter() - started) / 60, "manifest": str(manifest_path), "cached_paths": str(inventory_paths)}
    _atomic_json(report_path, report)
    parts = {name: [row for row in rows if row["partition"] == name] for name in ("train", "val", "test")}
    split_text = {name: len(partition) for name, partition in parts.items()}
    print(f"Raw inventory complete: {len(complete_ids):,} complete IDs; {incomplete_count:,} incomplete; {len(duplicate_ids):,} ambiguous; selected={sample_count:,}; split={split_text}", flush=True)
    print(f"Raw manifest saved: {manifest_path}; inventory time={(time.perf_counter()-started)/60:.1f} min", flush=True)
    return rows, parts, report


def _resize_array(array, size, is_mask=False):
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[..., None]
    mode = Image.Resampling.NEAREST if is_mask or array.shape[-1] == 1 else Image.Resampling.BILINEAR
    planes = [np.asarray(Image.fromarray(array[..., channel].astype(np.float32)).resize((size[1], size[0]), mode), dtype=np.float32) for channel in range(array.shape[-1])]
    return np.stack(planes, axis=-1)


def _read_exr(path: Path, kind: str):
    with path.open("rb") as handle:
        prefix = handle.read(16)
    if prefix[:4] != b"\x76\x2f\x31\x01":
        raise ValueError(f"{path} has no valid EXR magic header (first16={prefix.hex()})")
    try:
        import OpenEXR
    except ImportError as error:
        raise RuntimeError('Install OpenEXR in the DGX environment: pip install "OpenEXR>=3.3,<4"') from error
    if not hasattr(OpenEXR, "File"):
        raise RuntimeError("OpenEXR >=3.3 is required for the named-channel decoder")
    with OpenEXR.File(str(path), separate_channels=True) as exr:
        if len(exr.parts) != 1:
            raise ValueError(f"Multipart EXR is unsupported: {path}")
        channels = exr.channels()
        lookup = {name.upper(): name for name in channels}
        if kind == "normal":
            if {"R", "G", "B"} <= lookup.keys():
                keys = ("R", "G", "B")
            elif {"X", "Y", "Z"} <= lookup.keys():
                keys = ("X", "Y", "Z")
            else:
                raise ValueError(f"Normal EXR needs RGB or XYZ channels, found {sorted(channels)}: {path}")
        elif "Z" in lookup:
            keys = ("Z",)
        elif len(channels) == 1:
            keys = (next(iter(channels)).upper(),)
        elif "R" in lookup:
            keys = ("R",)
        else:
            raise ValueError(f"Ambiguous depth EXR channels {sorted(channels)}: {path}")
        planes = []
        for key in keys:
            pixels = channels[lookup[key]].pixels
            if pixels.ndim != 2 or pixels.dtype.kind not in "fiu":
                raise ValueError(f"Invalid EXR channel {key}: {path}")
            planes.append(np.array(pixels, dtype=np.float32, copy=True))
        return np.stack(planes, axis=-1)


def _read_rgb(path: Path, size):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size[1], size[0]), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


class RawSynthHumanDataset(Dataset):
    def __init__(self, rows, input_size=(256, 256), augment=False):
        self.rows = rows
        self.input_size = tuple(input_size)
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def close(self):
        return None

    def __getitem__(self, index):
        augmentation_seed = None
        if isinstance(index, tuple):
            index, augmentation_seed = index
        row = self.rows[index]
        image = _read_rgb(Path(row["image"]), self.input_size)
        raw_depth = _read_exr(Path(row["depth"]), "depth")
        normal = _read_exr(Path(row["normal"]), "normal")
        with Image.open(row["mask"]) as alpha_image:
            alpha = np.asarray(alpha_image.convert("L"))
        mask = _resize_array((alpha > 0).astype(np.uint8) * 255, self.input_size, True)[..., 0] > 0
        raw_depth = _resize_array(raw_depth.astype(np.float32), self.input_size)[..., 0]
        normal = _resize_array(normal.astype(np.float32), self.input_size)
        if normal.shape[-1] != 3:
            raise ValueError(f"Expected 3-channel normals at {row['normal']}; found {normal.shape}")
        valid_depth = np.isfinite(raw_depth) & (raw_depth > 0) & (raw_depth < 1000) & (~np.isclose(raw_depth, 65504.0, rtol=0.0, atol=1.0))
        depth = raw_depth[..., None] * float(row.get("depth_scale_to_m", 0.01))
        mask &= valid_depth & np.isfinite(normal).all(axis=-1)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        normal = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
        if mask.sum() == 0:
            raise ValueError(f"No valid labeled pixels for {row['subject_id']}")
        rng = random.Random(augmentation_seed) if augmentation_seed is not None else random
        if self.augment and rng.random() < 0.5:
            image, depth, normal, mask = [np.flip(array, axis=1).copy() for array in (image, depth, normal, mask)]
            normal[..., 0] *= -1
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        tensors = [torch.from_numpy(array.copy()).permute(2, 0, 1).float() for array in (image, depth, normal)]
        tensors.append(torch.from_numpy(mask[None].astype(np.float32)))
        tensors[2] = F.normalize(tensors[2], dim=0, eps=1e-6)
        tensors[3] *= (torch.linalg.vector_norm(tensors[2], dim=0, keepdim=True) > 0).float()
        return tuple(tensors)


class RealPseudoDepthDataset(Dataset):
    """Unlabeled real RGB images paired with frozen-teacher relative-depth maps."""

    def __init__(
        self, manifest_path: Path, pseudo_depth_root: Path, input_size,
        crop_probability=0.0, crop_zoom_range=(1.25, 2.0),
    ):
        self.manifest_path = Path(manifest_path)
        self.pseudo_depth_root = Path(pseudo_depth_root)
        self.input_size = tuple(int(value) for value in input_size)
        self.crop_probability = float(crop_probability)
        self.crop_zoom_range = tuple(float(value) for value in crop_zoom_range)
        if not 0.0 <= self.crop_probability <= 1.0:
            raise ValueError("crop_probability must be between zero and one")
        if (
            len(self.crop_zoom_range) != 2
            or self.crop_zoom_range[0] <= 1.0
            or self.crop_zoom_range[1] < self.crop_zoom_range[0]
        ):
            raise ValueError("crop_zoom_range must be [minimum, maximum] with 1 < minimum <= maximum")
        self.rows = [
            json.loads(line)
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"Real-image pseudo-depth manifest is empty: {self.manifest_path}")
        missing = []
        for row in self.rows:
            if "image_id" not in row or "image" not in row:
                raise ValueError(f"Malformed row in V9 real-data manifest: {row}")
            for path in (Path(row["image"]), self.pseudo_depth_root / f"{row['image_id']}.npy"):
                if not path.is_file():
                    missing.append(path)
                    if len(missing) == 5:
                        break
            if len(missing) == 5:
                break
        if missing:
            raise FileNotFoundError(
                "V9 real-data files are incomplete; first missing paths: "
                + ", ".join(str(path) for path in missing)
            )

    def __len__(self):
        return len(self.rows)

    def close(self):
        return None

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["image"])
        depth_path = self.pseudo_depth_root / f"{row['image_id']}.npy"
        if not image_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(f"Missing V9 real sample or teacher map: {image_path} | {depth_path}")
        image = _read_rgb(image_path, self.input_size)
        with depth_path.open("rb") as handle:
            pseudo_depth = np.load(handle, allow_pickle=False).astype(np.float32)
        if pseudo_depth.ndim != 2:
            raise ValueError(f"Expected a 2D pseudo-depth map in {depth_path}; got {pseudo_depth.shape}")
        valid = np.isfinite(pseudo_depth)
        if int(valid.sum()) < max(64, int(0.05 * valid.size)):
            raise ValueError(f"Too few valid pseudo-depth pixels in {depth_path}")
        valid = _resize_array(valid.astype(np.uint8), self.input_size, True)[..., 0] > 0
        fill_value = float(np.nanmedian(pseudo_depth))
        pseudo_depth = np.nan_to_num(
            pseudo_depth, nan=fill_value, posinf=fill_value, neginf=fill_value
        )
        pseudo_depth = np.asarray(
            Image.fromarray(pseudo_depth).resize(
                (self.input_size[1], self.input_size[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        if self.crop_probability and random.random() < self.crop_probability:
            height, width = self.input_size
            zoom = random.uniform(*self.crop_zoom_range)
            crop_height = max(32, round(height / zoom))
            crop_width = max(32, round(width / zoom))
            top = random.randint(0, height - crop_height)
            left = random.randint(0, width - crop_width)
            box = (left, top, left + crop_width, top + crop_height)
            image = np.asarray(
                Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
                .crop(box)
                .resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            pseudo_depth = np.asarray(
                Image.fromarray(pseudo_depth).crop(box).resize(
                    (width, height), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            valid = np.asarray(
                Image.fromarray(valid.astype(np.uint8)).crop(box).resize(
                    (width, height), Image.Resampling.NEAREST
                ),
                dtype=np.uint8,
            ) > 0
        pseudo_depth = np.where(valid, pseudo_depth, 0.0)
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return (
            torch.from_numpy(image.copy()).permute(2, 0, 1).float(),
            torch.from_numpy(pseudo_depth[None].copy()).float(),
            torch.from_numpy(valid[None].astype(np.float32)),
        )
