from __future__ import annotations

import hashlib
import io
import json
import random
import shutil
import tarfile
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_packed_manifest(shard_dir: Path, expected_samples: int, expected_seed: int):
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared shard manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    if not manifest.get("complete"):
        ready = sum(len(record.get("rows", [])) for record in manifest.get("shards", []))
        raise RuntimeError(f"Prepared dataset is incomplete ({ready:,}/{expected_samples:,} samples). Finish Colab shard preparation first.")
    if config.get("samples") != expected_samples or config.get("seed") != expected_seed:
        raise ValueError(f"Manifest sample count/seed mismatch: {config.get('samples')}/{config.get('seed')}")
    if config.get("input_size") != [256, 256]:
        raise ValueError(f"Unsupported prepared input size: {config.get('input_size')}")
    if not config.get("selection_sha256"):
        raise ValueError("Manifest has no selection_sha256")

    records = {}
    parts = {"train": [], "val": [], "test": []}
    sample_ids = set()
    for record in manifest.get("shards", []):
        name = record.get("name", "")
        if Path(name).name != name or not name.endswith(".tar"):
            raise ValueError(f"Invalid shard filename: {name!r}")
        if record.get("partition") not in parts:
            raise ValueError(f"Invalid shard partition for {name}")
        if not isinstance(record.get("bytes"), int) or record["bytes"] <= 0 or len(record.get("sha256", "")) != 64:
            raise ValueError(f"Invalid size/checksum metadata for {name}")
        path = shard_dir / name
        if not path.is_file() or path.stat().st_size != record.get("bytes"):
            raise OSError(f"Missing or incomplete shard: {path}")
        records[name] = record
        for row in record.get("rows", []):
            sample_id = row.get("subject_id")
            if not sample_id or sample_id in sample_ids:
                raise ValueError(f"Missing or duplicate sample ID in shard manifest: {sample_id}")
            if row.get("partition") != record["partition"] or row.get("shard") != name:
                raise ValueError(f"Sample-to-shard assignment mismatch for {sample_id}")
            if not row.get("member"):
                raise ValueError(f"Missing archive member for sample {sample_id}")
            sample_ids.add(sample_id)
            parts[record["partition"]].append(row)
    if len(sample_ids) != expected_samples:
        raise ValueError(f"Expected {expected_samples:,} samples; manifest contains {len(sample_ids):,}")
    if any(not parts[key] for key in parts):
        raise ValueError(f"Empty data partition: { {key: len(value) for key, value in parts.items()} }")
    return manifest, records, parts


class PreparedShardDataset(Dataset):
    def __init__(self, rows, records, shard_dir, cache_dir, cache_gib, min_free_gib, augment=False):
        self.rows = rows
        self.records = records
        self.shard_dir = Path(shard_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_limit = int(cache_gib * 1024**3)
        self.min_free = int(min_free_gib * 1024**3)
        self.augment = augment
        self.handles = OrderedDict()
        self.cached = OrderedDict()
        self.cached_bytes = 0
        self.stats = {"copies": 0, "local_hits": 0, "copied_bytes": 0, "copy_seconds": 0.0}
        for candidate in sorted(self.cache_dir.glob("*.tar"), key=lambda item: item.stat().st_mtime):
            record = records.get(candidate.name)
            if record and candidate.stat().st_size == record["bytes"]:
                self.cached[candidate.name] = record["bytes"]
                self.cached_bytes += record["bytes"]
            else:
                candidate.unlink(missing_ok=True)
        self._trim(0)

    def __len__(self):
        return len(self.rows)

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def _trim(self, required):
        while self.cached and self.cached_bytes + required > self.cache_limit:
            name, size = self.cached.popitem(last=False)
            handle = self.handles.pop(name, None)
            if handle is not None:
                handle.close()
            (self.cache_dir / name).unlink(missing_ok=True)
            self.cached_bytes -= size

    def _open_shard(self, name):
        if name in self.handles:
            self.cached.move_to_end(name)
            self.stats["local_hits"] += 1
            return self.handles[name]
        record = self.records[name]
        size = record["bytes"]
        if size > self.cache_limit:
            raise OSError(f"Shard {name} ({size / 2**30:.2f} GiB) exceeds local cache limit")
        target = self.cache_dir / name
        start = time.perf_counter()
        valid = target.is_file() and target.stat().st_size == size
        if valid:
            valid = sha256_file(target) == record["sha256"]
        if not valid:
            if name in self.cached:
                self.cached_bytes -= self.cached.pop(name)
            target.unlink(missing_ok=True)
            self._trim(size)
            if shutil.disk_usage(self.cache_dir).free < size + self.min_free:
                raise OSError("Insufficient local disk for next shard; clear local cache or select a larger filesystem")
            temporary = target.with_suffix(".tar.part")
            try:
                shutil.copyfile(self.shard_dir / name, temporary)
                if temporary.stat().st_size != size or sha256_file(temporary) != record["sha256"]:
                    raise IOError(f"Shard integrity check failed after staging: {name}")
                temporary.replace(target)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            elapsed = time.perf_counter() - start
            self.stats["copies"] += 1
            self.stats["copied_bytes"] += size
            self.stats["copy_seconds"] += elapsed
            print(f"Staged {name}: {size / 2**20:.1f} MiB in {elapsed:.1f}s", flush=True)
        if name not in self.cached:
            self.cached[name] = size
            self.cached_bytes += size
        self.cached.move_to_end(name)
        handle = tarfile.open(target, "r:")
        handle.getmembers()
        self.handles[name] = handle
        return handle

    def __getitem__(self, index):
        row = self.rows[index]
        archive = self._open_shard(row["shard"])
        member = archive.extractfile(row["member"])
        if member is None:
            raise OSError(f"Archive member missing: {row['member']} in {row['shard']}")
        with member, np.load(io.BytesIO(member.read()), allow_pickle=False) as sample:
            image = sample["rgb"].astype(np.float32) / 255.0
            depth = sample["depth"].astype(np.float32)
            normal = sample["normal"].astype(np.float32)
            mask = sample["mask"].astype(np.float32)
        size = (256, 256)
        expected = {"image": (3, *size), "depth": (1, *size), "normal": (3, *size), "mask": (1, *size)}
        for key, array in (("image", image), ("depth", depth), ("normal", normal), ("mask", mask)):
            if array.shape != expected[key] or not np.isfinite(array).all():
                raise ValueError(f"Invalid prepared {key} for {row['subject_id']}")
        if mask.sum() <= 0:
            raise ValueError(f"Empty valid mask for {row['subject_id']}")
        if self.augment and random.random() < 0.5:
            image, depth, normal, mask = [np.flip(array, axis=2).copy() for array in (image, depth, normal, mask)]
            normal[0] *= -1
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]) / np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
        tensors = [torch.from_numpy(array.copy()).float() for array in (image, depth, normal, mask)]
        tensors[2] = F.normalize(tensors[2], dim=0, eps=1e-6)
        return tuple(tensors)


def shard_sample_order(dataset: PreparedShardDataset, seed: int) -> list[int]:
    grouped = {}
    for index, row in enumerate(dataset.rows):
        grouped.setdefault(row["shard"], []).append(index)
    rng = random.Random(seed)
    names = sorted(grouped)
    rng.shuffle(names)
    order = []
    for name in names:
        rng.shuffle(grouped[name])
        order.extend(grouped[name])
    return order
