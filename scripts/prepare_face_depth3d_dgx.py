#!/usr/bin/env python3
"""Copy Face-Depth-3D Parquet from xxdrive and unpack its relative labels."""

import argparse
import io
import json
import os
import re
import time
from pathlib import Path


EXPECTED_FILES = 22
EXPECTED_ROWS = 6872
COPY_CHUNK = 16 * 1024 * 1024


def log(message):
    print(message, flush=True)


def duration(seconds):
    if seconds is None:
        return "?"
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def files_under(root):
    return sorted(root.rglob("*.parquet"))


def check_file_list(files, expected):
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} Parquet files, found {len(files)}")
    if any(path.stat().st_size == 0 for path in files):
        raise RuntimeError("At least one Parquet file is empty")


def copy_parquet(source, destination, expected_files):
    source_files = files_under(source)
    check_file_list(source_files, expected_files)
    destination.mkdir(parents=True, exist_ok=True)

    total = sum(path.stat().st_size for path in source_files)
    done = 0
    for path in source_files:
        target = destination / path.relative_to(source)
        partial = target.with_name(target.name + ".part")
        size = path.stat().st_size
        if target.is_file() and target.stat().st_size == size:
            done += size
        elif partial.is_file():
            done += min(partial.stat().st_size, size)

    log(f"Copying {len(source_files)} files, {total / 2**30:.2f} GiB; already present {done / 2**30:.2f} GiB")
    started = time.monotonic()
    transferred = 0
    last_report = 0.0

    for number, path in enumerate(source_files, 1):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        size = path.stat().st_size
        if target.is_file() and target.stat().st_size == size:
            log(f"[{number}/{len(source_files)}] Reusing {target.name}")
            continue
        if target.exists():
            raise RuntimeError(f"Existing destination has the wrong size: {target}")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise RuntimeError(f"Partial file exceeds source size: {partial}")
        log(f"[{number}/{len(source_files)}] Copying {path.name} from {offset / 2**20:.1f} MiB")
        with path.open("rb") as src, partial.open("ab") as dst:
            src.seek(offset)
            while True:
                block = src.read(COPY_CHUNK)
                if not block:
                    break
                dst.write(block)
                done += len(block)
                transferred += len(block)
                now = time.monotonic()
                if now - last_report >= 10:
                    elapsed = max(now - started, 1e-6)
                    rate = transferred / elapsed
                    eta = (total - done) / rate if rate > 0 else None
                    log(
                        f"Copy {done / total * 100:.1f}% | {done / 2**30:.2f}/{total / 2**30:.2f} GiB"
                        f" | {rate / 2**20:.1f} MiB/s | ETA {duration(eta)}"
                    )
                    last_report = now
            dst.flush()
            os.fsync(dst.fileno())
        if partial.stat().st_size != size:
            raise RuntimeError(f"Copy size mismatch for {path.name}: {partial.stat().st_size} != {size}")
        partial.replace(target)
    log(f"Copy complete: {len(source_files)} Parquet files in {destination}")


def image_bytes(value, column, sample_id):
    if not isinstance(value, dict) or not value.get("bytes"):
        raise RuntimeError(f"{sample_id}: {column} has no embedded image bytes")
    return value["bytes"]


def invert_inferno(rgb, palette):
    import numpy as np

    colors, inverse = np.unique(rgb.reshape(-1, 3), axis=0, return_inverse=True)
    levels = np.empty(len(colors), dtype=np.uint8)
    errors = np.empty(len(colors), dtype=np.float32)
    for start in range(0, len(colors), 2048):
        stop = min(start + 2048, len(colors))
        delta = colors[start:stop, None, :].astype(np.float32) - palette[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        nearest = np.argmin(squared, axis=1)
        levels[start:stop] = nearest.astype(np.uint8)
        errors[start:stop] = np.sqrt(squared[np.arange(stop - start), nearest])
    index_map = levels[inverse].reshape(rgb.shape[:2])
    pixel_errors = errors[inverse]
    return index_map, float(pixel_errors.mean()), float((pixel_errors > 5).mean())


def atomic_bytes(path, content):
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def extract_parquet(parquet_root, output_root, expected_files, expected_rows):
    try:
        import numpy as np
        import pyarrow.parquet as pq
        from matplotlib import colormaps
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Extraction needs pyarrow, matplotlib, numpy and Pillow in relight-dgx") from error

    files = files_under(parquet_root)
    check_file_list(files, expected_files)
    counts = []
    required_columns = {"image", "output_mask", "glb_3d_file"}
    for path in files:
        parquet = pq.ParquetFile(path)
        if not required_columns.issubset(parquet.schema_arrow.names):
            raise RuntimeError(f"Missing required columns in {path}")
        counts.append(parquet.metadata.num_rows)
    total_rows = sum(counts)
    if total_rows != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {total_rows}")

    image_dir = output_root / "images"
    label_dir = output_root / "inferno_index"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    palette = (colormaps["inferno"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.float32)
    manifest_rows = []
    done = 0
    started = time.monotonic()
    log(f"Extracting {total_rows} samples at original resolution; inferno index is relative, not meters")

    for file_number, path in enumerate(files, 1):
        parquet = pq.ParquetFile(path)
        row_number = 0
        for batch in parquet.iter_batches(batch_size=8, columns=sorted(required_columns)):
            for row in batch.to_pylist():
                sample_id = f"{path.stem}_{row_number:05d}"
                photo_bytes = image_bytes(row["image"], "image", sample_id)
                mask_bytes = image_bytes(row["output_mask"], "output_mask", sample_id)
                with Image.open(io.BytesIO(photo_bytes)) as photo:
                    photo_size = photo.size
                    photo_format = photo.format
                extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(photo_format)
                if extension is None:
                    raise RuntimeError(f"{sample_id}: unsupported source image format {photo_format}")
                with Image.open(io.BytesIO(mask_bytes)) as mask:
                    if mask.mode != "RGB" or mask.size != photo_size:
                        raise RuntimeError(f"{sample_id}: mask format/size {mask.mode}, {mask.size} vs photo {photo_size}")
                    mask_rgb = np.asarray(mask, dtype=np.uint8)
                index_map, mean_error, bad_fraction = invert_inferno(mask_rgb, palette)
                if mean_error > 4 or bad_fraction > 0.02:
                    raise RuntimeError(
                        f"{sample_id}: output_mask does not match inferno: "
                        f"mean error={mean_error:.2f}, error>5={bad_fraction:.2%}"
                    )
                image_path = image_dir / f"{sample_id}{extension}"
                label_path = label_dir / f"{sample_id}.png"
                if not image_path.exists():
                    atomic_bytes(image_path, photo_bytes)
                if not label_path.exists():
                    buffer = io.BytesIO()
                    Image.fromarray(index_map, mode="L").save(buffer, format="PNG")
                    atomic_bytes(label_path, buffer.getvalue())
                mesh = row["glb_3d_file"]
                match = re.match(r"^(task_[^ ]+)", mesh or "")
                manifest_rows.append({
                    "sample_id": sample_id,
                    "image": str(image_path.resolve()),
                    "inferno_index": str(label_path.resolve()),
                    "glb_3d_file": mesh,
                    "group_id": match.group(1) if match else sample_id,
                    "width": photo_size[0],
                    "height": photo_size[1],
                    "source_parquet": path.name,
                    "source_row": row_number,
                })
                row_number += 1
                done += 1
                if done % 50 == 0 or done == total_rows:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    rate = done / elapsed
                    log(
                        f"Extract {done}/{total_rows} ({done / total_rows * 100:.1f}%)"
                        f" | {rate:.1f} samples/s | ETA {duration((total_rows - done) / rate)}"
                    )
        if row_number != counts[file_number - 1]:
            raise RuntimeError(f"Row count mismatch while reading {path}")
        log(f"[{file_number}/{len(files)}] Finished {path.name}: {row_number} samples")

    manifest_path = output_root / "manifest.jsonl"
    temporary = manifest_path.with_name(manifest_path.name + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(manifest_path)
    summary = {
        "parquet_files": len(files),
        "samples": len(manifest_rows),
        "label_encoding": "nearest matplotlib inferno index, uint8 0..255",
        "label_units": "relative index; not meters",
        "manifest": str(manifest_path.resolve()),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(f"READY: {len(manifest_rows)} pairs; manifest={manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=Path.home() / "xxdrive/AppleCapstoneRelighting/datasets/face_depth_3d",
    )
    parser.add_argument("--output", type=Path, default=Path("data/face_depth_3d"))
    parser.add_argument("--copy-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--expected-files", type=int, default=EXPECTED_FILES)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    args = parser.parse_args()
    if args.copy_only and args.extract_only:
        parser.error("--copy-only and --extract-only cannot be combined")
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    parquet_root = output / "parquet"
    if not args.extract_only:
        if not source.is_dir():
            raise FileNotFoundError(f"xxdrive dataset directory not found: {source}")
        copy_parquet(source, parquet_root, args.expected_files)
    if not args.copy_only:
        extract_parquet(parquet_root, output, args.expected_files, args.expected_rows)


if __name__ == "__main__":
    main()
