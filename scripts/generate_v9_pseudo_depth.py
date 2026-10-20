from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Create frozen Depth Anything V2 relative-depth targets for V9 real images."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/v9_openimages_people/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/v9_openimages_people/pseudo_depth"))
    parser.add_argument("--teacher", default="depth-anything/Depth-Anything-V2-Base-hf")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of rows; 0 processes all")
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Image manifest not found: {args.manifest}")
    if args.limit < 0:
        parser.error("--limit must be zero or positive")
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as error:
        raise SystemExit("Install teacher dependencies first: python -m pip install 'transformers>=4.45,<5' safetensors") from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Pseudo-depth generation for V9 must run on a CUDA DGX node.")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading frozen teacher {args.teacher} on {torch.cuda.get_device_name(0)}", flush=True)
    processor = AutoImageProcessor.from_pretrained(args.teacher)
    teacher = AutoModelForDepthEstimation.from_pretrained(args.teacher).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        image_id = row["image_id"]
        output_path = args.output / f"{image_id}.npy"
        if output_path.is_file():
            print(f"{index}/{len(rows)}: exists {image_id}", flush=True)
            continue
        image_path = Path(row["image"])
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            original_size = (image.height, image.width)
            inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            predicted = teacher(**inputs).predicted_depth
            predicted = F.interpolate(
                predicted[:, None].float(), size=original_size, mode="bicubic", align_corners=False
            )[0, 0]
        depth = predicted.float().cpu().numpy().astype(np.float32)
        valid = np.isfinite(depth)
        if valid.mean() < 0.95 or float(np.nanstd(depth)) < 1e-6:
            raise ValueError(f"Teacher returned an invalid or nearly constant depth map for {image_id}")
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        temporary = output_path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, depth, allow_pickle=False)
        temporary.replace(output_path)
        print(
            f"{index}/{len(rows)}: {image_id} {depth.shape[1]}x{depth.shape[0]} "
            f"range=[{float(np.percentile(depth, 2)):.4f}, {float(np.percentile(depth, 98)):.4f}] "
            f"elapsed={(time.perf_counter()-started)/60:.1f}m",
            flush=True,
        )

    metadata = {
        "teacher": args.teacher,
        "depth_type": "relative inverse-depth-style teacher prediction; larger values indicate nearer surfaces; not metric ground truth",
        "source_manifest": str(args.manifest.resolve()),
        "processed_rows": len(rows),
        "output_dir": str(args.output.resolve()),
    }
    (args.output / "teacher_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Pseudo-depth maps saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
