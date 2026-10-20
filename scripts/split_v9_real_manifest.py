from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Create deterministic train/validation manifests for V9 real-image distillation."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/v9_openimages_people/manifest.jsonl"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between zero and one")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Real-image manifest not found: {args.manifest}")

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < 2:
        raise ValueError("At least two real images are required to create train and validation splits")
    if len({row.get("image_id") for row in rows}) != len(rows):
        raise ValueError("Real-image manifest contains duplicate or missing image_id values")

    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    validation_count = max(1, round(len(rows) * args.validation_fraction))
    validation_ids = set(order[:validation_count])
    train_rows = [row for index, row in enumerate(rows) if index not in validation_ids]
    validation_rows = [row for index, row in enumerate(rows) if index in validation_ids]

    for suffix, selected in (("train", train_rows), ("val", validation_rows)):
        target = args.manifest.with_name(f"{args.manifest.stem}_{suffix}.jsonl")
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row, default=str) + "\n" for row in selected),
            encoding="utf-8",
        )
        temporary.replace(target)
        print(f"{suffix}: {len(selected)} images -> {target}")

    metadata = {
        "source_manifest": str(args.manifest.resolve()),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "validation_image_ids": [row["image_id"] for row in validation_rows],
    }
    metadata_path = args.manifest.with_name(f"{args.manifest.stem}_split.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"split metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
