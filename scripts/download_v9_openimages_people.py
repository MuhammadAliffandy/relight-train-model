from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download an Open Images V7 subset containing people for V9 pseudo-depth distillation."
    )
    parser.add_argument("--output", type=Path, default=Path("data/v9_openimages_people"))
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error("--max-samples must be positive")

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as error:
        raise SystemExit("Install FiftyOne first: python -m pip install fiftyone") from error

    args.output.mkdir(parents=True, exist_ok=True)
    fo.config.dataset_zoo_dir = str((args.output / ".fiftyone-zoo").resolve())
    image_dir = args.output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite {manifest_path}. Choose another --output or move the previous dataset."
        )

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=["Person"],
        max_samples=args.max_samples,
        shuffle=True,
        seed=args.seed,
        dataset_name=f"v9-openimages-person-{args.max_samples}-seed{args.seed}",
    )

    rows = []
    for sample in dataset:
        source = Path(sample.filepath)
        image_id = f"oi_{sample.id}"
        target = image_dir / f"{image_id}{source.suffix.lower()}"
        if not target.exists():
            shutil.copy2(source, target)
        rows.append(
            {
                "image_id": image_id,
                "image": str(target.resolve()),
                # Preserve the downloaded sample's FiftyOne metadata for provenance.
                # This does not guarantee that per-image rights or license fields exist.
                "source_metadata": sample.to_dict(),
            }
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    print(f"Downloaded {len(rows):,} person images to {image_dir}")
    print(f"Source metadata and file mapping: {manifest_path}")
    print("FiftyOne metadata is provenance only; verify original image rights at the source before redistribution or commercial use.")


if __name__ == "__main__":
    main()
