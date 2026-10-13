from __future__ import annotations

import argparse

import torch

from relight_train.config import load_config
from relight_train.data import RawSynthHumanDataset, build_raw_manifest


def main():
    parser = argparse.ArgumentParser(description="Index and audit the raw SynthHuman source before training")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    rows, parts, report = build_raw_manifest(data["rclone_source"], data["source_root"], config["output"]["run_dir"], data["expected_samples"], config["experiment"]["seed"], data["inventory_dir"])
    print(f"Raw source inventory: {report.get('complete_samples', 'reused manifest')} complete samples; selected={len(rows):,}")
    print("Partition sizes:", {key: len(value) for key, value in parts.items()})
    dataset = RawSynthHumanDataset(parts["train"], config["experiment"]["input_size"], augment=False)
    image, depth, normal, mask = dataset[0]
    valid = depth[0][mask[0] > 0]
    print("sample_id:", parts["train"][0]["subject_id"])
    print("tensor shapes:", {"rgb": tuple(image.shape), "depth_m": tuple(depth.shape), "normal_xyz": tuple(normal.shape), "mask": tuple(mask.shape)})
    print("depth_m p01/p50/p99:", torch.quantile(valid, torch.tensor([0.01, 0.5, 0.99])).tolist())
    print("normal range:", float(normal.min()), float(normal.max()), "mask fraction:", float(mask.mean()))


if __name__ == "__main__":
    main()
