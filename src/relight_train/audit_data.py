from __future__ import annotations

import argparse

import torch

from relight_train.config import load_config
from relight_train.data import PreparedShardDataset, load_packed_manifest


def main():
    parser = argparse.ArgumentParser(description="Audit the prepared SynthHuman shards before training")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    manifest, records, parts = load_packed_manifest(data["shard_dir"], data["expected_samples"], data["expected_seed"])
    print(f"Complete: {manifest['complete']}; samples={sum(map(len, parts.values())):,}; shards={len(records)}")
    print("Partition sizes:", {key: len(value) for key, value in parts.items()})
    dataset = PreparedShardDataset(parts["train"], records, data["shard_dir"], data["cache_dir"], data["cache_gib"], data["min_free_gib"], augment=False)
    try:
        image, depth, normal, mask = dataset[0]
        valid = depth[0][mask[0] > 0]
        print("sample_id:", parts["train"][0]["subject_id"])
        print("tensor shapes:", {"rgb": tuple(image.shape), "depth_m": tuple(depth.shape), "normal_xyz": tuple(normal.shape), "mask": tuple(mask.shape)})
        print("depth_m p01/p50/p99:", torch.quantile(valid, torch.tensor([0.01, 0.5, 0.99])).tolist())
        print("normal range:", float(normal.min()), float(normal.max()), "mask fraction:", float(mask.mean()))
        print("shard cache stats:", dataset.stats)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
