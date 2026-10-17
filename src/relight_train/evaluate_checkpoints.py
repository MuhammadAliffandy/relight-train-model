from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from relight_train.data import RawSynthHumanDataset
from relight_train.model import PortraitGeometryModel
from relight_train.objectives import evaluate


class CoarseOutput(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image, return_coarse=True)[2:]


def load_model(state, device):
    exp = state["experiment"]
    model = PortraitGeometryModel(
        backbone=exp["backbone"],
        pretrained=False,
        fpn_width=int(exp["fpn_width"]),
        fpn_fusion=exp.get("fpn_fusion", "add"),
        decoder_refinement=exp.get("decoder_refinement", "none"),
        refinement_width=int(exp.get("refinement_width", 16)),
        fullres_width=int(exp.get("fullres_width", 24)),
        rgb_full_width=int(exp.get("rgb_full_width", 8)),
        guarded_refinement=bool(exp.get("guarded_refinement", False)),
        residual_depth_limit=float(exp.get("residual_depth_limit", 0.05)),
        residual_normal_limit=float(exp.get("residual_normal_limit", 0.15)),
        grayscale_refinement=bool(exp.get("grayscale_refinement", False)),
    )
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), exp


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints on one manifest partition")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--partition", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--include-coarse", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers nonnegative")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selection_sha256 = hashlib.sha256(
        "\n".join(sorted(row["subject_id"] for row in rows)).encode("utf-8")
    ).hexdigest()
    selected = [row for row in rows if row["partition"] == args.partition]
    if not selected:
        raise ValueError(f"No {args.partition} rows in {args.manifest}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    expected_size = None
    for item in args.checkpoint:
        label, separator, path_string = item.partition("=")
        if not separator or not label or not path_string or label in results:
            parser.error("Each --checkpoint must be a unique LABEL=PATH")
        checkpoint = Path(path_string)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("selection_sha256") != selection_sha256:
            raise ValueError(f"{checkpoint} uses a different sample selection from {args.manifest}")
        model, exp = load_model(state, device)
        input_size = tuple(int(value) for value in exp["input_size"])
        if expected_size is None:
            expected_size = input_size
        elif input_size != expected_size:
            raise ValueError(f"Input size mismatch: {input_size} vs {expected_size}")
        loader = DataLoader(
            RawSynthHumanDataset(selected, input_size=input_size, augment=False),
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        metrics = evaluate(model, loader, device, detail_metrics=True)
        results[label] = {"checkpoint": str(checkpoint), "final_metrics": metrics}
        if args.include_coarse:
            results[label]["coarse_metrics"] = evaluate(
                CoarseOutput(model), loader, device, detail_metrics=True
            )
        print(label, json.dumps(results[label], indent=2), flush=True)
        del model, loader, state

    report = {
        "manifest": str(args.manifest),
        "partition": args.partition,
        "sample_count": len(selected),
        "input_size": expected_size,
        "edge_definition": "GT Sobel magnitude top 10% within a 3x3-valid interior; 3x3 edge neighborhood",
        "depth_silog_definition": "100 * population_std(log(pred_depth) - log(gt_depth)) per image",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
