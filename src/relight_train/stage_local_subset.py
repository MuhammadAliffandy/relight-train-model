from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm

from relight_train.config import load_config
from relight_train.data import build_raw_manifest


MODALITIES = ("image", "mask", "depth", "normal")


def stage_local_subset(config_path: str | Path) -> None:
    """Select complete samples, copy their four raw modalities with rclone, and verify the copy."""
    config = load_config(config_path)
    data = config["data"]
    output = config["output"]
    source_root: Path = data["source_root"]
    run_dir: Path = output["run_dir"]
    source_root.mkdir(parents=True, exist_ok=True)

    rows, partitions, report = build_raw_manifest(
        data["rclone_source"],
        source_root,
        run_dir,
        int(data["expected_samples"]),
        int(config["experiment"]["seed"]),
        data["inventory_dir"],
    )
    file_list = run_dir / "rclone_files_to_copy.txt"
    file_list.parent.mkdir(parents=True, exist_ok=True)
    relative_files: list[str] = []
    for row in rows:
        for field in MODALITIES:
            path = Path(row[field])
            try:
                relative_files.append(path.relative_to(source_root).as_posix())
            except ValueError as error:
                raise ValueError(f"Manifest path is outside local destination {source_root}: {path}") from error
    if len(relative_files) != len(rows) * len(MODALITIES):
        raise RuntimeError("The copy list does not contain exactly four modality files per selected sample.")
    file_list.write_text("".join(f"{path}\n" for path in relative_files), encoding="utf-8")

    print(
        f"Selected {len(rows):,} complete samples ({len(relative_files):,} files); "
        f"split={ {name: len(part) for name, part in partitions.items()} }; seed={config['experiment']['seed']}",
        flush=True,
    )
    print(f"Destination inside project: {source_root}", flush=True)
    free_gib = shutil.disk_usage(source_root).free / 2**30
    print(f"Free space on destination filesystem: {free_gib:.1f} GiB (raw subset size depends on source files)", flush=True)
    print(f"Manifest and resumable file list: {run_dir}", flush=True)
    if report.get("complete_samples") is not None:
        print(f"Available complete source groups: {report['complete_samples']:,}", flush=True)

    command = [
        "rclone", "copy", data["rclone_source"], str(source_root),
        "--files-from-raw", str(file_list),
        "--progress", "--stats", "10s", "--transfers", "8", "--checkers", "16",
        "--retries", "5", "--low-level-retries", "10",
    ]
    print("Running resumable rclone copy. Rerunning this command continues missing files:", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)

    missing: list[str] = []
    for relative in tqdm(relative_files, desc="Verify local files", unit="file", dynamic_ncols=True):
        path = source_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)
    if missing:
        examples = ", ".join(missing[:8])
        raise RuntimeError(f"Copy verification failed: {len(missing):,} files missing/empty. Examples: {examples}. Rerun the staging command to resume.")

    # Decode a real local example so a complete filename set cannot hide truncated or unreadable files.
    from relight_train.data import RawSynthHumanDataset

    first = next(row for row in rows if row["partition"] == "train")
    sample = RawSynthHumanDataset([first], config["experiment"]["input_size"])[0]
    print(
        f"Verified {len(rows):,} samples / {len(relative_files):,} non-empty files; "
        f"local decode shapes={[tuple(tensor.shape) for tensor in sample]}",
        flush=True,
    )
    print(f"Ready to audit/train locally with: {config_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a deterministic complete SynthHuman subset from rclone into the project")
    parser.add_argument("--config", required=True, help="YAML config defining sample count, seed, source and project-local destination")
    args = parser.parse_args()
    stage_local_subset(args.config)


if __name__ == "__main__":
    main()
