# Apple Capstone Relighting

The repository has two workflows:

- `colab/notebooks/` contains notebooks for downloading/inspecting the source data, preparing the existing SynthHuman files into shards, and the Colab training workflow.
- `colab/scripts/` contains helper scripts used by the Colab notebooks.
- `src/relight_train/` contains the server-native Python training and data-audit code for DGX.
- `scripts/` contains DGX Conda setup, dataset audit, training, and log-following shell scripts.

The active experiment predicts **metric depth and surface normals** from RGB with a FastViT-T8 encoder and compact FPN. Albedo is not trained in this experiment. Dataset preparation runs in Colab against the existing Drive data; DGX reads the resulting prepared TAR shards through the rclone mount and stages each shard into a local bounded cache before training.

## DGX setup

Requirements: Linux DGX with NVIDIA driver/CUDA support, Conda, rclone remote `xxdrive:`, and the prepared 50k shard directory in Google Drive.

```bash
git clone https://github.com/MuhammadAliffandy/relight-train-model.git
cd relight-train-model
chmod +x scripts/*.sh
./scripts/setup_dgx.sh
```

The default config expects the Google Drive mount at `~/xxdrive` and the dataset at:
`~/xxdrive/AppleCapstoneRelighting/dataset/prepared_shards/synthhuman_50000_seed42_256x256_v1`.
It stages shards into a bounded 48 GiB local cache on `/raid`, enough for the expected prepared dataset size so epochs can reuse local files rather than reread Drive.

Check that all shards and the complete manifest are readable before training:

```bash
./scripts/audit_dgx.sh
```

Start a foreground run (it prints a live per-iteration progress bar and writes a timestamped log):

```bash
./scripts/train_dgx.sh
```

From a second SSH session, follow the latest log:

```bash
./scripts/tail_train.sh
```

Training can run detached from the SSH connection:

```bash
mkdir -p "$HOME/raid/Clara/relight_logs"
nohup ./scripts/train_dgx.sh > "$HOME/raid/Clara/relight_logs/launcher.log" 2>&1 < /dev/null &
```

Then reconnect and run `./scripts/tail_train.sh`. The trainer saves resumable checkpoints and history under the configured DGX run directory and copies them periodically to the configured rclone path. If the process stops, rerunning the same command resumes from `last_checkpoint.pt`; use a new run directory for a fresh run.

## Configuration

Edit `configs/dgx_fastvit_t8_depth_normals_50k.yaml` to change dataset mount, local shard cache, run directory, batch size, epochs, or the optional Drive backup remote. The trainer refuses incomplete manifests and validates shard sizes and SHA-256 checksums as archives are staged.

The preparation notebook previously reported a 500-sample benchmark. Check the current `manifest.json` on Drive; complete the remaining shard preparation in Colab before running `audit_dgx.sh`. The audit refuses an incomplete manifest.
