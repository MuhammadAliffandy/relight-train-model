# Apple Capstone Relighting

The active workflow separates Colab data inspection from DGX training:

- `colab/notebooks/` contains the Colab source-data inspection/acquisition notebooks. Previous shard-based experiments are archived under `colab/archive/shards/`.
- `colab/scripts/` contains helper scripts used by the Colab notebooks.
- `src/relight_train/` contains the server-native Python training and data-audit code for DGX.
- `scripts/` contains DGX Conda setup, dataset audit, training, and log-following shell scripts.

The current experiment predicts **metric depth and surface normals** from RGB with a FastViT-T8 encoder and compact FPN. Albedo is not trained. DGX indexes filenames through rclone, then reads RGB, alpha, depth EXR, and normal EXR directly from the mounted raw SynthHuman folder and preprocesses each sample in the data loader. No TAR shard preparation is required for the active DGX workflow.

## DGX setup

Requirements: Linux DGX with NVIDIA driver/CUDA support, Conda, rclone remote `xxdrive:`, and the existing SynthHuman source folder in Google Drive.

Clone into the project/work directory you want to use on the DGX (for example, `~/Projects`), not into `/raid`:

```bash
mkdir -p ~/Projects
cd ~/Projects
git clone https://github.com/MuhammadAliffandy/relight-train-model.git
cd relight-train-model
chmod +x scripts/*.sh
./scripts/setup_dgx.sh
```

The default config expects the Google Drive mount at `~/xxdrive` and raw dataset at:
`~/xxdrive/AppleCapstoneRelighting/dataset/sources/SynthHuman`.

Build/reuse the recursive filename index and decode a real sample before training:

```bash
./scripts/audit_dgx.sh
```

The first audit indexes filenames via `rclone lsf`; this reads metadata only. The trainer then reads the original files through the rclone mount and preprocesses them on demand. Direct reads can make indexing/training I/O-bound, so the first run reports inventory time and every batch reports data wait.

Run a short raw-data throughput pilot first (500 selected samples, one epoch, separate run/checkpoints):

```bash
./scripts/audit_dgx.sh configs/dgx_fastvit_t8_depth_normals_raw500_seed42_pilot.yaml
./scripts/train_dgx.sh configs/dgx_fastvit_t8_depth_normals_raw500_seed42_pilot.yaml
```

After the measured loader rate looks practical, start the 50k experiment:

```bash
./scripts/train_dgx.sh
```

From a second SSH session, follow the latest log:

```bash
./scripts/tail_train.sh
```

Training can run detached from SSH. Logs, checkpoints, and model cache default to folders inside the project checkout:

```bash
mkdir -p logs
nohup ./scripts/train_dgx.sh > logs/launcher.log 2>&1 < /dev/null &
```

Then reconnect and run `./scripts/tail_train.sh`. Checkpoints and history are saved in the configured DGX run directory and copied periodically to the configured rclone path. Rerun the same command to resume from `last_checkpoint.pt`; use a new run directory for a fresh run.

## Configuration

Edit `configs/dgx_fastvit_t8_depth_normals_50k.yaml` to change the mount paths, run directory, batch size, epochs, or worker count. The recursive index selects 50,000 complete, unambiguous RGB/alpha/depth/normal groups using a fixed seed and saves the exact local manifest and inventory report. The audit refuses to continue if fewer than 50,000 usable groups are found.

For rclone mount, keep its VFS cache on `/raid` and set an explicit maximum size, so reading raw dataset files cannot fill the system disk. See `docs/dgx-direct-data.md` for the mount command and expected I/O tradeoffs.

## Option 2: copy 30,000 samples into the project

If repeated reads through the rclone mount are too slow, stage a deterministic 30,000-sample subset into the cloned project. This copies the four original files per sample (RGB, alpha, depth EXR, normal EXR), about 120,000 files total. Data is stored under `data/synthhuman_local30000/`; its index/manifest and training outputs are also project-local and ignored by Git. The required disk space depends on source file sizes, so check free space before copying.

```bash
./scripts/stage_dgx_subset.sh
```

`rclone --progress` reports transfer progress. The command verifies each copied file is non-empty and decodes one sample. The exact selection and file list are saved under the project, so rerunning after an interruption resumes missing files and keeps the same samples. Split is 70/15/15 with seed 42.

After staging:

```bash
./scripts/audit_dgx.sh configs/dgx_fastvit_t8_depth_normals_local30000_seed42.yaml
./scripts/train_dgx.sh configs/dgx_fastvit_t8_depth_normals_local30000_seed42.yaml
```

This local-copy option uses rclone only for staging; training reads the copied files from the project disk. The separate 50k streaming path still uses the rclone mount and its bounded VFS cache on `/raid`.
