# Apple Capstone Relighting

The active workflow separates Colab data inspection from DGX training:

- `colab/notebooks/` contains the Colab source-data inspection/acquisition notebooks. Previous shard-based experiments are archived under `colab/archive/shards/`.
- `colab/scripts/` contains helper scripts used by the Colab notebooks.
- `src/relight_train/` contains the server-native Python training and data-audit code for DGX.
- `scripts/` contains DGX Conda setup, dataset audit, training, and log-following shell scripts.

The current experiment predicts **metric depth and surface normals** from RGB with a FastViT-T8 encoder and compact FPN. Albedo is not trained. DGX indexes filenames through rclone, then reads RGB, alpha, depth EXR, and normal EXR directly from the mounted raw SynthHuman folder and preprocesses each sample in the data loader. No TAR shard preparation is required for the active DGX workflow.

For a Colab comparison, `colab/notebooks/colab_synthhuman_6chunk_pilot.ipynb` stages six extracted Drive chunks (up to 30k samples) to `/content`, then runs raw-file preprocessing and a one-epoch depth+normal training pilot. It reports chunk-copy progress separately from per-batch training progress. The raw local copy is temporary; manifests and checkpoints are saved to Drive.

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

The source dataset remains in the Drive mounted at `~/xxdrive`. If repeated reads through the mount are too slow, stage a deterministic 30,000-sample subset from the same Drive remote (`xxdrive:`) into the cloned project. This copies the four original files per sample (RGB, alpha, depth EXR, normal EXR), about 120,000 files total. Data is stored under `data/synthhuman_local30000/`; its index/manifest and training outputs are also project-local and ignored by Git. The required disk space depends on source file sizes, so check free space before copying.

```bash
./scripts/stage_dgx_subset.sh
```

`rclone --progress` reports transfer progress. The command verifies each copied file is non-empty and decodes one sample. The exact selection and file list are saved under the project, so rerunning after an interruption resumes missing files and keeps the same samples. Split is 70/15/15 with seed 42.

After staging:

```bash
./scripts/audit_dgx.sh configs/dgx_fastvit_t8_depth_normals_local30000_seed42.yaml
./scripts/train_dgx.sh configs/dgx_fastvit_t8_depth_normals_local30000_seed42.yaml
```

For the 50k main run, use `configs/dgx_fastvit_t8_depth_normals_local50000_seed42.yaml`. It reuses the same local raw store and inventory as the 30k pilot, selects a deterministic 50k subset with seed 42 (including the pilot's 30k IDs), and copies only missing files. It uses a new run directory and one 100-epoch AdamW/cosine schedule; start once and select `best_checkpoint.pt` by validation score. The epoch count is evidence-informed by DAViD's 100-epoch training on SynthHuman, but the architecture and training setup differ, so treat validation as the stopping/model-selection signal. The training script supports both cosine and plateau schedulers.

```bash
./scripts/stage_dgx_subset.sh configs/dgx_fastvit_t8_depth_normals_local50000_seed42.yaml
./scripts/audit_dgx.sh configs/dgx_fastvit_t8_depth_normals_local50000_seed42.yaml
CUDA_VISIBLE_DEVICES=0 nohup ./scripts/train_dgx.sh configs/dgx_fastvit_t8_depth_normals_local50000_seed42.yaml > logs/train_local50000_cuda0.log 2>&1 < /dev/null &
```

This local-copy option uses rclone only for staging; training reads the copied files from the project disk. The separate 50k streaming path still uses the rclone mount and its bounded VFS cache on `/raid`.

## V7 geometry fine-tune from V6

`configs/dgx_fastvit_t8_depth_normals_local50000_seed42_v7_geometry_12ep.yaml` starts a separate 12-epoch run from the V6 `best_checkpoint.pt`. It keeps the V6 architecture and domain augmentation, with a smaller encoder learning rate. Full-frame images retain metric depth supervision. Crops gain a scale-invariant log-depth shape loss, so the zoomed crop does not force an absolute-distance target. Mask-valid depth gradients and GT-normal boundaries also receive small extra weights. The existing coarse-output auxiliary loss remains active.

Run this after the V6 checkpoint exists and the V6 process has released the GPU:

```bash
cd ~/raid/Clara/relight-train-model
git pull --ff-only origin main
mkdir -p logs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_dgx.sh \
  configs/dgx_fastvit_t8_depth_normals_local50000_seed42_v7_geometry_12ep.yaml \
  > logs/v7_geometry_launcher.log 2>&1 < /dev/null &
```

The launcher prints the timestamped training log path; `tail -f logs/v7_geometry_launcher.log` shows startup errors, and `./scripts/tail_train.sh` follows the training log. Compare V6 and V7 on the same held-out samples and real photos, with identical preprocessing, masks, and depth color scales. The V7 validation history also records depth-edge, normal-boundary, and scale-invariant depth metrics.
