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

## V7 geometry fine-tune from V5

`configs/dgx_fastvit_t8_depth_normals_local50000_seed42_v7_from_v5_geometry_30ep.yaml` starts a separate run from the V5 `best_checkpoint.pt`. V5 and V6 share an architecture; this run applies V6 domain augmentation and the new geometry losses directly to the V5 weights. Full-frame images retain metric depth supervision. Crops gain a scale-invariant log-depth shape loss, so the zoomed crop does not force an absolute-distance target. Mask-valid depth gradients and GT-normal boundaries also receive small extra weights. The existing coarse-output auxiliary loss remains active.

The 30 epochs are a maximum, not a target. Early stopping begins after eight epochs and stops after four further epochs without a validation-score improvement of at least 0.005. Every lower validation score can still update `best_checkpoint.pt`; the best checkpoint is used for held-out evaluation. The separate `v7_geometry_12ep.yaml` config remains available to compare initialization from V6.

Run this after the V6 process has released the GPU:

```bash
cd ~/raid/Clara/relight-train-model
git pull --ff-only origin main
mkdir -p logs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_dgx.sh \
  configs/dgx_fastvit_t8_depth_normals_local50000_seed42_v7_from_v5_geometry_30ep.yaml \
  > logs/v7_from_v5_launcher.log 2>&1 < /dev/null &
```

The launcher prints the timestamped training log path; `tail -f logs/v7_from_v5_launcher.log` shows startup errors, and `./scripts/tail_train.sh` follows the training log. Compare V5, V6, and V7 on the same held-out samples and real photos, with identical preprocessing, masks, and depth color scales. The V7 validation history also records depth-edge, normal-boundary, and scale-invariant depth metrics.

## V8 moderate-crop fine-tune and evaluation

The five-sample held-out control showed that head-anchored crops at 1.5× changed V7 mean normal error from 26.81° on a lower-resolution full frame to 105.16° on the crop. V8 starts again from the same V5 checkpoint as V7 and keeps its architecture, losses, learning rates, 30-epoch cap, and early stopping. This isolates the change in crop training and checkpoint selection. Training samples moderate head-anchored zooms from 1.25×–2.25×; the original V7 crop ranges left the 1.5× region underrepresented. Crops smaller than 128 source pixels high are skipped. Full-frame samples retain metric-depth supervision; cropped samples retain shape and normal supervision.

Each epoch also evaluates fixed 1.5× and 2× crops on 256 validation samples, selected once using the run seed. The selection score is the original full-frame validation score plus `0.5 × mean(crop normal error)/30` plus `0.15 × mean(crop SILog ×100)/5`. Early stopping and `best_checkpoint.pt` use this combined score. `history.json` stores the separate crop metrics; after training, `test_metrics.json` contains full-frame metrics and the same crop metrics on a separate fixed held-out subset. These crop results should be compared with the full-frame results; improvement on synthetic crops still needs a real-photo visual check.

```bash
cd ~/raid/Clara/relight-train-model
git pull --ff-only origin main
mkdir -p logs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_dgx.sh \
  configs/dgx_fastvit_t8_depth_normals_local50000_seed42_v8_moderate_crop_30ep.yaml \
  > logs/v8_moderate_crop_launcher.log 2>&1 < /dev/null &
tail -f logs/v8_moderate_crop_launcher.log
```

## V9: real-image depth distillation pilot

V9 initializes the student from the completed V8 checkpoint and uses a frozen Depth Anything V2 Base model as the teacher. `scripts/prepare_v9_dgx.sh` first stages the exact 5,000-sample SynthHuman subset selected for this V9 run, then installs the optional data/teacher dependencies, downloads up to 1,000 Open Images V7 training images containing people, and writes one teacher depth map per image. The real-image manifest preserves FiftyOne sample metadata for provenance; it does not guarantee per-image license information, so verify image rights at the source before redistribution or commercial use. The downloaded images do not have depth ground truth.

Training interleaves SynthHuman batches (metric depth plus GT normals) with real-image batches (relative teacher-depth shape loss only). The teacher loss is normalized within each image and does not claim metric depth. Real images have no normal supervision; normals continue to be learned from SynthHuman. The pilot starts from V8 at low learning rates, uses 5,000 SynthHuman samples, and early-stops against the existing SynthHuman validation/crop score. That score protects the known geometry behavior but does not measure real-photo accuracy; inspect real images separately and keep any visual evaluation photos out of the distillation pool.

On DGX, first confirm that the V8 checkpoint path in the config exists, then prepare the input data:

```bash
cd ~/raid/Clara/relight-train-model
git pull --ff-only origin main
chmod +x scripts/prepare_v9_dgx.sh
V9_REAL_SAMPLES=1000 ./scripts/prepare_v9_dgx.sh
```

The script can resume pseudo-depth generation: it reuses the downloaded image manifest and skips teacher maps already written. After checking `data/v9_openimages_people/manifest.jsonl`, start the separate pilot run:

```bash
mkdir -p logs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nohup ./scripts/train_dgx.sh \
  configs/dgx_fastvit_t8_depth_normals_local5000_seed42_v9_da2_distill_pilot.yaml \
  > logs/v9_da2_distill_launcher.log 2>&1 < /dev/null &
tail -f logs/v9_da2_distill_launcher.log
```

The V9 run writes its own `runs/fastvit_t8_fpn72_v9_da2_distill_openimages1k_synth5k_seed42_384x512_pilot/` directory and never overwrites V8. Logs include `distill_depth`; `history.json` also records its average per epoch. The trainer's current early-stopping and checkpoint selection use SynthHuman validation and crop metrics, not a pseudo-label metric.

### V9.1: paired real-image crops and real-photo checkpoint selection

V9.1 initializes from the V9 best checkpoint and continues frozen Depth Anything V2 depth distillation. The Open Images manifest is split deterministically into 80% train, 10% validation, and 10% test. Only train images contribute distillation gradients. Validation pseudo-depth agreement selects the checkpoint and drives early stopping; the V9 initialization is the fallback if no epoch improves the real-photo validation score. The separate test split is evaluated once after selecting the checkpoint. These are teacher-agreement metrics, not ground-truth accuracy. SynthHuman depth/normal full-frame and crop metrics remain in every epoch as labeled regression monitors. Real training images receive synchronized crop/zoom transformations with their teacher maps (`crop_probability: 0.5`, zoom 1.25–2.0). The run allows up to 20 epochs, with early stopping after at least 5 epochs and 4 stale epochs; 20 is a ceiling for this pilot, not a claim that every dataset needs exactly 20 epochs.

After V9 has completed and its best checkpoint is present, sync the latest repo. If the Open Images photos and teacher maps already exist, just regenerate the deterministic 80/10/10 split:

```bash
git pull --ff-only origin main
conda run -n relight-dgx --no-capture-output python scripts/split_v9_real_manifest.py \
  --manifest data/v9_openimages_people/manifest.jsonl \
  --validation-fraction 0.1 --test-fraction 0.1 --seed 42
```

Run `prepare_v9_dgx.sh` only if the Open Images photos or Depth Anything V2 maps are missing.

Then launch V9.1 into its own run directory:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nohup ./scripts/train_dgx.sh \
  configs/dgx_fastvit_t8_depth_normals_local5000_seed42_v9_1_da2_distill_openimages_crop.yaml \
  > logs/v9_1_da2_distill_launcher.log 2>&1 < /dev/null &
tail -f logs/v9_1_da2_distill_launcher.log
```
