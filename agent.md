# Project Context: Apple Capstone Portrait Relighting

## Goal and current status
- Build an iOS tool for independent creators to relight a portrait/body crop with controllable key, fill, and rim lights.
- The model predicts human-region geometry and reflectance; Swift/Vision handles person segmentation and compositing.
- Current repository is a research prototype: active Colab notebooks are in `colab/notebooks/`; DGX training/evaluation code is in `src/relight_train/`. It has not yet been trained on the complete 50k raw sample selection on DGX or validated on an iPhone.

## Current model hypothesis
- Model contract: one ImageNet-normalized RGB crop in; aligned depth, camera-frame surface normals, and diffuse albedo maps out.
- Baseline remains ImageNet-pretrained FastViT-T8 (`timm`) with a compact 48-channel depthwise-separable FPN and three dense task heads. MobileNetV3-Large is an optional backbone ablation, not a selected replacement.
- FastViT has published iPhone/Core ML classification measurements; these do not predict latency of this custom dense multi-head graph. MobileNetV3 has mobile CPU and dense-prediction evidence on other platforms. Compare both on identical training data and target iPhones before selecting.
- Bilinear resize + convolution only; avoid transposed convolutions for the intended Core ML path.
- Outputs: positive relative depth unless calibration metadata establishes metric scale; unit-length normals; albedo in [0, 1]. The notebook may use metric depth only when the dataset documents calibrated units; SynthHuman absolute depth is converted from cm to m. Do not call Sapiens monocular depth metric depth unless scale is independently calibrated.
- Keep FastViT-T8 as the research baseline. Any alternative (including MobileNetV3) must earn a switch through matched quality, Core ML compatibility, and target-iPhone latency measurements; neither paper claims guarantee 60 FPS for our graph.

## Colab and DGX workflow
- Report dataset chunk totals and incomplete paired samples; report train/validation duration, epoch wall time, throughput, and ETA, and persist timing in the run history on Drive. Do not dump or visualize every sample by default.
- Current DGX experiment plan: deterministic 50,000 SynthHuman samples, 5 initial epochs, depth + normals only. Keep a 500-sample I/O throughput pilot in a separate run directory; do not reuse its checkpoints.
- In Colab, mount Google Drive at the start of each fresh runtime and store dataset, manifest, caches, checkpoints, histories, metrics, and exported models under `MyDrive/AppleCapstoneRelighting`; reconnecting the runtime does not preserve local `/content` files.
- Keep active Colab notebooks under `colab/notebooks/` and their helpers under `colab/scripts/`. Archive old shard experiments under `colab/archive/shards/`. Keep DGX Python modules in `src/relight_train/`, environment/config files at the repository root and `configs/`, and server commands in `scripts/`.
- Keep acquisition notebooks separate from training. `colab/notebooks/download_synthhuman_full.ipynb` is the optional full-release downloader (~330 GB); `colab/notebooks/download_synthhuman_pilot_50k.ipynb` is optional. The active workflow reuses existing Drive source data and must not download the original release again.
- Previous Colab shard preparation/training is archived and is not part of the active DGX workflow.
- DGX training is Python/Conda based, never notebook based. Run `scripts/setup_dgx.sh`, verify data with `scripts/audit_dgx.sh`, start or resume with `scripts/train_dgx.sh`, and follow output with `scripts/tail_train.sh`. The streaming entry point indexes existing raw files through rclone, selects 50k complete, unambiguous modality groups, and preprocesses images/EXRs on demand from the mount. A separate `scripts/stage_dgx_subset.sh` copies a deterministic 30k complete subset (RGB/alpha/depth/normal) into the ignored `data/synthhuman_local30000/` folder in the cloned project and can resume after interruption. Keep clone and staged subset in the project/work folder, not `/raid`; put only the bounded VFS cache on `/raid` for streaming.
- Colab Pro does not guarantee a specific GPU or uninterrupted runtime; checkpoints on Drive support recovery, but the actual runtime/session must be started and authorized by the user. Core ML prediction parity requires macOS even though conversion can run on Linux.
- Training writes a mid-epoch resumable checkpoint locally on DGX every 10 minutes, with deterministic sample order, batch position, optimizer/scheduler/scaler and RNG state; checkpoint/history copies go to the configured rclone remote.
- Log CUDA availability/version, GPU name/count/capability and memory before training; show a per-batch live progress indicator, data-wait time and current losses during each epoch. DGX baseline initially uses one GPU until data throughput is measured.
- Keep Drive authentication, selected-file API indexing, preprocessing/cache, sample audit, model, training and evaluation in distinct Colab cells. The active experiment selects 50,000 sample IDs with a fixed seed from the existing Drive folder and trains depth/normals for 5 epochs. The split is by sample IDs because real subject identities are unavailable.

## Data contract
- Training uses a JSONL manifest with one record per sample: `image`, `depth`, `normal`, optional `albedo`, `mask`, and `subject_id` (paths may be absolute or relative to `DATA_ROOT`).
- Depth and normals can be initialized from Sapiens outputs, with provenance, coordinate convention, preprocessing, validity masks, and depth scale documented. Sapiens pseudo-labels are not ground truth; reserve independently annotated or captured relighting data for evaluation.
- Albedo requires suitable intrinsic-image labels or controlled/multi-light captures. RGB copies, brightness-scaled inputs, and random tensors are not valid albedo labels.
- Split by subject/identity or capture session before training. Never split adjacent frames/images from a subject across train and validation.
- Invalid/out-of-person pixels are excluded with masks. Augmentations must transform image and all spatial targets consistently; normals must be transformed for flips/rotations.

## Evaluation gates
- Geometry: masked depth MAE/RMSE and scale-aware or scale-invariant metrics as appropriate; median angular error and threshold accuracy for normals.
- Albedo: masked MAE, PSNR, SSIM plus a simple input-copy baseline. Interpret scores only when independent reference albedo exists.
- Relighting: evaluate rendered target-light images against held-out ground truth using PSNR/SSIM/LPIPS where the capture setup supplies paired targets; include visual review for skin/clothing boundaries, shadows, and highlights.
- Deployment: compare eager PyTorch and converted Core ML output numerically; report package size, peak memory, warm/cold latency, p50/p95 latency, compute-unit placement, device model, OS, input size, and sustained frame rate. Do not infer device performance from Colab/Mac timings.
- Do not claim production quality from proxy labels or a tiny smoke-test subset.

## Training and export conventions
- Pin and record package/model versions and random seeds; save split manifest, config, best validation checkpoint, optimizer/scheduler/scaler state, and training history.
- Keep training-only losses/metrics out of the exported forward graph. Use mixed precision only when numerically stable; clip gradients and detect non-finite loss.
- Prefer current Core ML Tools conversion flow, then inspect conversion diagnostics and run parity/performance tests. Weight compression/quantization is an experiment requiring accuracy and target-device latency evaluation.
- Avoid unsupported assumptions about exact tensor channels/strides; validate shapes, finite values, target ranges, and normalization at data/model boundaries.

## Research references
- MobileNetV3 (ICCV 2019): https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html
- Apple FastViT Core ML models and device measurements: https://developer.apple.com/machine-learning/models/
- FastViT (ICCV 2023): https://arxiv.org/abs/2303.14189 and https://github.com/apple-aiml-research/ml-fastvit
- Sapiens human vision models: https://arxiv.org/abs/2408.12569 and https://github.com/facebookresearch/sapiens
- Multi-task uncertainty weighting: https://arxiv.org/abs/1705.07115
- Core ML performance analysis: https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode
- Core ML Tools conversion: https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html

## Collaboration style
- Speak professional, friendly Bahasa Indonesia; call the user Fandy when natural.
- Separate evidence, project hypotheses, and measurements. Explain assumptions and give reproducible steps.

- Drive API inventory v3 is used during preparation to find complete RGB/depth/normal/alpha groups across subfolders, reporting incomplete and ambiguous groups. Packing validates decoded tensors and stores depth/normals as float32, RGB as uint8, mask as uint8, without baked-in augmentation. Persist each completed archive and its SHA256; training verifies the checksum when staging it.
- The active DGX raw-source run is separate from archived shard-based Colab experiments. Log raw inventory duration, per-batch data wait, throughput and GPU utilization. Do not promise a speedup before a DGX pilot measurement.
- The local-copy DGX option selects about 30,000 complete source samples, saves the exact manifest and rclone file list in the project, copies all four modalities with visible progress, verifies non-empty files, and decodes a copied sample. Use `configs/dgx_fastvit_t8_depth_normals_local30000_seed42.yaml` for local audit/training.
- For Colab speed comparison, use `colab/notebooks/colab_synthhuman_6chunk_pilot.ipynb`: it copies six extracted chunk folders from Drive to `/content` with a per-chunk byte progress bar, creates a sample manifest from local raw RGB/alpha/depth/normal, audits preprocessing, and runs a one-epoch pilot. The staged raw files are temporary across runtime resets; persist the manifest/checkpoints to Drive.

- Shard preparation checks the EXR magic header and prefers OpenEXR >=3.3 with explicit depth/normal channel names (Z or R for depth; RGB or XYZ for normals). Keep channel order consistent across decoders. A metadata-complete group can still have corrupt file content; report decode failures without fabricating labels.

- In an existing Colab runtime, EXR decoder activation must run after the original preprocessing definition. Preparation checks that the active read_array delegates to read_exr_array and stops before source reads if the old OpenCV-only function was restored.
