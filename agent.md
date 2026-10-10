# Project Context: Apple Capstone Portrait Relighting

## Goal and current status
- Build an iOS tool for independent creators to relight a portrait/body crop with controllable key, fill, and rim lights.
- The model predicts human-region geometry and reflectance; Swift/Vision handles person segmentation and compositing.
- Current repository is a research prototype: `training_pipeline_relighting.ipynb` defines a supervised multi-task training/evaluation/export workflow. It has not yet been trained on real labels or validated on an iPhone.

## Current model hypothesis
- Model contract: one ImageNet-normalized RGB crop in; aligned depth, camera-frame surface normals, and diffuse albedo maps out.
- Baseline remains ImageNet-pretrained FastViT-T8 (`timm`) with a compact 48-channel depthwise-separable FPN and three dense task heads. MobileNetV3-Large is an optional backbone ablation, not a selected replacement.
- FastViT has published iPhone/Core ML classification measurements; these do not predict latency of this custom dense multi-head graph. MobileNetV3 has mobile CPU and dense-prediction evidence on other platforms. Compare both on identical training data and target iPhones before selecting.
- Bilinear resize + convolution only; avoid transposed convolutions for the intended Core ML path.
- Outputs: positive relative depth unless calibration metadata establishes metric scale; unit-length normals; albedo in [0, 1]. The notebook may use metric depth only when the dataset documents calibrated units; SynthHuman absolute depth is converted from cm to m. Do not call Sapiens monocular depth metric depth unless scale is independently calibrated.
- Keep FastViT-T8 as the research baseline. Any alternative (including MobileNetV3) must earn a switch through matched quality, Core ML compatibility, and target-iPhone latency measurements; neither paper claims guarantee 60 FPS for our graph.

## Notebook workflow
- Report dataset chunk totals and incomplete paired samples; report train/validation duration, epoch wall time, throughput, and ETA, and persist timing in the run history on Drive. Do not dump or visualize every sample by default.
- Default the first training pilot to a deterministic 500-sample subset and 5 epochs, saved as a separate pilot manifest/run directory so checkpoints from earlier experiments cannot be mixed in. Require enough complete paired rows and keep the validation/test split non-empty.
- In Colab, mount Google Drive at the start of each fresh runtime and store dataset, manifest, caches, checkpoints, histories, metrics, and exported models under `MyDrive/AppleCapstoneRelighting`; reconnecting the runtime does not preserve local `/content` files.
- Keep full-release SynthHuman acquisition in a separate optional notebook cell, disabled by default, isolated from the pilot pipeline. That cell processes official chunks sequentially into Drive and resumes using per-chunk completion markers. The release download is about 330 GB plus extracted/temp space; do not start it unless the user explicitly enables the cell in Colab after checking Drive capacity.
- Colab Pro does not guarantee a specific GPU or uninterrupted runtime; checkpoints on Drive support recovery, but the actual runtime/session must be started and authorized by the user. Core ML prediction parity requires macOS even though conversion can run on Linux.
- Training writes a mid-epoch resumable checkpoint to Drive every 10 minutes, with deterministic sample order, batch position, optimizer/scheduler/scaler and RNG state; a runtime interruption can lose at most the work since the last checkpoint.
- Log CUDA availability/version, GPU name/count/capability and memory before training; show a per-batch live progress indicator and current losses during each epoch.
- The Colab notebook has distinct cells for Drive setup, official SynthHuman subset download, automatic manifest generation, EXR decoding, sample audit, model, training, and evaluation. Use one chunk as the download ceiling for the initial pilot, but train first on a deterministic 500-sample subset for 5 epochs; expand only after validating labels and pilot behavior. SynthHuman has no albedo, so this trial supervises depth/normals only and cannot be exported as a trained three-output model.

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
