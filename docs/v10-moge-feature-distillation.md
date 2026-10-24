# V10 FastViT depth distillation

V10 starts from `fastvit_t8.apple_in1k` ImageNet weights. The FPN and depth decoder are new; no V9.4 or V9.5 weights are loaded. The student predicts depth only. MoGe-3 ViT-L supplies the existing `depth_m` pseudo-labels and an offline feature target from its highest-resolution neck output. A deterministic 32-channel random projection keeps the feature cache small. The projection and student feature adapter are training tools; neither runs on iOS.

The teacher cache uses the **same MoGe checkpoint, manifest, resolution level and `refine_steps=0`** as the completed pseudo-label job. It rejects mismatched metadata. RGB images are fit and padded to width 384, height 512 for the student. The API sends raw RGB in `[0,1]`; the student normalizes RGB for its ImageNet-pretrained encoder internally. This is input normalization, not the old FaceDepth target scale or a surface-normal prediction branch.

## 1. Cache teacher features on DGX

From the repository root after `git pull origin main`:

```bash
conda activate moge3-teacher
MOGE_PYTHON="$(command -v python)" bash scripts/launch_v10_moge_feature_cache_dgx.sh
tail -f logs/v10_moge_feature_cache.log
```

The cache is resumable in `data/celebamask_hq/moge3_base_features_v10`. Its feature tensors are `.npz` files. The launcher uses `setsid nohup`; Ctrl+C on `tail` stops only the viewer. A server-side SIGTERM can still stop the job. Check the process with:

```bash
ps -p "$(cat logs/v10_moge_feature_cache.pid)" -o pid,stat,etime,cmd
```

The log must end with `READY: 30000/30000 cached features` and `FEATURE_EXIT_STATUS=0`. Do not start KD training from a partial cache. For a three-image extraction check, use `bash scripts/launch_v10_moge_feature_cache_dgx.sh --limit 3`, then run the launcher again without `--limit` to finish the cache.

## 2. Train the student

```bash
conda activate relight-dgx
V10_TRAIN_PYTHON="$(command -v python)" bash scripts/launch_v10_moge_distill_dgx.sh
tail -f logs/v10_moge_feature_kd_384x512.log
```

The training process logs every 100 batches, saves `last_checkpoint.pt` after each epoch, retains the best validation checkpoint, and writes `history.jsonl`. It ends with `TRAIN_EXIT_STATUS=0` on success. Feature loss has weight 0.02 after a five-epoch warmup. `best_deploy.pt` contains only the student model and inference metadata; use that file for Relight Lab and later iOS conversion. Rerun the same launcher after an interruption to resume from the last completed epoch. The held-out test split is not used for checkpoint selection. The validation split is only sample-disjoint unless an identity manifest was supplied during labeling.

For a controlled output-only comparison with exactly the same student architecture, set a separate run, log, and PID file and override the feature weight:

```bash
V10_RUN=runs/fastvit_t8_fpn72_v10_depth_only_384x512 \
V10_LOG=logs/v10_depth_only_384x512.log \
V10_PID=logs/v10_depth_only_384x512.pid \
bash scripts/launch_v10_moge_distill_dgx.sh --feature-weight 0
```

Compare the two selected checkpoints on the same real portraits, including face detail, hair edges, and temporal stability. The KD training loss includes a feature term, so compare **validation depth loss** and images, not total training loss. The feature adapter is saved for resuming training but is not part of the deployed student. V10 checkpoints carry `model_type: portrait_depth_student`, `input_size: [512,384]`, and `preprocessing: fit_pad_gray_0.5`; Relight Lab selects the depth-only loader and derives normals from depth for relighting.
