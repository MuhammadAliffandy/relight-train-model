# V9.5 depth fine-tuning from MoGe-3 pseudo-labels

This run starts from the V9.4 FastViT T8 checkpoint. Its normal prediction branch remains in the forward graph because the full-resolution depth refinement consumes the coarse normal, but `normal_head` and `normal_delta` are frozen. The objective trains depth only. It does not use the old Face-Depth-3D inferno-index loss: MoGe's `depth_m` is positive depth with smaller values nearer, and `valid_mask` excludes invalid pixels.

The loader reads each RGB photo and its `.npz`, resizes masked depth without mixing invalid zeros into valid pixels, and keeps the MoGe train/validation/test partitions. Loss combines median-aligned log-depth shape, a weak metric-scale term, and log-depth gradients; the coarse depth receives auxiliary supervision. This keeps the V9.4 initialization while learning the MoGe pseudo-label geometry. The validation split from the labeling worker is sample-level because no identity map was supplied, so its score is a training-selection proxy, not an identity-disjoint benchmark.

On DGX after labeling has printed `READY: 30000/30000 labels`:

```bash
cd ~/raid/Clara/relight-train-model
git pull --ff-only origin main
conda activate relight-dgx
bash scripts/launch_v9_5_moge3_base_depth_dgx.sh
tail -f logs/v9_5_moge3_base_depth_512.log
```

The launcher uses `setsid nohup`; Ctrl+C on `tail` does not stop training. It writes its PID to `logs/v9_5_moge3_base_depth_512.pid`. The worker requires all 30,000 labels and verifies `teacher_metadata.json` reports `Ruicheng/moge-3-vitl` with `refine_steps=0`. It uses 512×512 inputs, batch 4, 100 epochs, and no early stopping. It saves `last_checkpoint.pt` after every epoch, `best_checkpoint.pt` on validation improvement, and a numbered snapshot every 10 epochs. Re-running the launcher after a stopped process resumes from the last completed epoch in the same run directory. The log ends with `TRAIN_EXIT_STATUS=0` on success.

Check progress without holding a terminal open:

```bash
ps -p "$(cat logs/v9_5_moge3_base_depth_512.pid)" -o pid,stat,etime,cmd
tail -n 15 logs/v9_5_moge3_base_depth_512.log
```

Outputs are under `runs/fastvit_t8_fpn72_v9_5_moge3_base_depth_512`. The test partition remains held out; this script does not evaluate it during training. Compare the selected checkpoint on real portraits and a separate depth benchmark before replacing V9.4 in the app.
