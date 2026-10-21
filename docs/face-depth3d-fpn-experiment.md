# Face-Depth-3D depth-only FPN comparison

The extracted `inferno_index` PNG is a relative proximity target. Brighter values are treated as nearer. The experiment compares the existing sequential concat FPN with a new four-stage direct fusion. Both runs start from the same V9.1 metric-depth checkpoint, use the same 512×512 images, the same group-based train/validation split, and optimize depth only. The V9 normal head remains in the checkpoint for model compatibility but receives no direct normal loss.

`concat_all` preserves the old sequential FPN and adds C2, C3, C4, and C5 projected to C2 resolution through a zero-initialized residual 1×1 convolution. At initialization its output matches `concat` exactly. Input images are resized from the extracted 1024×1024 samples; the source files remain intact.

Wait until the preparation log says `READY: 6872 pairs` and `data/face_depth_3d/manifest.jsonl` exists. On DGX, from the repository root:

```bash
git pull --ff-only origin main
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
INIT="runs/fastvit_t8_fpn72_v9_1_da2_distill_openimages1k_crop_synth5k_seed42_384x512_pilot/best_checkpoint.pt"
test -s data/face_depth_3d/manifest.jsonl
test -s "$INIT"
mkdir -p logs

nohup python -u scripts/train_face_depth3d_dgx.py \
  --init-checkpoint "$INIT" \
  --fusion concat_all --size 512 --batch-size 4 --epochs 20 \
  --run-dir runs/face_depth3d_v9_concat_all_512 \
  > logs/face_depth3d_concat_all_512.log 2>&1 < /dev/null &
echo $! > logs/face_depth3d_concat_all_512.pid
tail -f logs/face_depth3d_concat_all_512.log
```

Run the baseline separately with the same options, changing `--fusion concat_all` to `--fusion concat`, the run directory to `runs/face_depth3d_v9_concat_512`, and the log/PID names accordingly. Do not run both concurrently on the same GPU. Compare validation relative loss **and** the same real portraits visually; this dataset's labels are smooth and may reward oversmoothing. Measure inference time and memory on the target iPhone before raising the input to 640×640.

The checkpoint's depth output remains in the model's positive metric-depth parameterization, but after this relative-only fine-tune its values are **not calibrated meters**. Any downstream app must treat them as relative until calibrated on metric data. The existing normal output is retained from V9 without new normal supervision and may change indirectly through the shared encoder and FPN; assess it before using it for relighting.
