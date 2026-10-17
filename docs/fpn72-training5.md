# FPN72 training 5

This run starts from the completed FPN72 full-resolution checkpoint. It uses the same 50,000 selected SynthHuman samples, seed, split, input resolution, and task heads. The optimizer and cosine schedule start fresh.

## Changes

- The stride-4 heads are bilinearly upsampled and activated into coarse depth and unit normals. Their masked losses receive weight 0.5; final-output losses receive weight 1.0. The existing learned depth/normal task weights apply to the combined losses.
- Final normals start from unit coarse normals. The residual normal vector has an L2 norm below 0.15 before final normalization. The depth correction is capped at 0.05 in logit space. Since softplus is 1-Lipschitz, this also caps the final-versus-coarse depth change at 0.05 m for the same model and input.
- FPN features and coarse predictions are detached when passed into the residual branch. The final loss still updates the FPN through the additive coarse path. The residual branch receives luminance rather than RGB chroma; this does not remove sensitivity to lighting.
- Encoder LR is `2e-5`; FPN, heads, refinement, and loss weights use `1e-4`. Cosine annealing uses a shared `1e-7` minimum LR.
- The separate held-out comparison computes depth Edge-MAE, boundary normal mean angular error, and depth SILog for both checkpoints. This avoids adding Sobel and quantile work to every validation pass during 100 training epochs. Sobel edges are the top 10% of GT gradient magnitude inside a valid 3×3 neighborhood, expanded to a 3×3 edge band. SILog is `100 × std(log(pred) - log(GT))` per image. All three new metrics average image scores equally.

The depth correction limit applies to coarse versus final predictions within training 5. It does not bound the difference between training 5 and an older checkpoint. Luminance input and a bounded residual do not guarantee a specific normal-map color or real-photo accuracy.

## Run on DGX

Place the previous FPN72 run at the path in the new config, then launch from the repository root:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  bash scripts/train_dgx.sh \
  configs/dgx_fastvit_t8_depth_normals_local50000_seed42_fpn72_guarded_aux_100ep.yaml
```

The run writes to `runs/fastvit_t8_fpn72_guarded_aux_50k_seed42_384x512_100ep`. Its log shows final and coarse depth/normal losses. The saved checkpoint retains the new experiment flags, so inference must reconstruct the model with those flags.

## Compare on exactly the same held-out test split

After the run completes:

```bash
PYTHONPATH=src python -m relight_train.evaluate_checkpoints \
  --manifest runs/fastvit_t8_fpn_depth_normals_local50000_seed42_fpn72_fullres_lr1e5_384x512_100ep_v1/raw_dataset_manifest.jsonl \
  --checkpoint old=runs/fastvit_t8_fpn_depth_normals_local50000_seed42_fpn72_fullres_lr1e5_384x512_100ep_v1/best_checkpoint.pt \
  --checkpoint new=runs/fastvit_t8_fpn72_guarded_aux_50k_seed42_384x512_100ep/best_checkpoint.pt \
  --include-coarse \
  --output runs/fpn72_training5_heldout_comparison.json
```

Run this inside the project's Conda environment. The comparison reports final and stride-4 coarse predictions for both checkpoints, and refuses checkpoints whose selected sample IDs differ from the manifest. `--include-coarse` requires a second data pass per checkpoint.
