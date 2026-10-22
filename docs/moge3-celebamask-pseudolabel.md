# MoGe-3 pseudo-label depth on DGX

This stage labels the extracted CelebAMask-HQ photos with MoGe-3 ViT-L. The result is **model-estimated depth**, not measured ground truth. Each `.npz` contains `depth_m` (`float32`, smaller positive values are nearer), `valid_mask` (`uint8`, 1 = usable), and normalized `intrinsics` (`float32`, 3×3). Invalid depth is stored as zero and must be excluded with the mask when training. MoGe's output at invalid pixels can be `inf`; never normalize across those pixels.

Run on the DGX from the repository root. The teacher uses its own Conda environment so installing MoGe does not alter `relight-dgx`. The CUDA 12.4 PyTorch pair matches `environment.yml` and the DGX driver recorded for this project.

```bash
cd ~/raid/Clara/relight-train-model
git pull --ff-only origin main
conda create -n moge3-teacher python=3.11 -y
conda activate moge3-teacher
python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --extra-index-url https://download.pytorch.org/whl/cu124
if [ ! -d ~/raid/Clara/MoGe/.git ]; then
  git clone https://github.com/microsoft/MoGe.git ~/raid/Clara/MoGe
fi
python -m pip install -e ~/raid/Clara/MoGe
python -c 'import torch; from moge.model.v3 import MoGeModel; print(torch.__version__, torch.cuda.get_device_name(0))'
```

The local checkpoint should be `models/moge-3-vitl/model.pt`. If it was downloaded elsewhere, set `MOGE_CHECKPOINT` to its path. The image root is `data/celebamask_hq/images`; the script searches nested folders too. It expects exactly 30,000 photos by default and refuses to label an incomplete extraction. A pilot run labels ten photos and can be resumed into the full run:

```bash
cd ~/raid/Clara/relight-train-model
conda activate moge3-teacher
MOGE_PYTHON="$(command -v python)" bash scripts/launch_moge3_pseudolabel_dgx.sh --limit 10
tail -f logs/moge3_celebamask_hq_pseudolabel.log
```

After the pilot prints `READY: 10/10 labels`, start the full job in the same output folder. Existing atomic `.npz` labels are reused:

```bash
MOGE_PYTHON="$(command -v python)" bash scripts/launch_moge3_pseudolabel_dgx.sh
tail -f logs/moge3_celebamask_hq_pseudolabel.log
```

Check the process with `ps -p "$(cat logs/moge3_celebamask_hq_pseudolabel.pid)" -o pid,stat,etime,cmd`. Stop it with `kill "$(cat logs/moge3_celebamask_hq_pseudolabel.pid)"`; rerun the same launcher to resume. `nohup` and `setsid` detach it from the SSH session, but a server-side SIGTERM still stops it. Completed labels remain usable. Unexpected model/CUDA errors stop the job; bad image reads and invalid outputs are listed in `data/celebamask_hq/moge3_pseudo/errors.jsonl`. The process exits nonzero if any requested labels are missing.

`source_manifest.jsonl` records the preassigned train/validation/test split. By default it is a reproducible **sample-level** 90/5/5 split, so identities may overlap. For an identity-disjoint split, first provide `--groups path/to/groups.jsonl` with one `{"sample_id":"relative/path.jpg","group_id":"identity"}` line per image. A completed `manifest.jsonl` maps photos, splits, and label files. Do not treat this pseudo-label validation split as independent metric ground truth; evaluate depth against separately measured images.

MoGe-3's official `infer` API returns metric depth, a valid mask, and normalized intrinsics; its default refinement is three steps. This job uses resolution level 9 and full precision by default. `--fp16` is an optional speed experiment. Keep the chosen settings fixed within an output folder; the job rejects mismatched teacher settings or changed source files on resume.

## FlexGEMM and Triton 3.2

The CUDA 12.4 PyTorch 2.6 environment includes Triton 3.2. FlexGEMM's pinned sparse convolution kernel reads `tl.dtype.itemsize`, which that Triton version does not define. The worker supplies the equivalent byte-size property (`primitive_bitwidth // 8`) in its own process before loading MoGe. If a previous pilot stopped with a `triton.compiler.errors.CompilationError` ending in `dtype ... no attribute itemsize`, pull the current script and rerun the same pilot; the existing output folder can be reused. A different compiler error needs its complete final traceback before changing CUDA or PyTorch packages.
