# Paste this whole file into one Colab cell. It loads V7 and tests one real photo.
# These are neutral, image-space lighting approximations, not physical relighting.
REUSE_TRACE_PHOTO = True
PERSON_THRESHOLD = 0.5
MASK_FEATHER_RADIUS = 5

import importlib.util
import io
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from google.colab import drive, files
from IPython.display import display
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

if importlib.util.find_spec("timm") is None:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)

drive.mount("/content/drive")
run_name = "fastvit_t8_fpn72_v7_from_v5_geometry_50k_seed42_384x512_30ep"
checkpoint_path = (
    Path("/content/drive/MyDrive/AppleCapstoneRelighting/runs")
    / run_name / "best_checkpoint.pt"
)
if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint V7 belum ada di Drive: {checkpoint_path}")

model_source = Path("/content/relight-train-model/src/relight_train/model.py")
if not model_source.is_file() or "grayscale_refinement" not in model_source.read_text():
    model_source = Path("/content/fpn72_v7_three_point_model.py")
    if not model_source.is_file():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/MuhammadAliffandy/relight-train-model/"
            "947b00b41414d6672ebebb080f799df01d8a1051/src/relight_train/model.py",
            model_source,
        )
spec = importlib.util.spec_from_file_location("fpn72_v7_three_point_model", model_source)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
exp = checkpoint["experiment"]
if not (
    exp.get("backbone") == "fastvit_t8.apple_in1k"
    and exp.get("fpn_fusion") == "concat"
    and exp.get("decoder_refinement") == "fullres"
    and exp.get("guarded_refinement")
):
    raise ValueError("Checkpoint ini tidak sesuai dengan arsitektur V7 guarded FPN72.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_v7_light = model_module.PortraitGeometryModel(
    backbone=exp["backbone"],
    pretrained=False,
    fpn_width=int(exp["fpn_width"]),
    fpn_fusion=exp["fpn_fusion"],
    decoder_refinement=exp["decoder_refinement"],
    refinement_width=int(exp.get("refinement_width", 16)),
    fullres_width=int(exp.get("fullres_width", 24)),
    rgb_full_width=int(exp.get("rgb_full_width", 8)),
    guarded_refinement=True,
    residual_depth_limit=float(exp.get("residual_depth_limit", 0.05)),
    residual_normal_limit=float(exp.get("residual_normal_limit", 0.15)),
    grayscale_refinement=bool(exp.get("grayscale_refinement", False)),
)
model_v7_light.load_state_dict(checkpoint["model"], strict=True)
model_v7_light.to(device).eval()
best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", "?"))
del checkpoint

if REUSE_TRACE_PHOTO and isinstance(globals().get("photo"), Image.Image):
    lighting_photo = globals()["photo"].convert("RGB")
    lighting_photo_name = globals().get("photo_name", "foto_dari_cell_trace")
else:
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("Upload satu foto untuk uji pencahayaan.")
    lighting_photo_name, photo_bytes = next(iter(uploaded.items()))
    lighting_photo = Image.open(io.BytesIO(photo_bytes)).convert("RGB")

height, width = map(int, exp["input_size"])
base_photo = lighting_photo.resize((width, height), Image.Resampling.BILINEAR)
base_rgb = np.asarray(base_photo, dtype=np.float32) / 255.0

# The fixed person mask controls where lighting is applied and where drift is measured.
# It is never passed to V7 as a fourth input channel.
seg_weights = DeepLabV3_ResNet50_Weights.DEFAULT
segmenter = deeplabv3_resnet50(weights=seg_weights).to(device).eval()
person_idx = seg_weights.meta["categories"].index("person")
seg_x = seg_weights.transforms()(base_photo).unsqueeze(0).to(device)
with torch.inference_mode():
    seg_logits = segmenter(seg_x)["out"]
    seg_logits = F.interpolate(
        seg_logits, size=(height, width), mode="bilinear", align_corners=False
    )
    person_prob = seg_logits.softmax(dim=1)[0, person_idx].cpu().numpy()
del segmenter, seg_x, seg_logits
if device.type == "cuda":
    torch.cuda.empty_cache()

person_mask = person_prob >= PERSON_THRESHOLD
if person_mask.sum() < 100:
    raise RuntimeError(
        "Segmenter tidak menemukan cukup piksel orang; coba foto lain atau turunkan "
        "PERSON_THRESHOLD menjadi 0.35."
    )
soft_mask = np.asarray(
    Image.fromarray((person_mask * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(MASK_FEATHER_RADIUS)
    ),
    dtype=np.float32,
) / 255.0

ys, xs = np.where(person_mask)
ymin, ymax = int(ys.min()), int(ys.max())
xmin, xmax = int(xs.min()), int(xs.max())
cx = 0.5 * (xmin + xmax)
half_width = max(0.5 * (xmax - xmin), 1.0)
xx = np.clip((np.arange(width, dtype=np.float32)[None, :] - cx) / half_width, -1, 1)
xx = np.broadcast_to(xx, (height, width))
left = (1 - xx) * 0.5
right = (1 + xx) * 0.5
yy = (np.arange(height, dtype=np.float32)[:, None] - ymin) / max(ymax - ymin, 1)
front = np.exp(-0.5 * ((xx / 0.7) ** 2 + ((yy - 0.35) / 0.32) ** 2))

eroded = np.asarray(
    Image.fromarray((person_mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(15)),
    dtype=np.float32,
) / 255.0
rim = np.clip(person_mask.astype(np.float32) - eroded, 0, 1)
rim = np.asarray(
    Image.fromarray((rim * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3)),
    dtype=np.float32,
) / 255.0

fields = [
    ("Asli", np.ones((height, width), np.float32), False),
    ("Terang merata +0.4 EV", np.full((height, width), 2 ** 0.4, np.float32), False),
    ("Key kiri", 1 + 0.55 * (left - 0.5), True),
    ("Key kanan", 1 + 0.55 * (right - 0.5), True),
    ("Fill depan", 1 + 0.30 * front, True),
    ("3-point: key kiri", 1 + 0.50 * (left - 0.5) + 0.12 * (right - 0.5) + 0.22 * rim * right, True),
    ("3-point: key kanan", 1 + 0.50 * (right - 0.5) + 0.12 * (left - 0.5) + 0.22 * rim * left, True),
]

def srgb_to_linear(value):
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(value):
    return np.where(value <= 0.0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - 0.055)

linear_base = srgb_to_linear(base_rgb)
imagenet_mean = np.array([0.485, 0.456, 0.406], np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225], np.float32)
variants = []
for label, field, equal_mean in fields:
    field = np.asarray(field, dtype=np.float32)
    if equal_mean:
        field = field / max(float(field[person_mask].mean()), 1e-6)
    field = np.clip(field, 0.55, 1.55)
    gain = 1 + soft_mask * (field - 1)
    lit_linear = linear_base * gain[..., None]
    added_clipping_pct = float(np.mean(np.any(lit_linear[person_mask] > 1, axis=-1)) * 100)
    lit_rgb = np.clip(linear_to_srgb(np.clip(lit_linear, 0, 1)), 0, 1).astype(np.float32)
    variants.append((label, lit_rgb, added_clipping_pct))

def to_input(image):
    normalized = ((image - imagenet_mean) / imagenet_std).copy()
    return torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)

def to_numpy_depth(tensor):
    return tensor[0, 0].detach().float().cpu().numpy()

def to_numpy_normal(tensor):
    return tensor[0].detach().float().cpu().numpy()

predictions = []
with torch.inference_mode():
    for label, lit_rgb, clipping_pct in variants:
        final_d, final_n, coarse_d, coarse_n = model_v7_light(
            to_input(lit_rgb), return_coarse=True
        )
        predictions.append({
            "label": label,
            "rgb": lit_rgb,
            "clipping_pct": clipping_pct,
            "depth": to_numpy_depth(final_d),
            "normal": to_numpy_normal(final_n),
            "coarse_depth": to_numpy_depth(coarse_d),
            "coarse_normal": to_numpy_normal(coarse_n),
        })

upper_mask = person_mask & (
    np.arange(height)[:, None] <= ymin + 0.43 * (ymax - ymin)
)
if upper_mask.sum() < 100:
    upper_mask = person_mask.copy()

def angular_difference_deg(a, b):
    dot = np.clip(np.sum(a * b, axis=0), -1, 1)
    return np.degrees(np.arccos(dot))

baseline = predictions[0]
rows = []
for pred in predictions:
    if pred is baseline:
        normal_drift = np.zeros((height, width), dtype=np.float32)
        coarse_drift = np.zeros((height, width), dtype=np.float32)
    else:
        normal_drift = angular_difference_deg(pred["normal"], baseline["normal"])
        coarse_drift = angular_difference_deg(pred["coarse_normal"], baseline["coarse_normal"])
    depth_change = pred["depth"] - baseline["depth"]
    pred["normal_drift"] = normal_drift
    pred["depth_change"] = depth_change
    rows.append({
        "cahaya": pred["label"],
        "depth_median_m": float(np.median(pred["depth"][person_mask])),
        "depth_abs_drift_cm": float(np.mean(np.abs(depth_change[person_mask])) * 100),
        "depth_median_drift_cm": float(np.median(depth_change[person_mask]) * 100),
        "normal_kasar_drift_deg": float(np.mean(coarse_drift[person_mask])),
        "normal_akhir_drift_deg": float(np.mean(normal_drift[person_mask])),
        "normal_atas_drift_deg": float(np.mean(normal_drift[upper_mask])),
        "normal_atas_p90_deg": float(np.percentile(normal_drift[upper_mask], 90)),
        "normal_atas_gt30_pct": float(np.mean(normal_drift[upper_mask] > 30) * 100),
        "klip_baru_pct": pred["clipping_pct"],
    })

lighting_table = pd.DataFrame(rows).round(3)
print(f"Foto: {lighting_photo_name} | V7 best epoch: {best_epoch}")
print(f"Mask orang: {person_mask.mean() * 100:.1f}% gambar; upper-person: {upper_mask.sum()} piksel")
print("Semua kondisi memakai foto, framing, checkpoint, dan mask evaluasi yang sama.")
display(lighting_table)

def masked_normal_rgb(normal):
    color = np.moveaxis(np.clip(normal, -1, 1) * 0.5 + 0.5, 0, -1)
    color[~person_mask] = 0
    return color

depth_shapes = [pred["depth"] - np.median(pred["depth"][person_mask]) for pred in predictions]
valid_depth = np.concatenate([value[person_mask] for value in depth_shapes])
depth_lo, depth_hi = np.percentile(valid_depth, [2, 98])
n = len(predictions)
fig, axes = plt.subplots(4, n, figsize=(3.4 * n, 13))
for col, (pred, depth_shape) in enumerate(zip(predictions, depth_shapes)):
    axes[0, col].imshow(pred["rgb"])
    axes[0, col].set_title(pred["label"], fontsize=10)
    depth_artist = axes[1, col].imshow(
        np.ma.masked_where(~person_mask, depth_shape),
        cmap="viridis", vmin=depth_lo, vmax=depth_hi,
    )
    axes[1, col].set_title(f"Depth akhir; median {np.median(pred['depth'][person_mask]):.2f} m", fontsize=9)
    axes[2, col].imshow(masked_normal_rgb(pred["coarse_normal"]))
    axes[2, col].set_title("Normal kasar", fontsize=9)
    axes[3, col].imshow(masked_normal_rgb(pred["normal"]))
    axes[3, col].set_title("Normal akhir", fontsize=9)
for axis in axes.flat:
    axis.axis("off")
fig.colorbar(depth_artist, ax=axes[1].tolist(), fraction=0.012, pad=0.01, label="meter relatif median sendiri")
fig.suptitle("V7: pengaruh arah cahaya pada foto yang sama", fontsize=15)
fig.subplots_adjust(top=0.92, hspace=0.14, wspace=0.05)
plt.show()

fig, axes = plt.subplots(2, n, figsize=(3.4 * n, 7))
for col, pred in enumerate(predictions):
    normal_artist = axes[0, col].imshow(
        np.ma.masked_where(~person_mask, pred["normal_drift"]),
        cmap="magma", vmin=0, vmax=60,
    )
    axes[0, col].set_title(pred["label"], fontsize=9)
    depth_artist = axes[1, col].imshow(
        np.ma.masked_where(~person_mask, pred["depth_change"] * 100),
        cmap="coolwarm", vmin=-30, vmax=30,
    )
for axis in axes.flat:
    axis.axis("off")
fig.colorbar(normal_artist, ax=axes[0].tolist(), fraction=0.012, pad=0.01, label="perubahan normal dari asli (°)")
fig.colorbar(depth_artist, ax=axes[1].tolist(), fraction=0.012, pad=0.01, label="perubahan depth dari asli (cm)")
fig.suptitle("Drift prediksi terhadap foto asli", fontsize=15)
fig.subplots_adjust(top=0.91, hspace=0.10, wspace=0.05)
plt.show()

print("Perubahan normal/depth di tabel adalah drift dari prediksi pada foto asli, bukan error terhadap GT.")
print("Pencahayaan 3-point di sini adalah simulasi gain 2D netral: key kiri/kanan, fill lawan arah, rim dekat siluet.")
