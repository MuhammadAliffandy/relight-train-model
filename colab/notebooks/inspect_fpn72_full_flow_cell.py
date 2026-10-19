# One Colab cell: inspect the complete guarded FastViT-T8/FPN72 forward path.
# Set RUN_NAME to V8 after its checkpoint has been copied to Drive.
RUN_NAME = globals().get(
    "RUN_NAME", "fastvit_t8_fpn72_v7_from_v5_geometry_50k_seed42_384x512_30ep"
)
CHECKPOINT_NAME = globals().get("CHECKPOINT_NAME", "best_checkpoint.pt")
ENCODER_LEAF_DETAIL = globals().get("ENCODER_LEAF_DETAIL", True)
CHANNEL_PREVIEWS = globals().get("CHANNEL_PREVIEWS", 3)

import importlib.util
import math
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from google.colab import drive, files
from IPython.display import display

drive.mount("/content/drive")
run_dir = Path("/content/drive/MyDrive/AppleCapstoneRelighting/runs") / RUN_NAME
checkpoint_path = run_dir / CHECKPOINT_NAME
if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint belum ada di Drive: {checkpoint_path}")

model_source = Path("/content/relight-train-model/src/relight_train/model.py")
if not model_source.is_file() or "grayscale_refinement" not in model_source.read_text():
    model_source = Path("/content/fpn72_model_for_trace.py")
    if not model_source.is_file():
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/MuhammadAliffandy/relight-train-model/"
            "947b00b41414d6672ebebb080f799df01d8a1051/src/relight_train/model.py", model_source,
        )
spec = importlib.util.spec_from_file_location("fpn72_trace_model", model_source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
exp = state["experiment"]
if not (exp.get("backbone") == "fastvit_t8.apple_in1k"
        and exp.get("fpn_fusion") == "concat"
        and exp.get("decoder_refinement") == "fullres"
        and exp.get("guarded_refinement")):
    raise ValueError("Cell ini khusus FastViT-T8 + concat FPN + fullres guarded refinement V7/V8.")
model = module.PortraitGeometryModel(
    backbone=exp["backbone"], pretrained=False,
    fpn_width=int(exp["fpn_width"]), fpn_fusion=exp["fpn_fusion"],
    decoder_refinement=exp["decoder_refinement"],
    refinement_width=int(exp.get("refinement_width", 16)),
    fullres_width=int(exp.get("fullres_width", 24)),
    rgb_full_width=int(exp.get("rgb_full_width", 8)),
    guarded_refinement=True,
    residual_depth_limit=float(exp.get("residual_depth_limit", 0.05)),
    residual_normal_limit=float(exp.get("residual_normal_limit", 0.15)),
    grayscale_refinement=bool(exp.get("grayscale_refinement", False)),
)
model.load_state_dict(state["model"], strict=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
best_epoch = state.get("best_epoch", state.get("epoch", "?"))
del state

uploaded = files.upload()
if not uploaded:
    raise ValueError("Upload satu foto asli untuk inspeksi.")
photo_name, photo_bytes = next(iter(uploaded.items()))
import io
photo = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
height, width = map(int, exp["input_size"])
# RawSynthHumanDataset memakai direct resize RGB bilinear dan normalisasi ImageNet.
resized = photo.resize((width, height), Image.Resampling.BILINEAR)
rgb = np.asarray(resized, dtype=np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)
x = torch.from_numpy(((rgb - mean) / std).copy()).permute(2, 0, 1)[None].to(device)

records = []
maps = {}
channel_maps = {}
encoder_leaves = []

def first_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for item in obj.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return None

def tensor_shape(value):
    tensor = first_tensor(value)
    return "×".join(str(int(dim)) for dim in tensor.shape) if tensor is not None else "?"

def tensor_stats(value):
    tensor = first_tensor(value)
    if tensor is None:
        return {"shape": str(type(value).__name__)}
    value = tensor.detach().float()
    finite = value[torch.isfinite(value)]
    result = {"shape": "×".join(str(int(dim)) for dim in value.shape)}
    if finite.numel():
        result.update(
            min=float(finite.min()), max=float(finite.max()),
            mean=float(finite.mean()), std=float(finite.std(unbiased=False)),
            mean_abs=float(finite.abs().mean()),
            nonfinite_pct=100 * (1 - finite.numel() / value.numel()),
        )
    return result

def record(name, value, keep_map=True):
    records.append({"step": len(records) + 1, "stage": name, **tensor_stats(value)})
    tensor = first_tensor(value)
    if keep_map and tensor is not None and tensor.ndim == 4:
        t = tensor[0].detach().float()
        maps[name] = t.abs().mean(dim=0).cpu().numpy()
        if t.shape[0] > 1 and CHANNEL_PREVIEWS and (
            name.startswith("encoder/C") or name.endswith("/output")
            or "concat" in name or "predict_conv1x1" in name
        ):
            channel_maps[name] = t[:CHANNEL_PREVIEWS].cpu().numpy()
    return value

def trace_seq(prefix, seq, value):
    record(prefix + "/input", value)
    for index, layer in enumerate(seq):
        value = layer(value)
        record(f"{prefix}/{index}_{type(layer).__name__}", value)
    return value

def trace_sep(prefix, block, value):
    record(prefix + "/input", value)
    value = record(prefix + "/depthwise_conv3x3", block.depthwise(value))
    value = record(prefix + "/depthwise_BN", block.depthwise_bn(value))
    value = record(prefix + "/depthwise_ReLU", F.relu(value, inplace=False))
    value = record(prefix + "/pointwise_conv1x1", block.pointwise(value))
    value = record(prefix + "/pointwise_BN", block.pointwise_bn(value))
    return record(prefix + "/pointwise_ReLU_output", F.relu(value, inplace=False))

pending = {}
handles = []
if ENCODER_LEAF_DETAIL:
    for path, layer in model.encoder.net.named_modules():
        if not path or any(layer.children()):
            continue
        def before(current, inputs, layer_path=path):
            pending[id(current)] = tensor_shape(inputs)
        def after(current, inputs, output, layer_path=path):
            output_tensor = first_tensor(output)
            encoder_leaves.append({
                "layer": layer_path, "type": type(current).__name__,
                "input_shape": pending.pop(id(current), "?"),
                "output_shape": tensor_shape(output),
                "output_mean_abs": float(output_tensor.detach().float().abs().mean())
                if output_tensor is not None else float("nan"),
            })
        handles.extend([
            layer.register_forward_pre_hook(before),
            layer.register_forward_hook(after),
        ])

with torch.inference_mode():
    record("input/RGB_ImageNet_normalized", x)
    features = model.encoder(x)
    for handle in handles:
        handle.remove()
    handles.clear()

    reductions = list(model.encoder.net.feature_info.reduction())
    for index, feat in enumerate(features):
        record(f"encoder/C{index+2}_stride{reductions[index]}", feat)

    # FPN: lateral 1×1, upsample, concat, fusion 1×1, smoothing 3×3 + 1×1.
    lateral = []
    for index, feat in enumerate(features):
        name = f"fpn/P{index+2}"
        record(name + "/lateral_input", feat)
        lateral.append(record(name + "/lateral_conv1x1", model.neck.lateral[index](feat)))
    deepest = len(features) - 1
    top = trace_sep(f"fpn/P{deepest+2}/smooth", model.neck.smooth[deepest], lateral[deepest])
    record(f"fpn/P{deepest+2}/output", top)
    for index in range(deepest - 1, -1, -1):
        name = f"fpn/P{index+2}"
        up = record(name + "/topdown_bilinear", F.interpolate(
            top, size=features[index].shape[-2:], mode="bilinear", align_corners=False
        ))
        pair = record(name + "/concat_topdown_lateral", torch.cat((up, lateral[index]), dim=1))
        merged = record(name + "/fusion_conv1x1", model.neck.fusions[index](pair))
        top = trace_sep(name + "/smooth", model.neck.smooth[index], merged)
        record(name + "/output", top)

    fused = trace_sep("pre_head_context", model.pre_head_context, top)
    depth_features = trace_sep("depth_head/refine", model.depth_head.refine, fused)
    record("depth_head/predict_conv1x1_input", depth_features)
    depth_raw = record("depth_head/predict_conv1x1_stride4", model.depth_head.predict(depth_features))
    normal_features = trace_sep("normal_head/refine", model.normal_head.refine, fused)
    record("normal_head/predict_conv1x1_input", normal_features)
    normal_raw = record("normal_head/predict_conv1x1_stride4", model.normal_head.predict(normal_features))
    depth_logits_full = record("depth_head/bilinear_to_input", F.interpolate(
        depth_raw, size=(height, width), mode="bilinear", align_corners=False
    ))
    normal_raw_full = record("normal_head/bilinear_to_input", F.interpolate(
        normal_raw, size=(height, width), mode="bilinear", align_corners=False
    ))
    coarse_depth = record("depth_head/coarse_softplus_m", F.softplus(depth_logits_full))
    coarse_normal = record("normal_head/coarse_L2_normalize", F.normalize(
        normal_raw_full, p=2, dim=1, eps=1e-6
    ))

    # Guarded full-resolution residual branch.
    ref = model.refinement
    record("residual/input_fused_detached", fused.detach())
    record("residual/input_depth_raw_detached", depth_raw.detach())
    record("residual/input_normal_raw_detached", normal_raw.detach())
    image = x
    if ref.grayscale_rgb:
        red = image[:, 0:1] * 0.229 + 0.485
        green = image[:, 1:2] * 0.224 + 0.456
        blue = image[:, 2:3] * 0.225 + 0.406
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        image = record("residual/grayscale_ImageNet_normalized", ((luminance - 0.459) / 0.226).expand(-1, 3, -1, -1))

    half_size = (height // 2, width // 2)
    raw_pair = record("residual/coarse_concat_depth_normal_stride4", torch.cat((depth_raw.detach(), normal_raw.detach()), dim=1))
    coarse_half = record("residual/coarse_bilinear_stride2", F.interpolate(
        raw_pair, size=half_size, mode="bilinear", align_corners=False
    ))
    image_half = record("residual/RGB_bilinear_stride2", F.interpolate(
        image, size=half_size, mode="bilinear", align_corners=False
    ))
    rgb_half = trace_seq("residual/rgb_half_stem", ref.rgb_half_stem, image_half)
    semantic_proj = record("residual/feature_projection_conv1x1", ref.feature_projection(fused.detach()))
    semantic_half = record("residual/semantic_bilinear_stride2", F.interpolate(
        semantic_proj, size=half_size, mode="bilinear", align_corners=False
    ))
    half_cat = record("residual/stride2_concat_semantic_rgb_coarse", torch.cat((semantic_half, rgb_half, coarse_half), dim=1))
    half = trace_seq("residual/coarse_fusion", ref.coarse_fusion, half_cat)
    for index, block in enumerate(ref.stride2_context):
        half = trace_sep(f"residual/stride2_context_{index}", block, half)
    full_features = record("residual/stride2_to_full_bilinear", F.interpolate(
        half, size=(height, width), mode="bilinear", align_corners=False
    ))
    coarse_full = record("residual/coarse_bilinear_full", F.interpolate(
        raw_pair, size=(height, width), mode="bilinear", align_corners=False
    ))
    rgb_full = trace_seq("residual/rgb_full_stem", ref.rgb_full_stem, image)
    full_cat = record("residual/fullres_concat_context_rgb_coarse", torch.cat((full_features, rgb_full, coarse_full), dim=1))
    full = trace_seq("residual/fullres_fusion", ref.fullres_fusion, full_cat)
    full = trace_sep("residual/fullres_context", ref.fullres_context, full)
    depth_delta_raw = record("residual/depth_delta_conv3x3_raw", ref.depth_delta(full))
    normal_delta_raw = record("residual/normal_delta_conv3x3_raw", ref.normal_delta(full))
    depth_delta = record("residual/depth_delta_bounded", model.residual_depth_limit * torch.tanh(depth_delta_raw))
    norm_mag = record("residual/normal_delta_vector_length", torch.linalg.vector_norm(normal_delta_raw, dim=1, keepdim=True).clamp_min(1e-6))
    normal_delta = record("residual/normal_delta_bounded", model.residual_normal_limit * torch.tanh(norm_mag) * normal_delta_raw / norm_mag)
    final_depth_logits = record("final/depth_logits_coarse_plus_residual", depth_logits_full + depth_delta)
    final_normal_pre = record("final/normal_coarse_plus_residual_before_L2", coarse_normal + normal_delta)
    final_depth = record("final/depth_softplus_m", F.softplus(final_depth_logits))
    final_normal = record("final/normal_L2_normalized", F.normalize(final_normal_pre, p=2, dim=1, eps=1e-6))

trace_table = pd.DataFrame(records)
pd.set_option("display.max_rows", None)
print(f"Foto: {photo_name} | checkpoint: {RUN_NAME}/{CHECKPOINT_NAME} | best epoch: {best_epoch}")
print("Encoder reductions:", reductions, "| feature channels:", model.encoder.channels)
print("Setiap baris adalah tensor sungguhan dari forward pass pada foto ini.")
display(trace_table.round(5))
if encoder_leaves:
    print(f"Detail {len(encoder_leaves)} modul daun di encoder FastViT (input/output tiap modul):")
    display(pd.DataFrame(encoder_leaves).round(5))

def normal_rgb(tensor):
    return np.moveaxis((tensor[0].detach().float().clamp(-1, 1) * 0.5 + 0.5).cpu().numpy(), 0, -1)

fig, axes = plt.subplots(1, 6, figsize=(25, 5))
axes[0].imshow(photo); axes[0].set_title("Foto asli")
axes[1].imshow(rgb); axes[1].set_title(f"Input model {width}×{height}")
depth_arrays = [coarse_depth[0, 0].cpu().numpy(), final_depth[0, 0].cpu().numpy()]
depth_lo, depth_hi = np.percentile(np.concatenate([a.ravel() for a in depth_arrays]), [2, 98])
axes[2].imshow(depth_arrays[0], cmap="viridis", vmin=depth_lo, vmax=depth_hi); axes[2].set_title("Depth kasar (m)")
axes[3].imshow(depth_arrays[1], cmap="viridis", vmin=depth_lo, vmax=depth_hi); axes[3].set_title("Depth akhir (m)")
axes[4].imshow(normal_rgb(coarse_normal)); axes[4].set_title("Normal kasar (XYZ→RGB)")
axes[5].imshow(normal_rgb(final_normal)); axes[5].set_title("Normal akhir (XYZ→RGB)")
for ax in axes: ax.axis("off")
fig.tight_layout(); plt.show()

for group in ("encoder/", "fpn/", "pre_head_context/", "depth_head/", "normal_head/", "residual/", "final/"):
    names = [name for name in maps if name.startswith(group)]
    if not names:
        continue
    cols = 4
    fig, axes = plt.subplots(math.ceil(len(names) / cols), cols,
                             figsize=(17, 3.0 * math.ceil(len(names) / cols)), squeeze=False)
    for ax, name in zip(axes.flat, names):
        heat = maps[name]
        lo, hi = np.percentile(heat[np.isfinite(heat)], [2, 98])
        ax.imshow(heat, cmap="magma", vmin=lo, vmax=max(hi, lo + 1e-9))
        ax.set_title(name.replace(group, ""), fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(names):]: ax.axis("off")
    fig.suptitle(group.rstrip("/") + " — rerata |aktivasi| per kanal", fontsize=14)
    fig.tight_layout(); plt.show()

for name, channels in channel_maps.items():
    fig, axes = plt.subplots(1, len(channels), figsize=(4 * len(channels), 3.5))
    axes = np.atleast_1d(axes)
    finite = channels[np.isfinite(channels)]
    limit = max(float(np.percentile(np.abs(finite), 98)), 1e-9)
    for channel_index, (ax, channel) in enumerate(zip(axes, channels)):
        ax.imshow(channel, cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_title(f"{name} | kanal {channel_index}", fontsize=8)
        ax.axis("off")
    fig.tight_layout(); plt.show()

print("Peta fitur multi-channel adalah rerata |aktivasi|, bukan depth, normal, atau attribution.")
print("Skala warna tiap peta otomatis; gunakan angka di tabel untuk membandingkan besar aktivasi antartahap.")
print("Panel kanal menampilkan tiga kanal pertama, bukan keseluruhan 72 kanal; warna biru/merah menandai tanda aktivasi.")
print("Normal ditampilkan tanpa mask segmentasi; piksel latar juga memiliki prediksi model.")
print("Tabel `trace_table` dan `encoder_leaves` tetap tersedia untuk inspeksi/filter lanjutan.")
