from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from relight_train.data import RawSynthHumanDataset, _resize_array


class DomainAugmentedDataset(Dataset):
    def __init__(self, rows, input_size, config):
        self.rows = rows
        self.input_size = tuple(input_size)
        self.config = dict(config)
        self.base = RawSynthHumanDataset(rows, input_size, augment=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        augmentation_seed = None
        if isinstance(index, tuple):
            index, augmentation_seed = index
        rng = random.Random(augmentation_seed)
        image, depth, normal, valid_mask = self.base[index]
        height, width = self.input_size
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = np.clip(image.permute(1, 2, 0).numpy() * std + mean, 0, 1)
        depth = depth.permute(1, 2, 0).numpy()
        normal = normal.permute(1, 2, 0).numpy()
        mask = valid_mask[0].numpy() > 0

        with Image.open(Path(self.rows[index]["mask"])) as alpha_image:
            alpha = np.asarray(
                alpha_image.convert("L").resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255

        if rng.random() < 0.5:
            rgb, depth, normal, mask, alpha = [
                np.flip(value, axis=1).copy() for value in (rgb, depth, normal, mask, alpha)
            ]
            normal[..., 0] *= -1

        if rng.random() < float(self.config.get("background_probability", 0)):
            background_rng = np.random.default_rng(rng.getrandbits(32))
            base_color = background_rng.uniform(0.2, 0.8, (1, 1, 3))
            color_grid = np.clip(base_color + background_rng.normal(0, 0.12, (4, 4, 3)), 0, 1)
            background = np.asarray(
                Image.fromarray(np.round(color_grid * 255).astype(np.uint8), "RGB")
                .resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255
            rgb = rgb * alpha[..., None] + background * (1 - alpha[..., None])

        cropped = False
        if rng.random() < float(self.config.get("crop_probability", 0)):
            ys, xs = np.nonzero(mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            bbox_height = y1 - y0
            if bbox_height >= int(self.config.get("minimum_person_height", 64)):
                head_crop = rng.random() < float(self.config.get("head_crop_fraction", 0.4))
                fraction_range = (0.28, 0.42) if head_crop else (0.58, 0.78)
                crop_height = min(height, max(24, round(rng.uniform(*fraction_range) * bbox_height)))
                crop_width = min(width, max(24, round(crop_height * width / height)))
                crop_height = min(height, max(18, round(crop_width * height / width)))
                top_band = ys < y0 + max(1, round(0.28 * bbox_height))
                center_x = float(np.median(xs[top_band])) if top_band.any() else float(np.median(xs))
                left = max(0, min(width - crop_width, round(center_x - crop_width / 2)))
                top = max(0, min(height - crop_height, round(y0 - 0.04 * bbox_height)))
                right, bottom = left + crop_width, top + crop_height
                rgb = _resize_array(rgb[top:bottom, left:right], self.input_size)
                depth = _resize_array(depth[top:bottom, left:right], self.input_size)
                normal = _resize_array(normal[top:bottom, left:right], self.input_size)
                mask = _resize_array(mask[top:bottom, left:right].astype(np.uint8), self.input_size, True)[..., 0] > 0
                cropped = True

        if rng.random() < float(self.config.get("jpeg_probability", 0)):
            low, high = (int(value) for value in self.config.get("jpeg_quality", (75, 95)))
            quality = rng.randint(low, high)
            buffer = io.BytesIO()
            Image.fromarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").save(
                buffer, format="JPEG", quality=quality
            )
            buffer.seek(0)
            rgb = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32) / 255

        rgb = (np.clip(rgb, 0, 1) - mean) / std
        image_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()
        depth_tensor = torch.from_numpy(depth.copy()).permute(2, 0, 1).float()
        normal_tensor = F.normalize(torch.from_numpy(normal.copy()).permute(2, 0, 1).float(), dim=0, eps=1e-6)
        mask_tensor = torch.from_numpy(mask[None].copy().astype(np.float32))
        mask_tensor *= (torch.linalg.vector_norm(normal_tensor, dim=0, keepdim=True) > 0).float()
        if cropped:
            interior = F.avg_pool2d(mask_tensor.unsqueeze(0), 3, stride=1, padding=1)[0] > 0.999
            mask_tensor *= interior.float()
        depth_mask = torch.zeros_like(mask_tensor) if cropped else mask_tensor.clone()
        return image_tensor, depth_tensor, normal_tensor, mask_tensor, depth_mask
