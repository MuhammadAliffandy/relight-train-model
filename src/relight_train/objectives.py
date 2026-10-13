from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryObjective(nn.Module):
    def __init__(self, depth_mode="metric"):
        super().__init__()
        self.depth_mode = depth_mode
        self.log_vars = nn.Parameter(torch.zeros(2))

    @staticmethod
    def masked_mean(value, mask):
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, predicted_depth, target_depth, predicted_normal, target_normal, mask):
        if self.depth_mode == "relative":
            pred = predicted_depth.clamp_min(1e-5).log()
            target = target_depth.clamp_min(1e-5).log()
            for batch in range(pred.shape[0]):
                valid = mask[batch, 0].bool()
                if not valid.any():
                    raise ValueError("A sample has an empty valid mask")
                pred[batch] -= torch.median(pred[batch, 0][valid])
                target[batch] -= torch.median(target[batch, 0][valid])
            depth_loss = self.masked_mean(F.smooth_l1_loss(pred, target, reduction="none"), mask)
        else:
            depth_loss = self.masked_mean(F.smooth_l1_loss(predicted_depth, target_depth, reduction="none"), mask)
        cosine = (F.normalize(predicted_normal, dim=1, eps=1e-6) * F.normalize(target_normal, dim=1, eps=1e-6)).sum(1, keepdim=True).clamp(-1, 1)
        normal_loss = self.masked_mean(1 - cosine, mask)
        total = sum(torch.exp(-self.log_vars[i]) * loss + self.log_vars[i] for i, loss in enumerate((depth_loss, normal_loss)))
        return total, {"depth_loss": depth_loss.detach(), "normal_loss": normal_loss.detach()}


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    sums = {}
    batch_count = 0
    for image, depth, normal, mask in loader:
        image, depth, normal, mask = [item.to(device, non_blocking=True) for item in (image, depth, normal, mask)]
        predicted_depth, predicted_normal = model(image)
        valid = mask[:, 0].bool()
        depth_error = (predicted_depth[:, 0] - depth[:, 0])[valid]
        pred_rel = predicted_depth[:, 0].clamp_min(1e-5).log()
        target_rel = depth[:, 0].clamp_min(1e-5).log()
        for index in range(image.shape[0]):
            sample_mask = valid[index]
            if not sample_mask.any():
                raise ValueError("Evaluation sample has empty mask")
            pred_rel[index] -= torch.median(pred_rel[index][sample_mask])
            target_rel[index] -= torch.median(target_rel[index][sample_mask])
        relative_error = (pred_rel - target_rel)[valid]
        dot = (F.normalize(predicted_normal, dim=1) * F.normalize(normal, dim=1)).sum(1).clamp(-1, 1)
        angle = torch.rad2deg(torch.acos(dot[valid]))
        batch_values = {
            "depth_mae_m": depth_error.abs().mean(),
            "depth_rmse_m": depth_error.square().mean().sqrt(),
            "depth_log_rmse_median_aligned": relative_error.square().mean().sqrt(),
            "normal_mean_deg": angle.mean(),
            "normal_median_deg": angle.median(),
            "normal_within_11_25_deg": (angle < 11.25).float().mean(),
        }
        for key, value in batch_values.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        batch_count += 1
    if batch_count == 0:
        raise ValueError("Evaluation partition is empty")
    return {key: value / batch_count for key, value in sums.items()}
