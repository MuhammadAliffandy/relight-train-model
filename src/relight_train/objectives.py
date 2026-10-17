from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryObjective(nn.Module):
    def __init__(self, depth_mode="metric", auxiliary_weight=0.0):
        super().__init__()
        self.depth_mode = depth_mode
        self.auxiliary_weight = float(auxiliary_weight)
        self.log_vars = nn.Parameter(torch.zeros(2))

    @staticmethod
    def masked_mean(value, mask):
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    def geometry_losses(self, predicted_depth, target_depth, predicted_normal, target_normal, mask):
        if self.depth_mode == "relative":
            pred = predicted_depth.clamp_min(1e-5).log().clone()
            target = target_depth.clamp_min(1e-5).log().clone()
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
        return depth_loss, normal_loss

    def forward(self, predicted_depth, target_depth, predicted_normal, target_normal, mask,
                coarse_depth=None, coarse_normal=None):
        depth_loss, normal_loss = self.geometry_losses(
            predicted_depth, target_depth, predicted_normal, target_normal, mask
        )
        parts = {"depth_loss": depth_loss.detach(), "normal_loss": normal_loss.detach()}
        if self.auxiliary_weight:
            if coarse_depth is None or coarse_normal is None:
                raise ValueError("Auxiliary loss requires both coarse predictions")
            coarse_depth_loss, coarse_normal_loss = self.geometry_losses(
                coarse_depth, target_depth, coarse_normal, target_normal, mask
            )
            depth_loss = depth_loss + self.auxiliary_weight * coarse_depth_loss
            normal_loss = normal_loss + self.auxiliary_weight * coarse_normal_loss
            parts.update({"coarse_depth_loss": coarse_depth_loss.detach(),
                          "coarse_normal_loss": coarse_normal_loss.detach()})
        total = sum(torch.exp(-self.log_vars[i]) * loss + self.log_vars[i] for i, loss in enumerate((depth_loss, normal_loss)))
        return total, parts


def sobel_magnitude(values):
    channels = values.shape[1]
    kernels = values.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))) / 8
    horizontal = F.conv2d(values, kernels[None, None].repeat(channels, 1, 1, 1), padding=1, groups=channels)
    vertical = F.conv2d(values, kernels.T[None, None].repeat(channels, 1, 1, 1), padding=1, groups=channels)
    return (horizontal.square() + vertical.square()).sum(dim=1).sqrt()


def gt_edge_region(target, mask, quantile=0.9):
    interior = F.avg_pool2d(mask.float(), kernel_size=3, stride=1, padding=1)[:, 0] == 1
    magnitude = sobel_magnitude(target * mask)
    regions = []
    for sample in range(target.shape[0]):
        valid = interior[sample]
        if not valid.any():
            regions.append(valid)
            continue
        threshold = torch.quantile(magnitude[sample][valid].float(), quantile)
        edge = valid & (magnitude[sample] >= threshold) & (magnitude[sample] > 0)
        neighborhood = F.max_pool2d(edge[None, None].float(), 3, stride=1, padding=1)[0, 0].bool()
        regions.append(neighborhood & valid)
    return torch.stack(regions)


@torch.inference_mode()
def evaluate(model, loader, device, detail_metrics=False):
    model.eval()
    sums = {}
    detail_sums = {"depth_edge_mae_m": 0.0, "normal_boundary_mean_deg": 0.0, "depth_silog_x100": 0.0} if detail_metrics else {}
    detail_counts = {key: 0 for key in detail_sums}
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
        if detail_metrics:
            angle_map = torch.rad2deg(torch.acos(dot))
            depth_edges = gt_edge_region(depth, mask)
            normal_edges = gt_edge_region(F.normalize(normal, dim=1, eps=1e-6), mask)
            for index in range(image.shape[0]):
                if depth_edges[index].any():
                    detail_sums["depth_edge_mae_m"] += float(
                        (predicted_depth[index, 0] - depth[index, 0]).abs()[depth_edges[index]].mean()
                    )
                    detail_counts["depth_edge_mae_m"] += 1
                if normal_edges[index].any():
                    detail_sums["normal_boundary_mean_deg"] += float(angle_map[index][normal_edges[index]].mean())
                    detail_counts["normal_boundary_mean_deg"] += 1
                log_difference = (
                    predicted_depth[index, 0].clamp_min(1e-5).log()
                    - depth[index, 0].clamp_min(1e-5).log()
                )[valid[index]]
                detail_sums["depth_silog_x100"] += float(
                    log_difference.var(unbiased=False).clamp_min(0).sqrt() * 100
                )
                detail_counts["depth_silog_x100"] += 1
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
    metrics = {key: value / batch_count for key, value in sums.items()}
    metrics.update({key: value / detail_counts[key] if detail_counts[key] else float("nan")
                    for key, value in detail_sums.items()})
    return metrics
