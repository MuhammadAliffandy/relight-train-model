from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pseudo_depth_distillation_loss(predicted_depth, teacher_depth, valid_mask):
    """Match per-image relative depth shape without assuming teacher metric scale."""
    predicted_values = predicted_depth.float()
    teacher = teacher_depth.float()
    losses = []
    for index in range(predicted_depth.shape[0]):
        valid = valid_mask[index, 0].bool() & torch.isfinite(teacher[index, 0])
        if int(valid.sum()) < 64:
            continue
        pred_values = predicted_values[index, 0][valid]
        teacher_values = teacher[index, 0][valid]
        pred_low, pred_high = torch.quantile(pred_values, pred_values.new_tensor([0.05, 0.95]))
        teacher_low, teacher_high = torch.quantile(teacher_values, teacher_values.new_tensor([0.05, 0.95]))
        pred_span = (pred_high - pred_low).clamp_min(1e-4)
        teacher_span = (teacher_high - teacher_low).clamp_min(1e-4)
        pred_norm = ((predicted_values[index, 0] - pred_low) / pred_span).clamp(-0.25, 1.25)
        teacher_norm = ((teacher[index, 0] - teacher_low) / teacher_span).clamp(-0.25, 1.25)
        losses.append(F.smooth_l1_loss(pred_norm[valid], teacher_norm[valid], beta=0.1))
    if not losses:
        raise ValueError("A real-image batch has no usable teacher depth pixels")
    return torch.stack(losses).mean()


class GeometryObjective(nn.Module):
    def __init__(self, depth_mode="metric", auxiliary_weight=0.0, crop_depth_shape_weight=0.0,
                 depth_gradient_weight=0.0, normal_edge_weight=0.0):
        super().__init__()
        self.depth_mode = depth_mode
        self.auxiliary_weight = float(auxiliary_weight)
        self.crop_depth_shape_weight = float(crop_depth_shape_weight)
        self.depth_gradient_weight = float(depth_gradient_weight)
        self.normal_edge_weight = float(normal_edge_weight)
        self.log_vars = nn.Parameter(torch.zeros(2))

    @staticmethod
    def masked_mean(value, mask):
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def crop_depth_shape_loss(predicted_depth, target_depth, mask, depth_mask):
        cropped = depth_mask.flatten(1).sum(1) == 0
        interior = F.avg_pool2d(mask.float(), 5, stride=1, padding=2) == 1
        valid = interior & (target_depth > 0) & cropped[:, None, None, None]
        predicted_log = predicted_depth.float().clamp_min(1e-5).log()
        target_log = target_depth.float().clamp_min(1e-5).log()
        difference = predicted_log - target_log
        counts = valid.flatten(1).sum(1).clamp_min(1)
        offsets = (difference * valid).flatten(1).sum(1) / counts
        centered = difference - offsets[:, None, None, None]
        per_image = (F.smooth_l1_loss(centered, torch.zeros_like(centered), reduction="none", beta=0.1)
                     * valid).flatten(1).sum(1) / counts
        return (per_image * cropped).sum() / cropped.sum().clamp_min(1)

    @staticmethod
    def depth_gradient_loss(predicted_depth, target_depth, mask):
        predicted_log = predicted_depth.float().clamp_min(1e-5).log()
        target_log = target_depth.float().clamp_min(1e-5).log()
        interior = F.avg_pool2d(mask.float(), 5, stride=1, padding=2) == 1
        valid = interior & (target_depth > 0)
        horizontal = valid[:, :, :, 1:] & valid[:, :, :, :-1]
        vertical = valid[:, :, 1:, :] & valid[:, :, :-1, :]
        horizontal_error = (predicted_log[:, :, :, 1:] - predicted_log[:, :, :, :-1]
                            - target_log[:, :, :, 1:] + target_log[:, :, :, :-1])
        vertical_error = (predicted_log[:, :, 1:, :] - predicted_log[:, :, :-1, :]
                          - target_log[:, :, 1:, :] + target_log[:, :, :-1, :])
        horizontal_loss = F.smooth_l1_loss(horizontal_error, torch.zeros_like(horizontal_error),
                                           reduction="none", beta=0.05)
        vertical_loss = F.smooth_l1_loss(vertical_error, torch.zeros_like(vertical_error),
                                         reduction="none", beta=0.05)
        return ((horizontal_loss * horizontal).sum() + (vertical_loss * vertical).sum()) / (
            horizontal.sum() + vertical.sum()).clamp_min(1)

    @staticmethod
    def normal_edge_loss(predicted_normal, target_normal, mask):
        target_unit = F.normalize(target_normal.float(), dim=1, eps=1e-6)
        predicted_unit = F.normalize(predicted_normal.float(), dim=1, eps=1e-6)
        cosine = (target_unit * predicted_unit).sum(1, keepdim=True).clamp(-1, 1)
        interior = F.avg_pool2d(mask.float(), 3, stride=1, padding=1) == 1
        strength = (sobel_magnitude(target_unit * mask)[:, None] / 0.15).clamp(0, 1)
        weights = (strength * interior).detach()
        return ((1 - cosine) * weights).sum() / weights.sum().clamp_min(1)

    def geometry_losses(self, predicted_depth, target_depth, predicted_normal, target_normal, mask,
                        depth_mask=None):
        depth_mask = mask if depth_mask is None else depth_mask
        if self.depth_mode == "relative":
            pred = predicted_depth.clamp_min(1e-5).log().clone()
            target = target_depth.clamp_min(1e-5).log().clone()
            for batch in range(pred.shape[0]):
                valid = depth_mask[batch, 0].bool()
                if not valid.any():
                    continue
                pred[batch] -= torch.median(pred[batch, 0][valid])
                target[batch] -= torch.median(target[batch, 0][valid])
            depth_loss = self.masked_mean(F.smooth_l1_loss(pred, target, reduction="none"), depth_mask)
        else:
            depth_loss = self.masked_mean(F.smooth_l1_loss(predicted_depth, target_depth, reduction="none"), depth_mask)
        cosine = (F.normalize(predicted_normal, dim=1, eps=1e-6) * F.normalize(target_normal, dim=1, eps=1e-6)).sum(1, keepdim=True).clamp(-1, 1)
        normal_loss = self.masked_mean(1 - cosine, mask)
        return depth_loss, normal_loss

    def forward(self, predicted_depth, target_depth, predicted_normal, target_normal, mask,
                coarse_depth=None, coarse_normal=None, depth_mask=None):
        depth_mask = mask if depth_mask is None else depth_mask
        depth_loss, normal_loss = self.geometry_losses(
            predicted_depth, target_depth, predicted_normal, target_normal, mask, depth_mask
        )
        parts = {"depth_loss": depth_loss.detach(), "normal_loss": normal_loss.detach()}
        if self.auxiliary_weight:
            if coarse_depth is None or coarse_normal is None:
                raise ValueError("Auxiliary loss requires both coarse predictions")
            coarse_depth_loss, coarse_normal_loss = self.geometry_losses(
                coarse_depth, target_depth, coarse_normal, target_normal, mask, depth_mask
            )
            depth_loss = depth_loss + self.auxiliary_weight * coarse_depth_loss
            normal_loss = normal_loss + self.auxiliary_weight * coarse_normal_loss
            parts.update({"coarse_depth_loss": coarse_depth_loss.detach(),
                          "coarse_normal_loss": coarse_normal_loss.detach()})
        if self.crop_depth_shape_weight:
            crop_shape_loss = self.crop_depth_shape_loss(
                predicted_depth, target_depth, mask, depth_mask
            )
            depth_loss = depth_loss + self.crop_depth_shape_weight * crop_shape_loss
            parts["crop_depth_shape_loss"] = crop_shape_loss.detach()
            if self.auxiliary_weight:
                coarse_crop_shape_loss = self.crop_depth_shape_loss(
                    coarse_depth, target_depth, mask, depth_mask
                )
                depth_loss = depth_loss + self.crop_depth_shape_weight * self.auxiliary_weight * coarse_crop_shape_loss
                parts["coarse_crop_depth_shape_loss"] = coarse_crop_shape_loss.detach()
        if self.depth_gradient_weight:
            gradient_loss = self.depth_gradient_loss(predicted_depth, target_depth, mask)
            depth_loss = depth_loss + self.depth_gradient_weight * gradient_loss
            parts["depth_gradient_loss"] = gradient_loss.detach()
        if self.normal_edge_weight:
            edge_loss = self.normal_edge_loss(predicted_normal, target_normal, mask)
            normal_loss = normal_loss + self.normal_edge_weight * edge_loss
            parts["normal_edge_loss"] = edge_loss.detach()
        depth_term = torch.exp(-self.log_vars[0]) * depth_loss + self.log_vars[0]
        normal_term = torch.exp(-self.log_vars[1]) * normal_loss + self.log_vars[1]
        has_depth_supervision = depth_mask.sum() > 0
        if self.crop_depth_shape_weight or self.depth_gradient_weight:
            has_depth_supervision = has_depth_supervision | (mask.sum() > 0)
        total = torch.where(has_depth_supervision, depth_term, depth_term * 0) + normal_term
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
