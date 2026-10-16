from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class FastViTEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.net = timm.create_model("fastvit_t8.apple_in1k", pretrained=pretrained, features_only=True)
        self.channels = tuple(self.net.feature_info.channels())

    def forward(self, x):
        return self.net(x)


class MobileNetV3Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.features = mobilenet_v3_large(weights=weights).features
        self.tap_indices = (3, 6, 12, 15)
        self.channels = (24, 40, 112, 160)

    def forward(self, x):
        outputs = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.tap_indices:
                outputs.append(x)
        if len(outputs) != len(self.channels):
            raise RuntimeError("Unexpected MobileNetV3 feature taps")
        return outputs


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = F.relu(self.depthwise_bn(self.depthwise(x)), inplace=True)
        return F.relu(self.pointwise_bn(self.pointwise(x)), inplace=True)


class CompactFPN(nn.Module):
    def __init__(self, channels, width=48, fusion="add"):
        super().__init__()
        if fusion not in ("add", "concat"):
            raise ValueError(f"Unsupported FPN fusion {fusion!r}; expected 'add' or 'concat'.")
        self.fusion = fusion
        self.lateral = nn.ModuleList([nn.Conv2d(channel, width, 1) for channel in channels])
        self.fusions = nn.ModuleList()
        if fusion == "concat":
            self.fusions = nn.ModuleList([nn.Conv2d(width * 2, width, 1) for _ in channels[:-1]])
            # Start from the exact behavior of element-wise addition, then learn the fusion.
            for layer in self.fusions:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                with torch.no_grad():
                    for channel in range(width):
                        layer.weight[channel, channel, 0, 0] = 1.0
                        layer.weight[channel, width + channel, 0, 0] = 1.0
        self.smooth = nn.ModuleList([DepthwiseSeparableBlock(width) for _ in channels])

    def forward(self, features):
        output = self.smooth[-1](self.lateral[-1](features[-1]))
        for index in range(len(features) - 2, -1, -1):
            output = F.interpolate(output, size=features[index].shape[-2:], mode="bilinear", align_corners=False)
            lateral = self.lateral[index](features[index])
            if self.fusion == "add":
                merged = output + lateral
            else:
                merged = self.fusions[index](torch.cat((output, lateral), dim=1))
            output = self.smooth[index](merged)
        return output


class MobileTaskHead(nn.Module):
    def __init__(self, width, output_channels):
        super().__init__()
        self.refine = DepthwiseSeparableBlock(width)
        self.predict = nn.Conv2d(width, output_channels, 1)

    def forward(self, x, output_size):
        return F.interpolate(self.predict(self.refine(x)), size=output_size, mode="bilinear", align_corners=False)


class Stride2GeometryRefinement(nn.Module):
    """Refine stride-4 geometry with stride-2 image and FPN cues."""
    def __init__(self, fpn_width, width=16):
        super().__init__()
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.feature_projection = nn.Conv2d(fpn_width, width, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(width * 2 + 4, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            DepthwiseSeparableBlock(width),
            DepthwiseSeparableBlock(width),
        )
        self.depth_delta = nn.Conv2d(width, 1, 3, padding=1)
        self.normal_delta = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.depth_delta.weight)
        nn.init.zeros_(self.depth_delta.bias)
        nn.init.zeros_(self.normal_delta.weight)
        nn.init.zeros_(self.normal_delta.bias)

    def forward(self, image, fused, depth_coarse, normal_coarse, size):
        image_detail = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        rgb_features = self.rgb_stem(image_detail)

        semantic = self.feature_projection(fused)
        semantic = F.interpolate(semantic, size=size, mode="bilinear", align_corners=False)
        coarse = F.interpolate(
            torch.cat((depth_coarse, normal_coarse), dim=1),
            size=size,
            mode="bilinear",
            align_corners=False,
        )

        features = self.fuse(torch.cat((rgb_features, semantic, coarse), dim=1))
        features = self.refine(features)
        return self.depth_delta(features), self.normal_delta(features)


class PortraitGeometryModel(nn.Module):
    def __init__(
        self,
        backbone="fastvit_t8.apple_in1k",
        pretrained=True,
        fpn_width=48,
        fpn_fusion="add",
        decoder_refinement="none",
        refinement_width=16,
    ):
        super().__init__()
        if backbone == "fastvit_t8.apple_in1k":
            self.encoder = FastViTEncoder(pretrained)
        elif backbone == "mobilenet_v3_large":
            self.encoder = MobileNetV3Encoder(pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone_name = backbone
        self.neck = CompactFPN(self.encoder.channels, width=fpn_width, fusion=fpn_fusion)
        self.depth_head = MobileTaskHead(fpn_width, 1)
        self.normal_head = MobileTaskHead(fpn_width, 3)
        if decoder_refinement not in ("none", "stride2"):
            raise ValueError("decoder_refinement must be 'none' or 'stride2'.")
        self.decoder_refinement = decoder_refinement
        if decoder_refinement == "stride2":
            self.refinement = Stride2GeometryRefinement(fpn_width, width=refinement_width)

    def forward(self, x):
        output_size = x.shape[-2:]
        fused = self.neck(self.encoder(x))
        depth_raw = self.depth_head.predict(self.depth_head.refine(fused))
        normal_raw = self.normal_head.predict(self.normal_head.refine(fused))

        depth_logits_full = F.interpolate(
            depth_raw, size=output_size, mode="bilinear", align_corners=False
        )
        normal_raw_full = F.interpolate(
            normal_raw, size=output_size, mode="bilinear", align_corners=False
        )

        if self.decoder_refinement == "stride2":
            refinement_size = (output_size[0] // 2, output_size[1] // 2)
            depth_coarse = F.interpolate(
                depth_raw, size=refinement_size, mode="bilinear", align_corners=False
            )
            normal_coarse = F.interpolate(
                normal_raw, size=refinement_size, mode="bilinear", align_corners=False
            )
            depth_delta, normal_delta = self.refinement(
                x, fused, depth_coarse, normal_coarse, refinement_size
            )
            depth_logits_full = depth_logits_full + F.interpolate(
                depth_delta, size=output_size, mode="bilinear", align_corners=False
            )
            normal_raw_full = normal_raw_full + F.interpolate(
                normal_delta, size=output_size, mode="bilinear", align_corners=False
            )

        depth = F.softplus(depth_logits_full)
        normals = F.normalize(normal_raw_full, p=2, dim=1, eps=1e-6)
        return depth, normals
