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


class PortraitGeometryModel(nn.Module):
    def __init__(self, backbone="fastvit_t8.apple_in1k", pretrained=True, fpn_width=48, fpn_fusion="add"):
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

    def forward(self, x):
        output_size = x.shape[-2:]
        fused = self.neck(self.encoder(x))
        depth = F.softplus(self.depth_head(fused, output_size))
        normals = F.normalize(self.normal_head(fused, output_size), p=2, dim=1, eps=1e-6)
        return depth, normals
