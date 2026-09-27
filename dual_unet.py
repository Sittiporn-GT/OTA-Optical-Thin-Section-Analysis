"""
dual_unet_encoder.py — Dual-modal U-Net style encoder.
"""

import torch
import torch.nn as nn
from torchvision.models import (resnet50, resnet101, resnet152,
                                 ResNet50_Weights, ResNet101_Weights,
                                 ResNet152_Weights)

# ── Simple CNN building blocks ────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=False),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False))
    def forward(self, x): return self.net(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)


# ── Dual U-Net Encoder ────────────────────────────────────────────────
class DualUNetEncoder(nn.Module):
    """U-Net style dual-modal encoder.
    """

    CHANNELS = {
        'simple':    [64, 128, 256, 512],
        'resnet50':  [256, 512, 1024, 2048],
        'resnet101': [256, 512, 1024, 2048],
        'resnet152': [256, 512, 1024, 2048],
    }

    def __init__(self, backbone_type: str = 'resnet50',
                 in_chans: int = 6,
                 pretrained: bool = True):
        super().__init__()
        assert backbone_type in self.CHANNELS, \
            f"backbone_type must be one of {list(self.CHANNELS.keys())}"
        self.backbone_type = backbone_type
        self.channels      = self.CHANNELS[backbone_type]

        if backbone_type == 'simple':
            self._build_simple(in_chans)
        else:
            self._build_resnet(backbone_type, pretrained)

    # ── Simple CNN mode ───────────────────────────────────────────────
    def _build_simple(self, in_chans):
        self.stage1 = DoubleConv(in_chans, 64)
        self.stage2 = DownBlock(64,  128)
        self.stage3 = DownBlock(128, 256)
        self.stage4 = DownBlock(256, 512)
        self._kaiming_init(self.modules())

    # ── ResNet mode ───────────────────────────────────────────────────
    def _build_resnet(self, backbone_type: str, pretrained: bool):
        # input_proj: 6ch → 3ch before ResNet
        # Init: equal contribution from both modalities (weight = 1/6)
        self.input_proj = nn.Sequential(
            nn.Conv2d(6, 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=False))
        nn.init.constant_(self.input_proj[0].weight, 1.0 / 6.0)
        nn.init.ones_(self.input_proj[1].weight)
        nn.init.zeros_(self.input_proj[1].bias)

        # Load ResNet
        backbone = self._load_resnet(backbone_type, pretrained)

        # Split into stages
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 256ch, H/4
        self.layer2 = backbone.layer2   # 512ch, H/8
        self.layer3 = backbone.layer3   # 1024ch, H/16
        self.layer4 = backbone.layer4   # 2048ch, H/32

    @staticmethod
    def _load_resnet(backbone_type: str, pretrained: bool):
        if backbone_type == 'resnet50':
            w = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet50(weights=w)
        elif backbone_type == 'resnet101':
            w = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet101(weights=w)
        elif backbone_type == 'resnet152':
            w = ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet152(weights=w)

    @staticmethod
    def _kaiming_init(modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def init_weights(self, pretrained=None):
        """Called by builder.py — no-op since pretrained loaded at init."""
        pass

    # ── Forward ───────────────────────────────────────────────────────
    def forward(self, rgb: torch.Tensor, modal_x: torch.Tensor) -> list:
        x = torch.cat([rgb, modal_x], dim=1)   # (B, 6, H, W)

        if self.backbone_type == 'simple':
            f1 = self.stage1(x)
            f2 = self.stage2(f1)
            f3 = self.stage3(f2)
            f4 = self.stage4(f3)
        else:
            x  = self.input_proj(x)     # (B, 3, H, W)
            x  = self.layer0(x)         # (B, 64,  H/4,  W/4 )
            f1 = self.layer1(x)         # (B, 256, H/4,  W/4 )
            f2 = self.layer2(f1)        # (B, 512, H/8,  W/8 )
            f3 = self.layer3(f2)        # (B, 1024,H/16, W/16)
            f4 = self.layer4(f3)        # (B, 2048,H/32, W/32)

        return [f1, f2, f3, f4]