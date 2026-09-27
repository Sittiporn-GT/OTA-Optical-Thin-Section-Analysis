import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (resnet50, resnet101, resnet152,
                                 ResNet50_Weights, ResNet101_Weights,
                                 ResNet152_Weights)
 
 
# ── Simple CNN blocks ─────────────────────────────────────────────────
 
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False))
    def forward(self, x): return self.net(x)
 
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super().__init__()
        self.c1 = ConvBNReLU(in_ch, out_ch, stride=stride, dilation=dilation)
        self.c2 = ConvBNReLU(out_ch, out_ch, dilation=dilation)
        self.sc = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
    def forward(self, x):
        return F.relu(self.sc(x) + self.c2(self.c1(x)), inplace=False)
 
 
# ── Dual DeepLab Encoder ──────────────────────────────────────────────
 
class DualDeepLabEncoder(nn.Module):
    """DeepLabV3+ encoder for dual-modal (XPL + PPL).
 
    Returns [f1, f2, f3, f4] for use with existing deeplabv3+ decoder.
    f1 = low-level features (H/4), f4 = ASPP input (H/16 or H/8).
    """
 
    CHANNELS = {
        'simple':    [64, 128, 256, 512],
        'resnet50':  [256, 512, 1024, 2048],
        'resnet101': [256, 512, 1024, 2048],
        'resnet152': [256, 512, 1024, 2048],
    }
 
    def __init__(self, backbone_type: str = 'resnet50',
                 in_chans: int = 6,
                 pretrained: bool = True,
                 output_stride: int = 8):
        super().__init__()
        assert backbone_type in self.CHANNELS, \
            f"backbone_type must be one of {list(self.CHANNELS.keys())}"
        assert output_stride in (8, 16), "output_stride must be 8 or 16"
 
        self.backbone_type = backbone_type
        self.output_stride = output_stride
        self.channels      = self.CHANNELS[backbone_type]
 
        if backbone_type == 'simple':
            self._build_simple(in_chans, output_stride)
        else:
            self._build_resnet(backbone_type, pretrained, output_stride)
 
    # ── Simple CNN mode ───────────────────────────────────────────────
    def _build_simple(self, in_chans, output_stride):
        self.stage1 = nn.Sequential(ConvBNReLU(in_chans, 32), ConvBNReLU(32, 64))
        self.stage2 = nn.Sequential(ResBlock(64,  128, stride=2))
        if output_stride == 8:
            # No stride in stage3/4 — use dilation
            self.stage3 = nn.Sequential(ResBlock(128, 256, stride=1, dilation=1))
            self.stage4 = nn.Sequential(ResBlock(256, 512, stride=1, dilation=2),
                                         ResBlock(512, 512, stride=1, dilation=2))
        else:  # output_stride=16
            self.stage3 = nn.Sequential(ResBlock(128, 256, stride=2))
            self.stage4 = nn.Sequential(ResBlock(256, 512, stride=1, dilation=2),
                                         ResBlock(512, 512, stride=1, dilation=2))
        self._kaiming_init(self.modules())
 
    # ── ResNet mode ───────────────────────────────────────────────────
    def _build_resnet(self, backbone_type, pretrained, output_stride):
        # input_proj 6→3: preserves all pretrained ResNet weights
        # Init: equal weight from both modalities (1/6 each input channel)
        self.input_proj = nn.Sequential(
            nn.Conv2d(6, 3, 3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=False))
        nn.init.constant_(self.input_proj[0].weight, 1.0 / 6.0)
        nn.init.ones_(self.input_proj[1].weight)
        nn.init.zeros_(self.input_proj[1].bias)
 
        backbone = self._load_resnet(backbone_type, pretrained)
 
        # ✅ FIX: correct dilation strategy per output_stride
        if output_stride == 16:
            # Dilate ONLY layer4 (layer3 keeps stride=2 → f3 at H/16)
            # f4 at H/16 with dilation=2
            self._apply_dilation(backbone.layer4, dilation=2)
        elif output_stride == 8:
            # Dilate layer3 (stride→dilation=2) AND layer4 (dilation=4)
            # f3 and f4 both at H/8
            self._apply_dilation(backbone.layer3, dilation=2)
            self._apply_dilation(backbone.layer4, dilation=4)
 
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 256ch  H/4   ← low-level features
        self.layer2 = backbone.layer2   # 512ch  H/8
        self.layer3 = backbone.layer3   # 1024ch H/16 (OS=16) or H/8 (OS=8)
        self.layer4 = backbone.layer4   # 2048ch H/16 (OS=16) or H/8 (OS=8) ← ASPP
 
    @staticmethod
    def _apply_dilation(layer, dilation: int):
        """Remove stride=2 and apply dilation to a ResNet layer group.
 
        Handles both:
          - 3x3 Conv2d (stride=2): remove stride + add dilation
          - 1x1 Conv2d (stride=2): downsample/shortcut branch
            remove stride only — both must match for residual addition.
        """
        for m in layer.modules():
            if not isinstance(m, nn.Conv2d): continue
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    # Main conv: apply dilation to maintain receptive field
                    m.dilation = (dilation, dilation)
                    m.padding  = (dilation, dilation)
                # 1x1 downsample conv: remove stride only, padding stays 0
            elif m.kernel_size == (3, 3) and m.dilation == (1, 1):
                # Other 3x3 convs in subsequent blocks
                m.dilation = (dilation, dilation)
                m.padding  = (dilation, dilation)
 
    @staticmethod
    def _load_resnet(backbone_type, pretrained):
        if backbone_type == 'resnet50':
            w = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet50(weights=w)
        elif backbone_type == 'resnet101':
            w = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet101(weights=w)
        else:  # resnet152
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
        """Called by builder.py — no-op since pretrained is loaded in __init__."""
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
            x  = self.input_proj(x)     # (B, 3,    H,    W  )
            x  = self.layer0(x)         # (B, 64,   H/4,  W/4)
            f1 = self.layer1(x)         # (B, 256,  H/4,  W/4)  ← low-level
            f2 = self.layer2(f1)        # (B, 512,  H/8,  W/8)
            f3 = self.layer3(f2)        # (B, 1024, H/16, W/16) OS=16 | H/8 OS=8
            f4 = self.layer4(f3)        # (B, 2048, H/16, W/16) OS=16 | H/8 OS=8
 
        return [f1, f2, f3, f4]