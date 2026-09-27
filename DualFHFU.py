"""
DualFHFU.py — FHFU dual-branch model adapted for our training pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError(
        "segmentation_models_pytorch is required.\n"
        "Install via: pip install segmentation-models-pytorch")

class GatedFusion(nn.Module):
    """Adaptive weighted fusion between 2 branches."""
    def __init__(self, in_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x, y):
        G = self.gate(torch.cat([x, y], dim=1))
        return x * G + y * (1 - G)


class DSC(nn.Module):
    """Depthwise Separable Convolution — texture feature enhancer."""
    def __init__(self, nin, nout, kernel_size=3, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nin, nin, kernel_size, padding=padding, groups=nin),
            nn.Conv2d(nin, nout, 1))

    def forward(self, x): return self.net(x)


class EnhancedDualBranch(nn.Module):
    """
    Dual-branch encoder:
    """
    def __init__(self, in_channels: int, encoder: str, weights: str):
        super().__init__()
        self.segnet_ppl       = smp.DeepLabV3Plus(
            encoder_name=encoder, encoder_weights=weights,
            in_channels=in_channels, classes=64, activation=None)
        self.texture_enhancer = DSC(64, 32)

        self.segnet_xpl       = smp.DeepLabV3Plus(
            encoder_name=encoder, encoder_weights=weights,
            in_channels=in_channels, classes=64, activation=None)
        self.color_enhancer   = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(inplace=False),
            nn.Conv2d(128, 32, 5, padding=2))

        self.gatefusion = GatedFusion(32)

    def forward(self, x_ppl, x_xpl):
        feat_ppl = self.texture_enhancer(self.segnet_ppl(x_ppl))
        feat_xpl = self.color_enhancer(self.segnet_xpl(x_xpl))
        return self.gatefusion(feat_ppl, feat_xpl)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
    def forward(self, x): return self.conv(x)


class _FHFU(nn.Module):
    """Core FHFU model — parametric, no config import."""
    def __init__(self, num_classes: int, encoder: str, weights: str,
                 in_ch_ppl: int = 3):
        super().__init__()
        self.in_ch_ppl         = in_ch_ppl
        self.dual_branch       = EnhancedDualBranch(in_ch_ppl, encoder, weights)
        self.segmentation_head = SegmentationHead(32, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 6, H, W) — cat(PPL, XPL)"""
        x_ppl, x_xpl = torch.split(
            images, [self.in_ch_ppl, images.shape[1] - self.in_ch_ppl], dim=1)
        fused  = self.dual_branch(x_ppl, x_xpl)
        return self.segmentation_head(fused)

class DualFHFU(nn.Module):
    """FHFU wrapped for our training pipeline.
    """

    def __init__(self, cfg, criterion,
                 norm_layer=nn.BatchNorm2d,   # kept for interface compatibility
                 encoder: str = 'resnet50',
                 weights: str = 'imagenet'):
        super().__init__()
        self.criterion = criterion
        self.model     = _FHFU(
            num_classes=cfg.num_classes,
            encoder=encoder,
            weights=weights)

    def encode_decode(self, rgb: torch.Tensor,
                      modal_x: torch.Tensor) -> torch.Tensor:
        images = torch.cat([rgb, modal_x], dim=1)   # (B, 6, H, W)
        out    = self.model(images)                   # (B, C, H', W')
        # smp decoder may produce slightly different size → interpolate
        out    = F.interpolate(out, size=rgb.shape[2:],
                               mode='bilinear', align_corners=False)
        return out

    def forward(self, rgb: torch.Tensor,
                modal_x: torch.Tensor,
                label: torch.Tensor = None):
        out = self.encode_decode(rgb, modal_x)
        if label is not None:
            return self.criterion(out, label.long())
        return out