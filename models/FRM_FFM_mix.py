"""
FRM_FFM.py — Feature Rectify Module + Feature Fusion Module
Mix Attention (FFMx) based on Zheng et al., Trans-SedNet, 2024

Key features (ตาม paper):
- Shared Km, Vm — K and V from both modalities concatenated then compressed via ChannelAttention
- Asymmetric Q — Q1 and Q2 attend to shared Km, Vm separately
- Pre-norm before attention (LayerNorm in MixAttention)
- Scaled dot-product attention (ไม่ใช่ linear attention)

Bug fixes vs original CMX:
- ReLU(inplace=False) ทุกจุด
- norm1, norm2 เพิ่ม Pre-norm ใน MixAttention.forward()
"""
import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_

class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim      = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp      = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=False),          
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x   = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max_ = self.max_pool(x).view(B, self.dim * 2)
        y   = torch.cat((avg, max_), dim=1)
        y   = self.mlp(y).view(B, self.dim * 2, 1)
        return y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=False),         
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        return self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super().__init__()
        self.lambda_c        = lambda_c
        self.lambda_s        = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights  = SpatialWeights(dim=dim, reduction=reduction)

    def forward(self, x1, x2):
        cw = self.channel_weights(x1, x2)
        sw = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * cw[1] * x2 + self.lambda_s * sw[1] * x2
        out_x2 = x2 + self.lambda_c * cw[0] * x1 + self.lambda_s * sw[0] * x1
        return out_x1, out_x2

class ChannelAttention(nn.Module):
    """SE-style channel attention บน concatenated K or V
    Input:  [B, heads, N, 2*d]
    Output: [B, heads, N, 2*d]  (gated)
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden    = max(channels // reduction, 8)
        self.fc1  = nn.Linear(channels, hidden, bias=True)
        self.act  = nn.ReLU(inplace=False)   
        self.fc2  = nn.Linear(hidden, channels, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, heads, N, C] — mean over N → SE gate
        s = x.mean(dim=2)               # [B, heads, C]
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        w = self.gate(s).unsqueeze(2)   # [B, heads, 1, C]
        return x * w


class MixAttention(nn.Module):
    """Mix Attention ตาม Zheng et al., 2024
    Flow:
      x1, x2 → QKV projections
      concat(K1,K2) → ChannelAttn → Km  (shared key)
      concat(V1,V2) → ChannelAttn → Vm  (shared value)
      Q1 @ Km → attn1 @ Vm → out1
      Q2 @ Km → attn2 @ Vm → out2
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., reduction=4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.d         = dim // num_heads
        self.scale     = qk_scale or self.d ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.qkv1  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv2  = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # ChannelAttention บน 2*d (concat K หรือ V จาก 2 modality)
        self.se_k  = ChannelAttention(channels=2 * self.d, reduction=reduction)
        self.se_v  = ChannelAttention(channels=2 * self.d, reduction=reduction)

        # project 2*d → d เพื่อให้ attention shape ถูกต้อง
        self.k_proj = nn.Linear(2 * self.d, self.d, bias=True)
        self.v_proj = nn.Linear(2 * self.d, self.d, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1     = nn.Linear(dim, dim)
        self.proj2     = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        B, N, C = x1.shape

        x1_n = self.norm1(x1)
        x2_n = self.norm2(x2)

        # QKV projection
        def _qkv(x, proj):
            return proj(x).reshape(B, N, 3, self.num_heads, self.d) \
                           .permute(2, 0, 3, 1, 4)

        q1, k1, v1 = _qkv(x1_n, self.qkv1).unbind(0)  # each [B, h, N, d]
        q2, k2, v2 = _qkv(x2_n, self.qkv2).unbind(0)

        # Shared Km, Vm via ChannelAttention
        k_cat = torch.cat([k1, k2], dim=-1)   # [B, h, N, 2d]
        v_cat = torch.cat([v1, v2], dim=-1)

        k_cat = self.se_k(k_cat)               # gated [B, h, N, 2d]
        v_cat = self.se_v(v_cat)

        Km = self.k_proj(k_cat)                # [B, h, N, d]
        Vm = self.v_proj(v_cat)                # [B, h, N, d]

        # Asymmetric attention — Q1, Q2 both attend Km, Vm
        def _attn(q):
            a = (q @ Km.transpose(-2, -1)) * self.scale
            a = self.attn_drop(a.softmax(dim=-1))
            return (a @ Vm).transpose(1, 2).reshape(B, N, C)

        x1_out = self.proj_drop(self.proj1(_attn(q1)))
        x2_out = self.proj_drop(self.proj2(_attn(q2)))

        return x1_out, x2_out

class CrossPath(nn.Module):
    """Dual-path cross fusion with MixAttention
    Matches FFMx Stage 2 in Zheng et al., 2024
    """
    def __init__(self, dim, reduction=1, num_heads=None,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1          = nn.ReLU(inplace=False)  
        self.act2          = nn.ReLU(inplace=False)
        self.cross_attn    = MixAttention(
            dim // reduction, num_heads=num_heads or max(1, dim // reduction // 64))
        self.end_proj1     = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2     = nn.Linear(dim // reduction * 2, dim)
        self.norm1         = norm_layer(dim)
        self.norm2         = norm_layer(dim)

    def forward(self, x1, x2):
        # Channel projection → split into (residual, attention input)
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        # Mix attention on u1, u2
        v1, v2 = self.cross_attn(u1, u2)
        # Concat residual + attended
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        # Residual + project + LayerNorm
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2

class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.out_channels  = out_channels
        self.residual      = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction,
                      kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction,
                      kernel_size=3, stride=1, padding=1, bias=True,
                      groups=out_channels // reduction),
            nn.ReLU(inplace=False),       
            nn.Conv2d(out_channels // reduction, out_channels,
                      kernel_size=1, bias=True),
            norm_layer(out_channels))
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x        = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x        = self.channel_embed(x)
        return self.norm(residual + x)

class FeatureFusionModule(nn.Module):
    """Mix Attention FFM — Zheng et al., Trans-SedNet, 2024
    Drop-in replacement สำหรับ CrossPath-based FFM ใน CMX
    """
    def __init__(self, dim, reduction=1, num_heads=None,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross       = CrossPath(
            dim=dim, reduction=reduction, num_heads=num_heads,
            norm_layer=nn.LayerNorm)
        self.channel_emb = ChannelEmbed(
            in_channels=dim * 2, out_channels=dim,
            reduction=reduction, norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        # flatten to sequence
        x1 = x1.flatten(2).transpose(1, 2)   # [B, N, C]
        x2 = x2.flatten(2).transpose(1, 2)
        # cross attention
        x1, x2 = self.cross(x1, x2)
        # concat + channel embed back to spatial
        merge = torch.cat((x1, x2), dim=-1)   # [B, N, 2C]
        merge = self.channel_emb(merge, H, W)  # [B, C, H, W]
        return merge