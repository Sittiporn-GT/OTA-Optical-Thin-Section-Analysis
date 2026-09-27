import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from timm.models.layers import trunc_normal_
from torch.nn.init import xavier_uniform_, constant_
try:
    from torch.cuda import amp
except ImportError:
    amp = None


# ── Feature Rectify Module ──────────────────────────
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
        max = self.max_pool(x).view(B, self.dim * 2)
        y   = torch.cat((avg, max), dim=1)
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


# ── ConvGFU — Convolutional Gated Fusion Unit (จาก DeformCAT) ────

class ConvGFU(nn.Module):
    """Gated fusion + dimension reduction: [B,2C,H,W] → [B,C,H,W]"""
    def __init__(self, n_features: int, kernel_size: int = 3):
        super().__init__()
        self.n_features = n_features
        C2 = n_features * 2
        self.fc      = nn.Conv2d(C2, C2, kernel_size=1)
        self.dw_conv = nn.Conv2d(C2, n_features, kernel_size=kernel_size,
                                  padding=kernel_size // 2, groups=n_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2C, H, W]
        # ✅ BUG FIX: ส่ง x (2C) เข้า fc และ dw_conv โดยตรง ไม่ reshape
        # dw_conv: Conv2d(2C, C, groups=C) → 2C/C=2 channels per group ✅
        x1, x2 = torch.chunk(x, 2, dim=1)    # each [B, C, H, W]
        x_     = self.fc(x)                    # [B, 2C, H, W]
        alpha  = self.sigmoid(self.dw_conv(x_)) # [B, C, H, W]
        return x1 * alpha + x2 * (1 - alpha)   # [B, C, H, W]


# ── Deformable Cross Attention (adapted from DeformCAT) ───────────

class DeformableCrossAttention(nn.Module):
    """
    Deformable Cross Attention จาก DeformCAT (Hu et al., 2025)
    Adapted สำหรับ dual-modal segmentation (XPL/PPL mineral grain)

    query: [B, C, H, W] — modality ที่เป็น query
    source: [B, C, H, W] — modality ที่ sample K,V จาก
    """
    def __init__(self, n_features: int = 256, n_heads: int = 8,
                 n_points: int = 4, sampling_field: int = 7,
                 attn_drop: float = 0.1,
                 use_pos_in_offset: bool = True):
        super().__init__()
        assert n_features % n_heads == 0
        assert sampling_field % 2 == 1

        self.n_features      = n_features
        self.n_heads         = n_heads
        self.n_points        = n_points
        self.n_feat_per_head = n_features // n_heads
        self._scale          = math.sqrt(self.n_feat_per_head)
        self._SF             = sampling_field
        self._N_SAMPLERS     = (sampling_field - 1) // 2
        self._use_pos        = use_pos_in_offset

        _NG = n_points * 2

        # Offset network (ConvGFU + DWConv blocks + projection)
        offset_layers = [ConvGFU(n_features, kernel_size=3)]
        for _ in range(self._N_SAMPLERS - 1):
            offset_layers.append(nn.Sequential(
                nn.Conv2d(n_features, n_features, kernel_size=3,
                          padding=1, groups=n_features),
                nn.GroupNorm(_NG, n_features),
                nn.SiLU()
            ))
        offset_layers.append(nn.Conv2d(n_features, n_points * 2, kernel_size=1))
        self.offset_net = nn.Sequential(*offset_layers)

        # Q, K, V projections
        self.q_proj   = nn.Conv2d(n_features, n_features, 1, groups=n_heads)
        self.k_proj   = nn.Conv2d(n_features * n_points, n_features * n_points,
                                   1, groups=n_heads * n_points)
        self.v_proj   = nn.Conv2d(n_features * n_points, n_features * n_points,
                                   1, groups=n_heads * n_points)
        self.out_proj = nn.Conv2d(n_features, n_features, 1)

        # Kernel positional bias table (Swin-style)
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(1, n_heads, sampling_field, sampling_field))

        self.attn_drop = nn.Dropout(p=attn_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.q_proj.weight.data)
        constant_(self.q_proj.bias.data, 0.)
        xavier_uniform_(self.k_proj.weight.data)
        constant_(self.k_proj.bias.data, 0.)
        xavier_uniform_(self.v_proj.weight.data)
        constant_(self.v_proj.bias.data, 0.)
        xavier_uniform_(self.out_proj.weight.data)
        constant_(self.out_proj.bias.data, 0.)

    def _get_kernel_bias(self, actual_offsets, B, H, W):
        """Interpolate kernel positional bias ตาม sampling offset"""
        n_q    = H * W
        pos    = actual_offsets.view(B, self.n_points, n_q, 2) \
                               .transpose(1, 2) \
                               .reshape(B * n_q, 1, self.n_points, 2)
        bias   = F.grid_sample(
            self.rel_pos_bias.expand(B * n_q, -1, -1, -1),
            pos, mode='bilinear', align_corners=False)           # [B*n_q, h, 1, p]
        return bias.view(B, n_q, self.n_heads, self.n_points) \
                   .transpose(1, 2) \
                   .reshape(B * self.n_heads * n_q, 1, self.n_points)

    def forward(self, query: torch.Tensor, source: torch.Tensor,
                reference_points: torch.Tensor,
                pos_emb=None) -> torch.Tensor:
        B, C, H, W = query.shape
        n_q = H * W

        # Offset network
        q_emb = query + pos_emb if pos_emb is not None else query
        s_emb = source + pos_emb if (pos_emb is not None and self._use_pos) else source
        reception = torch.cat([q_emb, s_emb], dim=1)   # [B, 2C, H, W] → ConvGFU → [B,C,H,W]
        offsets   = self.offset_net(reception).tanh()   # [B, np*2, H, W]

        # Sampling positions
        offsets_reshaped = offsets.reshape(B * self.n_points, 2, H, W) \
                                  .permute(0, 2, 3, 1)              # [B*p, H, W, 2]
        scaler  = torch.as_tensor(
            [self._SF / 2 / W, self._SF / 2 / H],
            dtype=offsets.dtype, device=offsets.device)[None, None, None, :]
        offsets_reshaped = offsets_reshaped * scaler
        sampling_loc     = torch.clamp(reference_points + offsets_reshaped, -1, 1) \
                               .view(B, self.n_points, H, W, 2)

        # Bilinear sampling from source
        sampled = []
        for i in range(self.n_points):
            sampled.append(F.grid_sample(source, sampling_loc[:, i],
                                         mode='bilinear', align_corners=False))
        sampled = torch.stack(sampled, dim=1).view(B, self.n_points * C, H, W)

        # Q, K, V
        Q = self.q_proj(q_emb).reshape(B * self.n_heads, self.n_feat_per_head, n_q) \
                               .transpose(1, 2) \
                               .reshape(B * self.n_heads * n_q, 1, self.n_feat_per_head)
        K = self.k_proj(sampled).view(B, self.n_points, self.n_heads,
                                       self.n_feat_per_head, n_q) \
                                .permute(0, 2, 4, 1, 3) \
                                .reshape(B * self.n_heads * n_q,
                                          self.n_points, self.n_feat_per_head)
        V = self.v_proj(sampled).view(B, self.n_points, self.n_heads,
                                       self.n_feat_per_head, n_q) \
                                .permute(0, 2, 4, 1, 3) \
                                .reshape(B * self.n_heads * n_q,
                                          self.n_points, self.n_feat_per_head)

        # Kernel positional bias
        actual_offsets = sampling_loc.view(-1, H, W, 2) - reference_points
        kernel_bias    = self._get_kernel_bias(actual_offsets, B, H, W)

        # Scaled-dot attention + kernel bias
        attn = torch.matmul(Q, K.transpose(1, 2)) / self._scale + kernel_bias
        attn = self.attn_drop(F.softmax(attn, dim=2))

        out = torch.matmul(attn, V) \
                   .view(B * self.n_heads, n_q, self.n_feat_per_head) \
                   .transpose(1, 2) \
                   .reshape(B, C, H, W)
        return self.out_proj(out)


# ── Deformable FFM Layer ──────────────────────────────────────────

class DeformableCrossPath(nn.Module):
    """
    Dual deformable cross attention + FFN (ตาม DeformCAT Eq.8)
    XPL attend PPL และ PPL attend XPL พร้อมกัน
    """
    def __init__(self, dim: int, n_heads: int = 8, n_points: int = 4,
                 sampling_field: int = 7, attn_drop: float = 0.1,
                 ffn_drop: float = 0.1, ffn_ratio: int = 4,
                 use_pos_in_offset: bool = True):
        super().__init__()

        # Dual cross attention
        self.xpl_attn = DeformableCrossAttention(
            dim, n_heads, n_points, sampling_field, attn_drop, use_pos_in_offset)
        self.ppl_attn = DeformableCrossAttention(
            dim, n_heads, n_points, sampling_field, attn_drop, use_pos_in_offset)

        # LayerNorm (post cross-attn)
        self.norm_xpl_attn = nn.LayerNorm(dim)
        self.norm_ppl_attn = nn.LayerNorm(dim)

        # FFN
        self.xpl_ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio), nn.GELU(),
            nn.Dropout(ffn_drop),
            nn.Linear(dim * ffn_ratio, dim), nn.Dropout(ffn_drop))
        self.ppl_ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio), nn.GELU(),
            nn.Dropout(ffn_drop),
            nn.Linear(dim * ffn_ratio, dim), nn.Dropout(ffn_drop))

        self.norm_xpl_ffn = nn.LayerNorm(dim)
        self.norm_ppl_ffn = nn.LayerNorm(dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                pos_emb, reference_points) -> tuple:
        B, C, H, W = x1.shape

        # x1 (XPL) cross-attend x2 (PPL) + residual
        xpl_ca   = x1 + self.xpl_attn(x1, x2, reference_points, pos_emb)
        xpl_flat = xpl_ca.flatten(2).transpose(1, 2)
        xpl_flat = self.norm_xpl_attn(xpl_flat)
        xpl_flat = self.norm_xpl_ffn(xpl_flat + self.xpl_ffn(xpl_flat))
        x1_out   = xpl_flat.transpose(1, 2).view(B, C, H, W)

        # x2 (PPL) cross-attend x1 (XPL) + residual
        ppl_ca   = x2 + self.ppl_attn(x2, x1, reference_points, pos_emb)
        ppl_flat = ppl_ca.flatten(2).transpose(1, 2)
        ppl_flat = self.norm_ppl_attn(ppl_flat)
        ppl_flat = self.norm_ppl_ffn(ppl_flat + self.ppl_ffn(ppl_flat))
        x2_out   = ppl_flat.transpose(1, 2).view(B, C, H, W)

        return x1_out, x2_out


# ── Channel Embed ────────────────────────────────────

class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
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


# ── Feature Fusion Module ─────────────────────

class FeatureFusionModule(nn.Module):
    """
    Deformable Cross Attention FFM
    adapted from DeformCAT (Hu et al., IEEE TMM 2025)
    สำหรับ dual-modal mineral grain segmentation (XPL + PPL)
    """
    def __init__(self, dim: int, reduction: int = 1,
                 num_heads: int = None, norm_layer=nn.BatchNorm2d,
                 n_points: int = 4, sampling_field: int = 7,
                 attn_drop: float = 0.1, ffn_drop: float = 0.1,
                 use_pos_in_offset: bool = True):
        super().__init__()

        n_heads = num_heads if num_heads is not None else max(1, dim // 64)

        self.cross = DeformableCrossPath(
            dim=dim, n_heads=n_heads, n_points=n_points,
            sampling_field=sampling_field,
            attn_drop=attn_drop, ffn_drop=ffn_drop,
            use_pos_in_offset=use_pos_in_offset)

        self.channel_emb = ChannelEmbed(
            in_channels=dim * 2, out_channels=dim,
            reduction=reduction, norm_layer=norm_layer)

        # Sinusoidal positional embedding (Global Positional Bias)
        self._temperature = 10000
        self._scale       = 2 * math.pi
        self._n_features  = dim
        self._n_points    = n_points

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

    @torch.no_grad()
    def _get_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        """2D Sinusoidal positional embedding"""
        H, W = x.shape[2:]
        y_e  = torch.arange(H, dtype=x.dtype, device=x.device)[None, :, None].expand(1, H, W)
        x_e  = torch.arange(W, dtype=x.dtype, device=x.device)[None, None, :].expand(1, H, W)
        eps  = 1e-6
        y_e  = (y_e - 0.5) / (y_e[:, -1:, :] + eps) * self._scale
        x_e  = (x_e - 0.5) / (x_e[:, :, -1:] + eps) * self._scale
        dim_t = torch.arange(self._n_features // 2,
                              dtype=x.dtype, device=x.device)
        dim_t = self._temperature ** (2 * torch.div(dim_t, 2,
                    rounding_mode='trunc') / (self._n_features // 2))
        px = x_e[..., None] / dim_t
        py = y_e[..., None] / dim_t
        px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()),
                          dim=4).flatten(3)
        py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()),
                          dim=4).flatten(3)
        return torch.cat((py, px), dim=3).permute(0, 3, 1, 2)  # [1,C,H,W]

    @torch.no_grad()
    def _get_reference_points(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized reference grid [-1,1] สำหรับ bilinear sampling"""
        B, _, H, W = x.shape
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5/H, (H-0.5)/H, H, dtype=x.dtype, device=x.device),
            torch.linspace(0.5/W, (W-0.5)/W, W, dtype=x.dtype, device=x.device),
            indexing='ij')
        ref = torch.stack((ref_x[None], ref_y[None]), dim=-1) \
                   .expand(B * self._n_points, -1, -1, -1)  # [B*p, H, W, 2]
        return ref * 2 - 1

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x1.shape

        pos_emb          = self._get_pos_emb(x1)
        reference_points = self._get_reference_points(x1)

        x1_out, x2_out = self.cross(x1, x2, pos_emb, reference_points)

        # Channel embed: concat → fuse
        merge = torch.cat((x1_out, x2_out), dim=1)          # [B, 2C, H, W]
        merge = merge.flatten(2).transpose(1, 2)              # [B, N, 2C]
        merge = self.channel_emb(merge, H, W)                 # [B, C, H, W]
        return merge