# --------------------------------------------------------
# Dual Swin Transformer with Cross-Modal Channel Gating
# Based on Swin Transformer (Liu et al., 2021)
#
# Key modification over original dual_swin.py:
#   CrossModalGate — SE-like lightweight gate applied after each stage
#   Uses global context of modality B to gate modality A channel-wise.
#   Inserted BEFORE FRM+FFM so features are pre-aligned before full fusion.
#
# Why this design:
#   - Does NOT change BasicLayer → all pretrained Swin weights load perfectly
#   - Only 8 new gate modules (4 stages × 2 modalities), ~few K params each
#   - Initialized near identity → stable at the start of training
#   - Provides cross-modal information EARLIER than FRM+FFM
# --------------------------------------------------------

import time
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from engine.logger import get_logger
from models.FRM_FFM import FeatureFusionModule as FFM
from models.FRM_FFM import FeatureRectifyModule as FRM

logger = get_logger()

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h        = torch.arange(self.window_size[0])
        coords_w        = torch.arange(self.window_size[1])
        coords          = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten  = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q    = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        attn = attn + rpb.permute(2, 0, 1).contiguous().unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                   + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim         = dim
        self.num_heads   = num_heads
        self.window_size = window_size
        self.shift_size  = shift_size
        assert 0 <= shift_size < window_size
        self.norm1     = norm_layer(dim)
        self.attn      = WindowAttention(
            dim, window_size=to_2tuple(window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.mlp       = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W    = self.H, self.W
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x; attn_mask = None

        x_windows  = window_partition(shifted_x, self.window_size).view(
            -1, self.window_size * self.window_size, C)
        attn_out   = self.attn(x_windows, mask=attn_mask).view(
            -1, self.window_size, self.window_size, C)
        shifted_x  = window_reverse(attn_out, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x.view(B, H * W, C))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = window_size // 2
        self.depth       = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt; cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = checkpoint.checkpoint(blk, x, attn_mask) if self.use_checkpoint else blk(x, attn_mask)
        return x, H, W


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm      = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = torch.cat([x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],
                       x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size  = to_2tuple(patch_size)
        self.proj   = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm   = norm_layer(embed_dim) if norm_layer else None
        self.patch_size = patch_size
        self.embed_dim  = embed_dim

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0: x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0: x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm:
            Wh, Ww = x.size(2), x.size(3)
            x = self.norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x

class CrossModalGate(nn.Module):
    """SE-like cross-modal channel gating.

    Uses global average pooling of source modality to compute a
    channel-wise gate for target modality. Applied AFTER each stage
    and BEFORE FRM+FFM — pre-aligns features before full fusion.

    Key design choices:
      - Global average pool: lightweight, no positional dependency
      - Initialized near 1.0: gate ≈ identity at start of training
      - sigmoid output: smooth, bounded gate [0, 1]
      - Very few parameters: dim × (dim//reduction) × 2

    Input/output: spatial feature map (B, C, H, W)
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        mid = max(dim // reduction, 16)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Sequential(
            nn.Linear(dim, mid, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(mid, dim, bias=True),
            nn.Tanh(),         
        )
        # Init: zero weights → tanh(0) = 0 → output = x*(1+0) = x
        # Exact identity at init, full gradient (tanh'(0) = 1.0)
        nn.init.zeros_(self.fc[-2].weight)
        nn.init.zeros_(self.fc[-2].bias)

    def forward(self, x_target: torch.Tensor,
                x_source: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_target: (B, C, H, W) — modality to be gated
            x_source: (B, C, H, W) — modality providing context
        Returns:
            (B, C, H, W) — gated x_target
        """
        B, C, H, W = x_source.shape
        ctx   = self.gap(x_source).view(B, C)      # (B, C) — global context
        delta = self.fc(ctx).view(B, C, 1, 1)      # tanh → range [-1, 1]
        return x_target * (1 + delta)              # residual gate — init = 1.0×x

class DualSwinTransformer(nn.Module):
    """Dual Swin Transformer with Cross-Modal Channel Gating.

    Modification over the baseline dual_swin.py:
        After each stage, before FRM+FFM, we apply CrossModalGate:
            x_rgb ← x_rgb * gate(x_xpl)   [XPL context gates RGB]
            x_xpl ← x_xpl * gate(x_rgb)   [RGB context gates XPL]

    This pre-aligns features from both modalities at each scale,
    giving FRM+FFM better-aligned inputs for final fusion.

    The BasicLayer (Swin self-attention blocks) is UNCHANGED →
    all pretrained weights load perfectly, no random-init interference.
    """

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 ape=False, patch_norm=True, out_indices=(0, 1, 2, 3),
                 frozen_stages=-1, use_checkpoint=False,
                 gate_reduction=4):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers    = len(depths)
        self.embed_dim     = embed_dim
        self.ape           = ape
        self.patch_norm    = patch_norm
        self.out_indices   = out_indices
        self.frozen_stages = frozen_stages

        self.patch_embed   = PatchEmbed(patch_size, in_chans, embed_dim,
                                        norm_layer if patch_norm else None)
        self.patch_embed_d = PatchEmbed(patch_size, in_chans, embed_dim,
                                        norm_layer if patch_norm else None)

        if ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size_t = to_2tuple(patch_size)
            pr = [pretrain_img_size[0] // patch_size_t[0],
                  pretrain_img_size[1] // patch_size_t[1]]
            self.absolute_pos_embed   = nn.Parameter(torch.zeros(1, embed_dim, pr[0], pr[1]))
            self.absolute_pos_embed_d = nn.Parameter(torch.zeros(1, embed_dim, pr[0], pr[1]))
            trunc_normal_(self.absolute_pos_embed,   std=.02)
            trunc_normal_(self.absolute_pos_embed_d, std=.02)

        self.pos_drop   = nn.Dropout(p=drop_rate)
        self.pos_drop_d = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers    = nn.ModuleList()
        self.layers_d  = nn.ModuleList()
        self.downsamples   = nn.ModuleList()
        self.downsamples_d = nn.ModuleList()
        self.FRMs = nn.ModuleList()
        self.FFMs = nn.ModuleList()

        # ── NEW: Cross-Modal Gates (one pair per stage) ───────────────
        self.cross_gates_rgb = nn.ModuleList()  # gates RGB using XPL context
        self.cross_gates_xpl = nn.ModuleList()  # gates XPL using RGB context

        for i_layer in range(self.num_layers):
            dim_i = int(embed_dim * 2 ** i_layer)

            self.layers.append(BasicLayer(
                dim=dim_i, depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=norm_layer, use_checkpoint=use_checkpoint))

            self.layers_d.append(BasicLayer(
                dim=dim_i, depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=norm_layer, use_checkpoint=use_checkpoint))

            # ── NEW ──────────────────────────────────────────────────
            self.cross_gates_rgb.append(CrossModalGate(dim_i, reduction=gate_reduction))
            self.cross_gates_xpl.append(CrossModalGate(dim_i, reduction=gate_reduction))

            self.FRMs.append(FRM(dim=dim_i, reduction=1))
            self.FFMs.append(FFM(dim=dim_i, reduction=1,
                                 num_heads=num_heads[i_layer], norm_layer=norm_fuse))

            if i_layer < self.num_layers - 1:
                self.downsamples.append(PatchMerging(dim_i, norm_layer))
                self.downsamples_d.append(PatchMerging(dim_i, norm_layer))

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        for i_layer in out_indices:
            self.add_module(f'norm{i_layer}',   norm_layer(num_features[i_layer]))
            self.add_module(f'norm_d{i_layer}', norm_layer(num_features[i_layer]))

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for p in self.patch_embed.parameters(): p.requires_grad = False
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(self.frozen_stages - 1):
                for m in [self.layers[i], self.layers_d[i]]:
                    m.eval()
                    for p in m.parameters(): p.requires_grad = False

    def _reset_gate_init(self):
        # Re-apply identity init for CrossModalGate AFTER self.apply()
        # apply() overwrites ALL Linear weights with trunc_normal,
        # breaking zero-init needed for tanh residual identity.
        # Must be called after every self.apply(_init_weights).
        for gate in list(self.cross_gates_rgb) + list(self.cross_gates_xpl):
            # fc[-2] is the last Linear before Tanh
            nn.init.zeros_(gate.fc[-2].weight)
            nn.init.zeros_(gate.fc[-2].bias)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            load_dualpath_model(self, pretrained)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        self._reset_gate_init()

    def forward(self, x, x_d):
        x   = self.patch_embed(x)
        x_d = self.patch_embed_d(x_d)
        Wh, Ww = x.size(2), x.size(3)

        if self.ape:
            x   = (x   + F.interpolate(self.absolute_pos_embed,   size=(Wh, Ww), mode='bicubic')).flatten(2).transpose(1, 2)
            x_d = (x_d + F.interpolate(self.absolute_pos_embed_d, size=(Wh, Ww), mode='bicubic')).flatten(2).transpose(1, 2)
        else:
            x   = x.flatten(2).transpose(1, 2)
            x_d = x_d.flatten(2).transpose(1, 2)

        x   = self.pos_drop(x)
        x_d = self.pos_drop_d(x_d)

        outs = []
        for i in range(self.num_layers):
            # ── Standard Swin self-attention (unchanged) ──────────────
            x,   H, W = self.layers[i](x,   Wh, Ww)
            x_d, _, _ = self.layers_d[i](x_d, Wh, Ww)

            x_out, x_out_d = x, x_d

            if i < self.num_layers - 1:
                x   = self.downsamples[i](x,   H, W)
                x_d = self.downsamples_d[i](x_d, H, W)
                Wh, Ww = (H + 1) // 2, (W + 1) // 2

            if i in self.out_indices:
                # Normalize
                x_out   = getattr(self, f'norm{i}')(x_out)
                x_out_d = getattr(self, f'norm_d{i}')(x_out_d)

                # Reshape to spatial (B, C, H, W)
                x_out   = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                x_out_d = x_out_d.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

                # ── NEW: Cross-Modal Gate (before FRM+FFM) ────────────
                # gate RGB ด้วย original XPL context
                # gate XPL ด้วย original RGB context (ไม่ใช่ gated RGB)
                x_out_new   = self.cross_gates_rgb[i](x_out,   x_out_d)
                x_out_d_new = self.cross_gates_xpl[i](x_out_d, x_out)
                x_out, x_out_d = x_out_new, x_out_d_new

                # FRM + FFM (unchanged)
                x_out, x_out_d = self.FRMs[i](x_out, x_out_d)
                outs.append(self.FFMs[i](x_out, x_out_d))

        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()


# ── Model variants ───────────────────────────────────────────────────
class swin_t(DualSwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(pretrain_img_size=224, patch_size=4, in_chans=3,
                         embed_dim=96, depths=[2, 2, 6, 2],
                         num_heads=[3, 6, 12, 24], window_size=7,
                         mlp_ratio=4., qkv_bias=True, qk_scale=None,
                         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                         norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                         out_indices=(0, 1, 2, 3), frozen_stages=-1,
                         use_checkpoint=False, **kwargs)

class swin_s(DualSwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(pretrain_img_size=224, patch_size=4, in_chans=3,
                         embed_dim=96, depths=[2, 2, 18, 2],
                         num_heads=[3, 6, 12, 24], window_size=7,
                         mlp_ratio=4., qkv_bias=True, qk_scale=None,
                         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                         norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                         out_indices=(0, 1, 2, 3), frozen_stages=-1,
                         use_checkpoint=False, **kwargs)

class swin_b(DualSwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(pretrain_img_size=384, patch_size=4, in_chans=3,
                         embed_dim=128, depths=[2, 2, 18, 2],
                         num_heads=[4, 8, 16, 32], window_size=12,
                         mlp_ratio=4., qkv_bias=True, qk_scale=None,
                         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                         norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                         out_indices=(0, 1, 2, 3), frozen_stages=-1,
                         use_checkpoint=False, **kwargs)

class swin_l(DualSwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(pretrain_img_size=384, patch_size=4, in_chans=3,
                         embed_dim=192, depths=[2, 2, 18, 2],
                         num_heads=[6, 12, 24, 48], window_size=12,
                         mlp_ratio=4., qkv_bias=True, qk_scale=None,
                         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                         norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                         out_indices=(0, 1, 2, 3), frozen_stages=-1,
                         use_checkpoint=False, **kwargs)


# ── Weight loading ───────────────────────────────────────────────────

def load_dualpath_model(model, model_file, is_restore=False):
    """Load pretrained Swin weights.
    BasicLayer weights → both layers and layers_d (same as original).
    CrossModalGate weights → NOT in pretrained, initialized separately.
    """
    t_start = time.time()
    raw_state_dict = torch.load(model_file, map_location='cpu') \
        if isinstance(model_file, str) else model_file
    if 'model' in raw_state_dict:
        raw_state_dict = raw_state_dict['model']

    state_dict = {}
    for k, v in raw_state_dict.items():
        if 'downsample' in k and 'layer' in k:
            name = k.replace('downsample.', '').replace('layers', 'downsamples')
            state_dict[name] = v
            state_dict[name.replace('downsamples', 'downsamples_d')] = v
        elif 'patch_embed' in k:
            state_dict[k] = v
            state_dict[k.replace('patch_embed', 'patch_embed_d')] = v
        elif 'layer' in k:
            state_dict[k] = v
            state_dict[k.replace('layers', 'layers_d')] = v
        elif 'norm' in k:
            state_dict[k] = v
            state_dict[k.replace('norm', 'norm_d')] = v

    t_ioend = time.time()
    if is_restore:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    del state_dict
    t_end = time.time()
    logger.info("Load model, Time usage:\n\tIO: {:.3f}s, initialize: {:.3f}s".format(
        t_ioend - t_start, t_end - t_ioend))
    return model