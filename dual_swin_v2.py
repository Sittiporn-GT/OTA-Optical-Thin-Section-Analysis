# --------------------------------------------------------
# Dual Swin Transformer V2
# Based on: "Swin Transformer V2: Scaling Up Capacity and Resolution"
# Liu et al., CVPR 2022 — https://arxiv.org/abs/2111.09883
#
# Adapted from dual_swin.py (V1) for dual-modal segmentation (XPL + PPL)
#
# V1 → V2 changes:
#   1. WindowAttention     → WindowAttentionV2
#      - Scaled Cosine Attention (replaces dot-product)
#      - Log-spaced Continuous Relative Position Bias via cpb_mlp (replaces discrete table)
#      - Separate Q/V bias (no K bias)
#   2. SwinTransformerBlock → SwinTransformerBlockV2
#      - Post-norm (replaces pre-norm)
#   3. PatchMerging
#      - Norm after reduction (replaces norm before reduction)
#   4. DualSwinTransformerV2
#      - _init_respostnorm() for stable post-norm training
# --------------------------------------------------------

import functools
import time
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from utils.load_utils import load_state_dict
from engine.logger import get_logger

from models.FRM_FFM import FeatureFusionModule as FFM
from models.FRM_FFM import FeatureRectifyModule as FRM

logger = get_logger()


# ── Mlp (unchanged from V1) ──────────────────────────────────────

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
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ── window_partition / window_reverse (unchanged from V1) ────────

def window_partition(x, window_size):
    """(B, H, W, C) → (nW*B, ws, ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
                   W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous() \
               .view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """(nW*B, ws, ws, C) → (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                        window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ── V2 Change 1: WindowAttentionV2 ───────────────────────────────

class WindowAttentionV2(nn.Module):
    """Window Attention for Swin Transformer V2.

    Key changes vs V1:
      a) Scaled Cosine Attention:
            attn = F.normalize(q) @ F.normalize(k)^T  / τ
         where τ = clamp(logit_scale, max=log(1/0.01)).exp()  (learnable per head)
      b) Log-spaced Continuous Relative Position Bias via cpb_mlp:
            bias = 16 * sigmoid(cpb_mlp(log_coords))
         → transfers across window sizes (unlike discrete table in V1)
      c) Separate Q/V bias — no K bias
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0., pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size          # (Wh, Ww)
        self.num_heads   = num_heads
        self.pretrained_window_size = pretrained_window_size

        # ── a) Learnable log temperature per head ─────────────────
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # ── b) cpb_mlp: 2 → 512 → num_heads ─────────────────────
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=False),             # inplace=False per our convention
            nn.Linear(512, num_heads, bias=False))

        # Log-spaced relative coordinate table
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w],
                           indexing='ij')           # ✅ indexing='ij'
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # (1, 2Wh-1, 2Ww-1, 2)

        # Normalize by pretrained window size if available
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * \
            torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # Relative position index (same as V1)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords   = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing='ij'))  # ✅ indexing='ij'
        coords_flatten   = torch.flatten(coords, 1)
        relative_coords  = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords  = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # ── c) Separate Q/V bias — no K bias ─────────────────────
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        # Build QKV with Q and V bias only (no K bias)
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat([
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias])
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads) \
                  .permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ── a) Scaled Cosine Attention ────────────────────────────
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(torch.tensor(1. / 0.01,
                          device=self.logit_scale.device))).exp()
        attn = attn * logit_scale

        # ── b) Continuous Position Bias via cpb_mlp ───────────────
        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + \
                   mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ── V2 Change 2: SwinTransformerBlockV2 ─────────────────────────

class SwinTransformerBlockV2(nn.Module):
    """Swin Transformer Block V2.

    Key change vs V1:
      Post-norm: norm applied AFTER attention/FFN + residual
        V1: x = shortcut + drop_path(norm1(attn(x)))
        V2: x = norm1(shortcut + drop_path(attn(x)))   ← here
    """

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 pretrained_window_size=0):
        super().__init__()
        self.dim         = dim
        self.num_heads   = num_heads
        self.window_size = window_size
        self.shift_size  = shift_size
        self.mlp_ratio   = mlp_ratio
        assert 0 <= self.shift_size < self.window_size

        self.attn = WindowAttentionV2(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Post-norm layers (V2)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W    = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # Pad to multiples of window_size (same as V1)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift (same as V1)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask  = None

        # Window partition (same as V1)
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA using WindowAttentionV2
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows (same as V1)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x    = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # ── V2 Post-norm ──────────────────────────────────────────
        # V1: x = shortcut + drop_path(x);  x = x + drop_path(mlp(norm2(x)))
        # V2: x = norm1(shortcut + drop_path(x));  x = norm2(x + drop_path(mlp(x)))
        x = self.norm1(shortcut + self.drop_path(x))
        x = self.norm2(x + self.drop_path(self.mlp(x)))

        return x


# ── V2 Change 3: PatchMerging ────────────────────────────────────

class PatchMerging(nn.Module):
    """Patch Merging Layer V2.

    Key change vs V1:
      Norm after reduction (V2) vs norm before reduction (V1)
        V1: norm(4C) → linear(4C→2C)
        V2: linear(4C→2C) → norm(2C)   ← here
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim       = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm      = norm_layer(2 * dim)    # V2: norm 2C (after reduction)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)

        # V2: linear first, then norm
        x = self.reduction(x)
        x = self.norm(x)
        return x


# ── BasicLayerV2 ─────────────────────────────────────────────────

class BasicLayerV2(nn.Module):
    """A basic Swin Transformer V2 layer for one stage."""

    def __init__(self, dim, depth, num_heads, window_size=8,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 use_checkpoint=False, pretrained_window_size=0):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = window_size // 2
        self.depth       = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlockV2(
                dim=dim, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

    def forward(self, x, H, W):
        # Compute attention mask for SW-MSA (same as V1)
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask    = attn_mask.masked_fill(attn_mask != 0, -100.0) \
                                .masked_fill(attn_mask == 0,   0.0)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        return x, H, W

    def _init_respostnorm(self):
        """Init post-norm weights=0 for stable training (V2 paper Sec 3.2)."""
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias,   0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias,   0)
            nn.init.constant_(blk.norm2.weight, 0)


# ── PatchEmbed (same as fixed V1 — F.pad instead of assert) ──────

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size      = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans   = in_chans
        self.embed_dim  = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim,
                               kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


# ── DualSwinTransformerV2 ─────────────────────────────────────────

class DualSwinTransformerV2(nn.Module):
    """Dual Swin Transformer V2 backbone for XPL + PPL segmentation.

    Identical structure to DualSwinTransformer (V1) except:
      - Uses BasicLayerV2 (with SwinTransformerBlockV2 + WindowAttentionV2)
      - Uses PatchMerging V2 (norm after reduction)
      - Calls _init_respostnorm() after weight init
      - pretrained_window_sizes parameter for cpb_mlp normalization
    """

    def __init__(self,
                 pretrain_img_size=256,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 norm_fuse=nn.BatchNorm2d,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pretrained_window_sizes=[0, 0, 0, 0]):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers  = len(depths)
        self.embed_dim   = embed_dim
        self.ape         = ape
        self.patch_norm  = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # Dual patch embeddings (XPL + PPL)
        self.patch_embed   = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        self.patch_embed_d = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        if ape:
            pretrain_img_size_t = to_2tuple(pretrain_img_size)
            patch_size_t        = to_2tuple(patch_size)
            patches_resolution  = [pretrain_img_size_t[0] // patch_size_t[0],
                                    pretrain_img_size_t[1] // patch_size_t[1]]
            self.absolute_pos_embed   = nn.Parameter(
                torch.zeros(1, embed_dim,
                            patches_resolution[0], patches_resolution[1]))
            self.absolute_pos_embed_d = nn.Parameter(
                torch.zeros(1, embed_dim,
                            patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            trunc_normal_(self.absolute_pos_embed_d, std=.02)

        self.pos_drop   = nn.Dropout(p=drop_rate)
        self.pos_drop_d = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build dual layers + FRM + FFM (same structure as V1)
        self.layers      = nn.ModuleList()
        self.layers_d    = nn.ModuleList()
        self.downsamples   = nn.ModuleList()
        self.downsamples_d = nn.ModuleList()
        self.FRMs = nn.ModuleList()
        self.FFMs = nn.ModuleList()

        for i_layer in range(self.num_layers):
            dim_i = int(embed_dim * 2 ** i_layer)

            layer = BasicLayerV2(
                dim=dim_i, depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

            layer_d = BasicLayerV2(
                dim=dim_i, depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers_d.append(layer_d)

            self.FRMs.append(FRM(dim=dim_i, reduction=1))

            if i_layer < self.num_layers - 1:
                self.downsamples.append(
                    PatchMerging(dim=dim_i, norm_layer=norm_layer))
                self.downsamples_d.append(
                    PatchMerging(dim=dim_i, norm_layer=norm_layer))

            self.FFMs.append(
                FFM(dim=dim_i, reduction=1,
                    num_heads=num_heads[i_layer], norm_layer=norm_fuse))

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        for i_layer in out_indices:
            self.add_module(f'norm{i_layer}',   norm_layer(num_features[i_layer]))
            self.add_module(f'norm_d{i_layer}', norm_layer(num_features[i_layer]))

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad   = False
            self.absolute_pos_embed_d.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(self.frozen_stages - 1):
                self.layers[i].eval()
                for p in self.layers[i].parameters():
                    p.requires_grad = False

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
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

        # ── V2: init post-norm weights=0 for stable training ──────
        for layer in self.layers:
            layer._init_respostnorm()
        for layer in self.layers_d:
            layer._init_respostnorm()

    def forward(self, x, x_d):
        """Forward function — identical structure to V1."""
        x   = self.patch_embed(x)
        x_d = self.patch_embed_d(x_d)

        Wh, Ww = x.size(2), x.size(3)

        if self.ape:
            abs_pos   = F.interpolate(self.absolute_pos_embed,
                                       size=(Wh, Ww), mode='bicubic')
            abs_pos_d = F.interpolate(self.absolute_pos_embed_d,
                                       size=(Wh, Ww), mode='bicubic')
            x   = (x   + abs_pos).flatten(2).transpose(1, 2)
            x_d = (x_d + abs_pos_d).flatten(2).transpose(1, 2)
        else:
            x   = x.flatten(2).transpose(1, 2)
            x_d = x_d.flatten(2).transpose(1, 2)

        x   = self.pos_drop(x)
        x_d = self.pos_drop_d(x_d)

        outs = []
        for i in range(self.num_layers):
            x,   H, W = self.layers[i](x,   Wh, Ww)
            x_d, _, _ = self.layers_d[i](x_d, Wh, Ww)

            x_out   = x
            x_out_d = x_d

            if i < self.num_layers - 1:
                x   = self.downsamples[i](x,   H, W)
                x_d = self.downsamples_d[i](x_d, H, W)
                Wh, Ww = (H + 1) // 2, (W + 1) // 2

            if i in self.out_indices:
                norm_rgb   = getattr(self, f'norm{i}')
                norm_modal = getattr(self, f'norm_d{i}')

                x_out   = norm_rgb(x_out).view(
                    -1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                x_out_d = norm_modal(x_out_d).view(
                    -1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

                # FRM after norm → consistent with FFM (Bug fix from V1)
                x_out, x_out_d = self.FRMs[i](x_out, x_out_d)
                outs.append(self.FFMs[i](x_out, x_out_d))

        return tuple(outs)

    def train(self, mode=True):
        super(DualSwinTransformerV2, self).train(mode)
        self._freeze_stages()


# ── Model variants ────────────────────────────────────────────────
# window_size=8 → 512÷8=64 exact at all stages
# pretrained: swinv2_tiny/small/base_patch4_window8_256.pth

class swinv2_t(DualSwinTransformerV2):
    def __init__(self, **kwargs):
        super().__init__(
            pretrain_img_size=256, patch_size=4, in_chans=3,
            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
            window_size=8, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            out_indices=(0, 1, 2, 3), frozen_stages=-1,
            pretrained_window_sizes=[8, 8, 8, 8])


class swinv2_s(DualSwinTransformerV2):
    def __init__(self, **kwargs):
        super().__init__(
            pretrain_img_size=256, patch_size=4, in_chans=3,
            embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
            window_size=8, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            out_indices=(0, 1, 2, 3), frozen_stages=-1,
            pretrained_window_sizes=[8, 8, 8, 8])


class swinv2_b(DualSwinTransformerV2):
    def __init__(self, **kwargs):
        super().__init__(
            pretrain_img_size=256, patch_size=4, in_chans=3,
            embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
            window_size=8, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.5,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            out_indices=(0, 1, 2, 3), frozen_stages=-1,
            pretrained_window_sizes=[8, 8, 8, 8])


class swinv2_l(DualSwinTransformerV2):
    def __init__(self, **kwargs):
        super().__init__(
            pretrain_img_size=256, patch_size=4, in_chans=3,
            embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
            window_size=8, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.5,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            out_indices=(0, 1, 2, 3), frozen_stages=-1,
            pretrained_window_sizes=[8, 8, 8, 8])


# ── load_dualpath_model (same as V1) ─────────────────────────────

def load_dualpath_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('downsample') >= 0 and k.find('layer') >= 0:
            name = k.replace('downsample.', '').replace('layers', 'downsamples')
            state_dict[name] = v
            state_dict[name.replace('downsamples', 'downsamples_d')] = v
        elif k.find('patch_embed') >= 0:
            state_dict[k] = v
            state_dict[k.replace('patch_embed', 'patch_embed_d')] = v
        elif k.find('layer') >= 0:
            state_dict[k] = v
            state_dict[k.replace('layers', 'layers_d')] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v
            state_dict[k.replace('norm', 'norm_d')] = v

    t_ioend = time.time()

    if is_restore:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    del state_dict
    t_end = time.time()
    logger.info("Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
        t_ioend - t_start, t_end - t_ioend))
    return model