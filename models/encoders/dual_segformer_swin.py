import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from torch.nn.init import trunc_normal_
from torch.nn.modules.utils import _pair as to_2tuple
from timm.models.layers import DropPath
from models.FRM_FFM_deformable import FeatureFusionModule as FFM
from models.FRM_FFM_deformable import FeatureRectifyModule as FRM

import math
import time

from engine.logger import get_logger

logger = get_logger()


class DWConv(nn.Module):
    """Depthwise convolution: (B N C) → (B N C)"""
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1    = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act    = act_layer()
        self.fc2    = nn.Linear(hidden_features, out_features)
        self.drop   = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size, H, W):
    """(B, N, C) → (B*nW, ws*ws, C)"""
    B, N, C = x.shape
    ws_h, ws_w = int(window_size[0]), int(window_size[1])
    x = x.view(B, H // ws_h, ws_h, W // ws_w, ws_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws_h * ws_w, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """(B*nW, ws*ws, C) → (B, N, C)"""
    ws_h, ws_w = int(window_size[0]), int(window_size[1])
    B = int(windows.shape[0] / (H * W / ws_h / ws_w))
    x = windows.view(B, H // ws_h, W // ws_w, ws_h, ws_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * W, -1)
    return x


def shift_attn_mask(H, W, window_size, shift_size, device, dtype=torch.float32):
    """Swin-style cyclic-shift attention mask → (nW, ws*ws, ws*ws)"""
    ws = int(window_size)
    ss = int(shift_size)
    img_mask = torch.zeros((1, H, W, 1), device=device, dtype=dtype)

    cnt = 0
    for h in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
        for w in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_tokens  = img_mask.view(1, H * W, 1)
    mask_windows = window_partition(mask_tokens, [ws, ws], H, W).squeeze(-1)  # (nW, ws*ws)
    attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)       # (nW, ws*ws, ws*ws)
    attn_mask    = attn_mask.masked_fill(attn_mask != 0, -100.0) \
                            .masked_fill(attn_mask == 0,    0.0)
    return attn_mask


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1, window_size=0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim        = dim
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.q         = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv        = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr   = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        # Relative Position Bias — ใช้เฉพาะ window attention (window_size > 0)
        if window_size > 0:
            ws = window_size
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
            )
            trunc_normal_(self.relative_position_bias_table, std=.02)

            coords_h    = torch.arange(ws)
            coords_w    = torch.arange(ws)
            coords      = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
            coords_flat = torch.flatten(coords, 1)

            relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += ws - 1
            relative_coords[:, :, 1] += ws - 1
            relative_coords[:, :, 0] *= 2 * ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.relative_position_bias_table = None
            self.relative_position_index      = None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def _get_rel_pos_bias(self):
        ws2  = self.window_size * self.window_size
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(ws2, ws2, self.num_heads)
        return bias.permute(2, 0, 1).contiguous()  # (num_heads, ws², ws²)

    def forward(self, x, H, W, attn_mask=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # ใช้ sr เฉพาะ global attention (window_size=0) เท่านั้น
        use_sr = (self.sr_ratio > 1) and (self.window_size == 0)
        if use_sr:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # บวก Relative Position Bias ก่อน softmax (window attention เท่านั้น)
        if self.relative_position_bias_table is not None:
            rel_bias = self._get_rel_pos_bias()       # (num_heads, ws², ws²)
            attn = attn + rel_bias.unsqueeze(0)        # (B*nW, num_heads, ws², ws²)

        if attn_mask is not None:
            attn_mask = attn_mask.to(device=attn.device, dtype=attn.dtype)
            nW  = attn_mask.shape[0]
            B_  = attn.shape[0]
            Nq  = attn.shape[2]
            if B_ % nW != 0:
                raise ValueError(f"attn batch ({B_}) not divisible by nW ({nW}).")
            attn = attn.view(B_ // nW, nW, self.num_heads, Nq, Nq)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(B_, self.num_heads, Nq, Nq)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer Block with Shifted Window Attention (Swin-style).
    shift_size=0  → W-MSA  (regular window)
    shift_size>0  → SW-MSA (shifted window)
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1,
                 window_size=16, shift_size=0):   # ✅ default shift_size=0
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, attn_drop=attn_drop,
                               proj_drop=drop, sr_ratio=sr_ratio,
                               window_size=window_size)  # ✅ ส่ง window_size เพื่อสร้าง bias table
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.mlp       = Mlp(in_features=dim,
                             hidden_features=int(dim * mlp_ratio),
                             act_layer=act_layer, drop=drop)

        self.window_size = int(window_size) if window_size is not None else 0
        self.shift_size  = int(shift_size)  if shift_size  is not None else 0
        self.register_buffer("attn_mask", None, persistent=False)
        self._mask_hw    = None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def forward(self, x, H, W):
        B, N, C = x.shape
        ws = self.window_size
        ss = self.shift_size

        x_norm = self.norm1(x)

        if ws > 0:
            # ── Cyclic shift (SW-MSA) ──────────────────────────────
            if ss > 0:
                x_view  = x_norm.view(B, H, W, C)
                x_shift = torch.roll(x_view, shifts=(-ss, -ss), dims=(1, 2)).view(B, N, C)
                if self.attn_mask is None or self._mask_hw != (H, W):
                    self.attn_mask = shift_attn_mask(H, W, ws, ss, x.device, dtype=torch.float32)
                    self._mask_hw  = (H, W)
                attn_mask = self.attn_mask
            else:
                # ── Regular window (W-MSA) ─────────────────────────
                x_shift   = x_norm
                attn_mask = None

            x_windows = window_partition(x_shift, [ws, ws], H, W)          # (B*nW, ws², C)
            attn_out  = self.attn(x_windows, ws, ws, attn_mask=attn_mask)
            x_merge   = window_reverse(attn_out, [ws, ws], H, W)            # (B, N, C)

            if ss > 0:
                x_merge = torch.roll(
                    x_merge.view(B, H, W, C), shifts=(ss, ss), dims=(1, 2)
                ).view(B, N, C)
        else:
            # Global attention fallback
            x_merge = self.attn(x_norm, H, W)

        x = x + self.drop_path(x_merge)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.H           = img_size[0] // patch_size[0]
        self.W           = img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class RGBXTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
                 window_size=16):   # ✅ BUG FIX #3: รับ window_size เป็น parameter
        super().__init__()
        self.num_classes = num_classes
        self.depths      = depths

        ws = window_size  # ใช้ชื่อสั้นภายใน

        # ── Patch embeddings ──────────────────────────────────────
        self.patch_embed1 = OverlapPatchEmbed(img_size,       7, 4, in_chans,      embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size // 4,  3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size // 8,  3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3])

        self.extra_patch_embed1 = OverlapPatchEmbed(img_size,       7, 4, in_chans,      embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size // 4,  3, 2, embed_dims[0], embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size // 8,  3, 2, embed_dims[1], embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3])

        # ── Stochastic depth ─────────────────────────────────────
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # ── ✅ BUG FIX #1: สลับ shift_size ทุก block (W-MSA / SW-MSA)
        # ── ✅ BUG FIX #2: ใช้ dpr[cur+i] ทุก stage
        def make_layer(depth, dim, nh, mlpr, sr_r, dpr_start, stage_ws):
            return nn.ModuleList([
                Block(
                    dim=dim, num_heads=nh, mlp_ratio=mlpr,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[dpr_start + i],
                    norm_layer=norm_layer, sr_ratio=sr_r,
                    window_size=stage_ws,
                    shift_size=0 if (i % 2 == 0) else stage_ws // 2
                )
                for i in range(depth)
            ])

        def make_global_layer(depth, dim, nh, mlpr, sr_r, dpr_start):
            return nn.ModuleList([
                Block(
                    dim=dim, num_heads=nh, mlp_ratio=mlpr,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[dpr_start + i],
                    norm_layer=norm_layer, sr_ratio=sr_r,
                    window_size=0,
                    shift_size=0
                )
                for i in range(depth)
            ])

        # V1: ws=16 ทุก stage
        self.block1       = make_layer(depths[0], embed_dims[0], num_heads[0], mlp_ratios[0], sr_ratios[0], cur, stage_ws=16)
        self.extra_block1 = make_layer(depths[0], embed_dims[0], num_heads[0], mlp_ratios[0], sr_ratios[0], cur, stage_ws=16)
        self.norm1        = norm_layer(embed_dims[0])
        self.extra_norm1  = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2       = make_layer(depths[1], embed_dims[1], num_heads[1], mlp_ratios[1], sr_ratios[1], cur, stage_ws=16)
        self.extra_block2 = make_layer(depths[1], embed_dims[1], num_heads[1], mlp_ratios[1], sr_ratios[1], cur, stage_ws=16)
        self.norm2        = norm_layer(embed_dims[1])
        self.extra_norm2  = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3       = make_layer(depths[2], embed_dims[2], num_heads[2], mlp_ratios[2], sr_ratios[2], cur, stage_ws=16)
        self.extra_block3 = make_layer(depths[2], embed_dims[2], num_heads[2], mlp_ratios[2], sr_ratios[2], cur, stage_ws=16)
        self.norm3        = norm_layer(embed_dims[2])
        self.extra_norm3  = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4       = make_layer(depths[3], embed_dims[3], num_heads[3], mlp_ratios[3], sr_ratios[3], cur, stage_ws=16)
        self.extra_block4 = make_layer(depths[3], embed_dims[3], num_heads[3], mlp_ratios[3], sr_ratios[3], cur, stage_ws=16)
        self.norm4        = norm_layer(embed_dims[3])
        self.extra_norm4  = norm_layer(embed_dims[3])

        # ── FRM + FFM ─────────────────────────────────────────────
        self.FRMs = nn.ModuleList([FRM(dim=d, reduction=1) for d in embed_dims])
        self.FFMs = nn.ModuleList([
            FFM(dim=embed_dims[i], reduction=1, num_heads=num_heads[i], norm_layer=norm_fuse)
            for i in range(4)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x_rgb, x_e):
        B = x_rgb.shape[0]
        outs = []

        # stage 1
        x_rgb, H, W = self.patch_embed1(x_rgb)
        x_e,   _, _ = self.extra_patch_embed1(x_e)
        for blk in self.block1:       x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block1: x_e   = blk(x_e,   H, W)
        x_rgb = self.norm1(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e   = self.extra_norm1(x_e).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.FRMs[0](x_rgb, x_e)
        outs.append(self.FFMs[0](x_rgb, x_e))

        # stage 2
        x_rgb, H, W = self.patch_embed2(x_rgb)
        x_e,   _, _ = self.extra_patch_embed2(x_e)
        for blk in self.block2:       x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block2: x_e   = blk(x_e,   H, W)
        x_rgb = self.norm2(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e   = self.extra_norm2(x_e).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.FRMs[1](x_rgb, x_e)
        outs.append(self.FFMs[1](x_rgb, x_e))

        # stage 3
        x_rgb, H, W = self.patch_embed3(x_rgb)
        x_e,   _, _ = self.extra_patch_embed3(x_e)
        for blk in self.block3:       x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block3: x_e   = blk(x_e,   H, W)
        x_rgb = self.norm3(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e   = self.extra_norm3(x_e).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.FRMs[2](x_rgb, x_e)
        outs.append(self.FFMs[2](x_rgb, x_e))

        # stage 4
        x_rgb, H, W = self.patch_embed4(x_rgb)
        x_e,   _, _ = self.extra_patch_embed4(x_e)
        for blk in self.block4:       x_rgb = blk(x_rgb, H, W)
        for blk in self.extra_block4: x_e   = blk(x_e,   H, W)
        x_rgb = self.norm4(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e   = self.extra_norm4(x_e).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = self.FRMs[3](x_rgb, x_e)
        outs.append(self.FFMs[3](x_rgb, x_e))

        return outs

    def forward(self, x_rgb, x_e):
        return self.forward_features(x_rgb, x_e)


def load_dualpath_model(model, model_file):
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('patch_embed') >= 0:
            state_dict[k] = v
            state_dict[k.replace('patch_embed', 'extra_patch_embed')] = v
        elif k.find('block') >= 0:
            state_dict[k] = v
            state_dict[k.replace('block', 'extra_block')] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v
            state_dict[k.replace('norm', 'extra_norm')] = v

    t_ioend = time.time()
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    t_end = time.time()
    logger.info("Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
        t_ioend - t_start, t_end - t_ioend))


# ── Model variants — window_size=16 ปลอดภัยกับ 512x512 ──────────
# (128÷16=8, 64÷16=4, 32÷16=2, 16÷16=1 → ลงตัวทุก stage)

class mit_b0(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, window_size=16)

class mit_b1(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, window_size=16)

class mit_b2(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, window_size=16)

class mit_b3(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, window_size=16)

class mit_b4(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, window_size=16)

class mit_b5(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, window_size=16)
