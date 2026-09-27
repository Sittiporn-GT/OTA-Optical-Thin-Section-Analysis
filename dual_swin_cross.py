# --------------------------------------------------------
# Dual Swin Transformer with Cross-Modal Window Attention
# Based on Swin Transformer (Liu et al., 2021)
# Extended with cross-modal attention for dual-modal segmentation
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


class Mlp(nn.Module):
    """ Multilayer perceptron."""

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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) with relative position bias.
    Supports both shifted and non-shifted windows.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim ** -0.5

        # Relative position bias table
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
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

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
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                   + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x    = self.proj(x)
        x    = self.proj_drop(x)
        return x


# ══════════════════════════════════════════════════════════════════════
# NEW: Cross-Modal Window Attention
# Q from source modality, K/V from target modality
# ══════════════════════════════════════════════════════════════════════

class CrossModalWindowAttention(nn.Module):
    """Cross-modal window attention: Q from modality A, K/V from modality B.

    This is the key novel contribution — instead of attending within the same
    modality (W-MSA), each window in one modality attends to the corresponding
    window in the other modality. This enables early-stage cross-modal fusion
    at the feature level, before the FFM.

    Key difference from WindowAttention:
        WindowAttention:       Q, K, V all from same modality x
        CrossModalWindowAttn:  Q from x_src, K and V from x_tgt

    Relative position bias is shared with self-attention (same window layout).
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim ** -0.5

        # ── Separate projections for Q (source) and KV (target) ──────
        self.q_proj  = nn.Linear(dim, dim, bias=qkv_bias)   # query from src
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)  # key+value from tgt

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # ── Relative position bias (same layout as W-MSA) ────────────
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
        relative_position_index  = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_src, x_tgt, mask=None):
        """
        Args:
            x_src: (nW*B, N, C) — query source (e.g. XPL windows)
            x_tgt: (nW*B, N, C) — key/value source (e.g. PPL windows)
            mask:  attention mask for cyclic shift (same as W-MSA)
        Returns:
            out: (nW*B, N, C) — cross-modal attended features
        """
        B_, N, C = x_src.shape

        # Q from src, K/V from tgt
        q  = self.q_proj(x_src).reshape(B_, N, self.num_heads,
                                         C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(x_tgt).reshape(B_, N, 2, self.num_heads,
                                          C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q    = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                   + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        out  = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out  = self.proj(out)
        out  = self.proj_drop(out)
        return out


# ══════════════════════════════════════════════════════════════════════
# NEW: Dual Swin Transformer Block
# Replaces SwinTransformerBlock — processes both modalities jointly
# ══════════════════════════════════════════════════════════════════════

class DualSwinTransformerBlock(nn.Module):
    """Dual-stream Swin Transformer Block with cross-modal attention.

    Each block processes XPL and PPL simultaneously:
        1. Self-attention  (W-MSA or SW-MSA) for each modality independently
        2. Cross-modal attention — XPL attends PPL, PPL attends XPL
        3. Gated fusion: out = self_out + alpha * cross_out
        4. Shared FFN (separate weights per modality)

    Args:
        alpha_init: initial value for learnable gating (default 0.1)
                    small value keeps self-attention dominant early in training
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 alpha_init=0.1):
        super().__init__()
        self.dim        = dim
        self.num_heads  = num_heads
        self.window_size = window_size
        self.shift_size  = shift_size
        self.mlp_ratio   = mlp_ratio
        assert 0 <= self.shift_size < self.window_size

        # ── Self-attention (one per modality) ───────────────────────
        self.norm1_rgb = norm_layer(dim)
        self.norm1_xpl = norm_layer(dim)
        self.attn_rgb  = WindowAttention(
            dim, window_size=to_2tuple(window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.attn_xpl  = WindowAttention(
            dim, window_size=to_2tuple(window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        # ── Cross-modal attention ────────────────────────────────────
        # rgb_cross: Q=PPL, K/V=XPL  (PPL queries XPL)  — wait, we want:
        # cross_rgb: Q=XPL, K/V=PPL  → enriches XPL features with PPL context
        # cross_xpl: Q=PPL, K/V=XPL  → enriches PPL features with XPL context
        self.cross_norm_rgb = norm_layer(dim)
        self.cross_norm_xpl = norm_layer(dim)
        self.cross_attn_rgb = CrossModalWindowAttention(
            dim, window_size=to_2tuple(window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn_xpl = CrossModalWindowAttention(
            dim, window_size=to_2tuple(window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        # ── Learnable gating: controls cross-modal contribution ──────
        # Initialized small so self-attention dominates early training
        self.alpha_rgb = nn.Parameter(torch.full((1,), alpha_init))
        self.alpha_xpl = nn.Parameter(torch.full((1,), alpha_init))

        # ── FFN (one per modality) ───────────────────────────────────
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_rgb = norm_layer(dim)
        self.norm2_xpl = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_rgb   = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=act_layer, drop=drop)
        self.mlp_xpl   = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def _window_attn(self, x, attn_module, mask_matrix, pad_r, pad_b, Hp, Wp):
        """Helper: apply window attention to a single-modality feature."""
        B, L, C = x.shape
        H, W = self.H, self.W

        x_2d = x.view(B, H, W, C)
        if pad_r > 0 or pad_b > 0:
            x_2d = F.pad(x_2d, (0, 0, 0, pad_r, 0, pad_b))

        if self.shift_size > 0:
            shifted = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted   = x_2d
            attn_mask = None

        windows     = window_partition(shifted, self.window_size)  # nW*B, ws, ws, C
        windows     = windows.view(-1, self.window_size * self.window_size, C)
        attn_out    = attn_module(windows, mask=attn_mask)
        attn_out    = attn_out.view(-1, self.window_size, self.window_size, C)
        shifted_out = window_reverse(attn_out, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            shifted_out = torch.roll(shifted_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        if pad_r > 0 or pad_b > 0:
            shifted_out = shifted_out[:, :H, :W, :].contiguous()

        return shifted_out.view(B, H * W, C)

    def _cross_window_attn(self, x_src, x_tgt, cross_attn_module,
                           mask_matrix, pad_r, pad_b, Hp, Wp):
        """Helper: apply cross-modal window attention."""
        B, L, C = x_src.shape
        H, W = self.H, self.W

        src_2d = x_src.view(B, H, W, C)
        tgt_2d = x_tgt.view(B, H, W, C)
        if pad_r > 0 or pad_b > 0:
            src_2d = F.pad(src_2d, (0, 0, 0, pad_r, 0, pad_b))
            tgt_2d = F.pad(tgt_2d, (0, 0, 0, pad_r, 0, pad_b))

        if self.shift_size > 0:
            src_shift  = torch.roll(src_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            tgt_shift  = torch.roll(tgt_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask  = mask_matrix
        else:
            src_shift  = src_2d
            tgt_shift  = tgt_2d
            attn_mask  = None

        src_windows = window_partition(src_shift, self.window_size).view(
            -1, self.window_size * self.window_size, C)
        tgt_windows = window_partition(tgt_shift, self.window_size).view(
            -1, self.window_size * self.window_size, C)

        cross_out   = cross_attn_module(src_windows, tgt_windows, mask=attn_mask)
        cross_out   = cross_out.view(-1, self.window_size, self.window_size, C)
        cross_2d    = window_reverse(cross_out, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            cross_2d = torch.roll(cross_2d, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        if pad_r > 0 or pad_b > 0:
            cross_2d = cross_2d[:, :H, :W, :].contiguous()

        return cross_2d.view(B, H * W, C)

    def forward(self, x_rgb, x_xpl, mask_matrix):
        """
        Args:
            x_rgb:        (B, H*W, C) XPL features
            x_xpl:        (B, H*W, C) PPL features
            mask_matrix:  attention mask for SW-MSA

        Returns:
            x_rgb_out, x_xpl_out: (B, H*W, C) each
        """
        B, L, C = x_rgb.shape
        H, W = self.H, self.W
        assert L == H * W

        # Compute padding
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        Hp    = H + pad_b
        Wp    = W + pad_r

        # ── 1. Self-attention ────────────────────────────────────────
        self_rgb = self._window_attn(self.norm1_rgb(x_rgb), self.attn_rgb,
                                     mask_matrix, pad_r, pad_b, Hp, Wp)
        self_xpl = self._window_attn(self.norm1_xpl(x_xpl), self.attn_xpl,
                                     mask_matrix, pad_r, pad_b, Hp, Wp)

        # ── 2. Cross-modal attention ─────────────────────────────────
        # RGB queries PPL context
        cross_rgb = self._cross_window_attn(
            self.cross_norm_rgb(x_rgb), self.cross_norm_xpl(x_xpl),
            self.cross_attn_rgb, mask_matrix, pad_r, pad_b, Hp, Wp)
        # PPL queries RGB context
        cross_xpl = self._cross_window_attn(
            self.cross_norm_xpl(x_xpl), self.cross_norm_rgb(x_rgb),
            self.cross_attn_xpl, mask_matrix, pad_r, pad_b, Hp, Wp)

        # ── 3. Gated fusion: self + alpha * cross ────────────────────
        alpha_rgb = torch.sigmoid(self.alpha_rgb)   # 0~1
        alpha_xpl = torch.sigmoid(self.alpha_xpl)

        x_rgb = x_rgb + self.drop_path(self_rgb + alpha_rgb * cross_rgb)
        x_xpl = x_xpl + self.drop_path(self_xpl + alpha_xpl * cross_xpl)

        # ── 4. FFN ───────────────────────────────────────────────────
        x_rgb = x_rgb + self.drop_path(self.mlp_rgb(self.norm2_rgb(x_rgb)))
        x_xpl = x_xpl + self.drop_path(self.mlp_xpl(self.norm2_xpl(x_xpl)))

        return x_rgb, x_xpl


# ══════════════════════════════════════════════════════════════════════
# NEW: Dual Basic Layer (replaces BasicLayer)
# ══════════════════════════════════════════════════════════════════════

class DualBasicLayer(nn.Module):
    """One stage of dual-stream Swin with cross-modal blocks.

    Replaces two separate BasicLayer instances — processes both
    XPL and PPL inside the same layer, enabling cross-modal
    window attention at every block.
    """

    def __init__(self, dim, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False,
                 alpha_init=0.1):
        super().__init__()
        self.window_size = window_size
        self.shift_size  = window_size // 2
        self.depth       = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            DualSwinTransformerBlock(
                dim=dim, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                alpha_init=alpha_init)
            for i in range(depth)])

    def forward(self, x_rgb, x_xpl, H, W):
        # Build SW-MSA mask
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x_rgb.device)
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
                                .masked_fill(attn_mask == 0,    0.0)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x_rgb, x_xpl = checkpoint.checkpoint(blk, x_rgb, x_xpl, attn_mask)
            else:
                x_rgb, x_xpl = blk(x_rgb, x_xpl, attn_mask)

        return x_rgb, x_xpl, H, W


# ══════════════════════════════════════════════════════════════════════
# Patch Merging, PatchEmbed (unchanged)
# ══════════════════════════════════════════════════════════════════════

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim       = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm      = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)
        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x  = self.norm(x)
        x  = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans   = in_chans
        self.embed_dim  = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

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


# ══════════════════════════════════════════════════════════════════════
# Dual Swin Transformer (main backbone)
# ══════════════════════════════════════════════════════════════════════

class DualSwinTransformer(nn.Module):
    """Dual Swin Transformer backbone with cross-modal window attention.

    Key difference from original dual_swin:
        - DualBasicLayer replaces paired BasicLayer instances
        - Cross-modal attention occurs inside every transformer block
        - Learnable alpha gates control the cross-modal contribution
        - FRM and FFM still used for final stage fusion

    Args:
        alpha_init: initial gating value for cross-modal attention (default 0.1)
    """

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 ape=False, patch_norm=True, out_indices=(0, 1, 2, 3),
                 frozen_stages=-1, use_checkpoint=False, alpha_init=0.1):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers    = len(depths)
        self.embed_dim     = embed_dim
        self.ape           = ape
        self.patch_norm    = patch_norm
        self.out_indices   = out_indices
        self.frozen_stages = frozen_stages

        # Patch embeddings (separate per modality)
        self.patch_embed   = PatchEmbed(patch_size, in_chans, embed_dim,
                                        norm_layer if patch_norm else None)
        self.patch_embed_d = PatchEmbed(patch_size, in_chans, embed_dim,
                                        norm_layer if patch_norm else None)

        if ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size_t      = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size_t[0],
                                   pretrain_img_size[1] // patch_size_t[1]]
            self.absolute_pos_embed   = nn.Parameter(torch.zeros(
                1, embed_dim, patches_resolution[0], patches_resolution[1]))
            self.absolute_pos_embed_d = nn.Parameter(torch.zeros(
                1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed,   std=.02)
            trunc_normal_(self.absolute_pos_embed_d, std=.02)

        self.pos_drop   = nn.Dropout(p=drop_rate)
        self.pos_drop_d = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ── DualBasicLayer (NEW) replaces paired BasicLayer ───────────
        self.dual_layers = nn.ModuleList()
        self.downsamples   = nn.ModuleList()
        self.downsamples_d = nn.ModuleList()
        self.FRMs = nn.ModuleList()
        self.FFMs = nn.ModuleList()

        for i_layer in range(self.num_layers):
            dual_layer = DualBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                alpha_init=alpha_init)
            self.dual_layers.append(dual_layer)

            self.FRMs.append(FRM(dim=int(embed_dim * 2 ** i_layer), reduction=1))
            self.FFMs.append(FFM(dim=int(embed_dim * 2 ** i_layer), reduction=1,
                                 num_heads=num_heads[i_layer], norm_layer=norm_fuse))

            if i_layer < self.num_layers - 1:
                self.downsamples.append(
                    PatchMerging(int(embed_dim * 2 ** i_layer), norm_layer))
                self.downsamples_d.append(
                    PatchMerging(int(embed_dim * 2 ** i_layer), norm_layer))

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
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.dual_layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
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

    def forward(self, x, x_d):
        x   = self.patch_embed(x)
        x_d = self.patch_embed_d(x_d)

        Wh, Ww = x.size(2), x.size(3)

        if self.ape:
            abs_emb   = F.interpolate(self.absolute_pos_embed,   size=(Wh, Ww), mode='bicubic')
            abs_emb_d = F.interpolate(self.absolute_pos_embed_d, size=(Wh, Ww), mode='bicubic')
            x   = (x   + abs_emb  ).flatten(2).transpose(1, 2)
            x_d = (x_d + abs_emb_d).flatten(2).transpose(1, 2)
        else:
            x   = x.flatten(2).transpose(1, 2)
            x_d = x_d.flatten(2).transpose(1, 2)

        x   = self.pos_drop(x)
        x_d = self.pos_drop_d(x_d)

        outs = []
        for i in range(self.num_layers):
            # ── DualBasicLayer: both streams processed jointly ────────
            x, x_d, H, W = self.dual_layers[i](x, x_d, Wh, Ww)

            x_out, x_out_d = x, x_d

            if i < self.num_layers - 1:
                x   = self.downsamples[i](x,   H, W)
                x_d = self.downsamples_d[i](x_d, H, W)
                Wh, Ww = (H + 1) // 2, (W + 1) // 2

            if i in self.out_indices:
                norm_layer_rgb = getattr(self, f'norm{i}')
                norm_layer_xpl = getattr(self, f'norm_d{i}')
                x_out   = norm_layer_rgb(x_out)
                x_out_d = norm_layer_xpl(x_out_d)

                x_out   = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                x_out_d = x_out_d.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

                x_out, x_out_d = self.FRMs[i](x_out, x_out_d)
                out = self.FFMs[i](x_out, x_out_d)
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        super(DualSwinTransformer, self).train(mode)
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
    """Load pretrained Swin weights into dual-stream model.
    Self-attention weights are copied to both RGB and XPL streams.
    Cross-modal attention weights are initialized from scratch (no pretrained).
    """
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location='cpu')
        if 'model' in raw_state_dict:
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        # Map pretrained Swin keys → DualBasicLayer keys
        if k.find('downsample') >= 0 and k.find('layer') >= 0:
            # e.g. layers.0.downsample.* → downsamples.0.*
            name = k.replace('downsample.', '')
            name = name.replace('layers', 'downsamples')
            state_dict[name] = v
            state_dict[name.replace('downsamples', 'downsamples_d')] = v

        elif k.find('layers') >= 0:
            # e.g. layers.0.blocks.0.attn.qkv.weight
            # → dual_layers.0.blocks.0.attn_rgb.qkv.weight  (XPL self-attn)
            # → dual_layers.0.blocks.0.attn_xpl.qkv.weight  (PPL self-attn)
            new_k = k.replace('layers', 'dual_layers')
            # Map .attn. → .attn_rgb. and .attn_xpl.
            if '.attn.' in new_k:
                state_dict[new_k.replace('.attn.', '.attn_rgb.')] = v
                state_dict[new_k.replace('.attn.', '.attn_xpl.')] = v
            elif '.norm1.' in new_k:
                state_dict[new_k.replace('.norm1.', '.norm1_rgb.')] = v
                state_dict[new_k.replace('.norm1.', '.norm1_xpl.')] = v
            elif '.norm2.' in new_k:
                state_dict[new_k.replace('.norm2.', '.norm2_rgb.')] = v
                state_dict[new_k.replace('.norm2.', '.norm2_xpl.')] = v
            elif '.mlp.' in new_k:
                state_dict[new_k.replace('.mlp.', '.mlp_rgb.')] = v
                state_dict[new_k.replace('.mlp.', '.mlp_xpl.')] = v
            else:
                state_dict[new_k] = v
            # Note: cross_attn_rgb, cross_attn_xpl are NOT initialized from pretrained
            # They start from random init (trunc_normal) and learn from scratch

        elif k.find('patch_embed') >= 0:
            state_dict[k] = v
            state_dict[k.replace('patch_embed', 'patch_embed_d')] = v

        elif k.find('norm') >= 0:
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