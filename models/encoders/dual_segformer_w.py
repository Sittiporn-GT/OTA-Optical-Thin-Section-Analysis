import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from torch.nn.init import trunc_normal_
from torch.nn.modules.utils import _pair as to_2tuple
from timm.models.layers import DropPath
from models.FRM_FFM import FeatureFusionModule as FFM
from models.FRM_FFM import FeatureRectifyModule as FRM

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
    """(B, N, C) → (B*nW, ws_h*ws_w, C)"""
    B, N, C = x.shape
    ws_h, ws_w = int(window_size[0]), int(window_size[1])
    x = x.view(B, H // ws_h, ws_h, W // ws_w, ws_w, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws_h * ws_w, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """(B*nW, ws_h*ws_w, C) → (B, N, C)"""
    ws_h, ws_w = int(window_size[0]), int(window_size[1])
    B = int(windows.shape[0] / (H * W / ws_h / ws_w))
    x = windows.view(B, H // ws_h, W // ws_w, ws_h, ws_w, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * W, -1)
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim       = dim
        self.num_heads = num_heads
        head_dim       = dim // num_heads
        self.scale     = qk_scale or head_dim ** -0.5

        self.q         = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv        = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio  = sr_ratio
        if sr_ratio > 1:
            self.sr   = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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

    def forward(self, x, H, W, is_window=False):
        """
        is_window=True  → window attention: ห้ามใช้ sr (H,W = window size ไม่ใช่ feature map)
        is_window=False → global attention: ใช้ sr ได้ตามปกติ
        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1 and not is_window:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v   = kv[0], kv[1]
        attn   = (q @ k.transpose(-2, -1)) * self.scale
        attn   = attn.softmax(dim=-1)
        attn   = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Transformer Block ตาม Trans-SedNet paper (Zheng et al., 2024)
    ใช้ window-attention แบบ local ไม่มี shifted window
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1, window_size=16):
        super().__init__()
        self.norm1     = norm_layer(dim)
        self.attn      = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop,
                                   proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.mlp       = Mlp(in_features=dim,
                             hidden_features=int(dim * mlp_ratio),
                             act_layer=act_layer, drop=drop)
        self.window_size = window_size
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
        shortcut   = x
        x_norm     = self.norm1(x)

        ws         = self.window_size
        window_size = [ws, ws]

        x_window   = window_partition(x_norm, window_size=window_size, H=H, W=W)

        attn_window = self.attn(x_window, ws, ws, is_window=True)

        window_x   = window_reverse(attn_window, window_size=window_size, H=H, W=W)

        x = shortcut + self.drop_path(window_x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        if not isinstance(img_size, tuple):
            img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.proj  = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                               stride=stride,
                               padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm  = nn.LayerNorm(embed_dim)
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
                 window_size=16, pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths      = depths

        logger.info(f"seg&window[{window_size},{window_size}]")

        self.patch_embed1 = OverlapPatchEmbed(img_size,       7, 4, in_chans,      embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size // 4,  3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size // 8,  3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3])

        self.extra_patch_embed1 = OverlapPatchEmbed(img_size,       7, 4, in_chans,      embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size // 4,  3, 2, embed_dims[0], embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size // 8,  3, 2, embed_dims[1], embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size // 16, 3, 2, embed_dims[2], embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        def make_layer(depth, dim, nh, mlpr, sr_r, dpr_start):
            return nn.ModuleList([
                Block(dim=dim, num_heads=nh, mlp_ratio=mlpr,
                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate,
                      drop_path=dpr[dpr_start + i],
                      norm_layer=norm_layer, sr_ratio=sr_r,
                      window_size=window_size)
                for i in range(depth)
            ])

        self.block1       = make_layer(depths[0], embed_dims[0], num_heads[0], mlp_ratios[0], sr_ratios[0], cur)
        self.extra_block1 = make_layer(depths[0], embed_dims[0], num_heads[0], mlp_ratios[0], sr_ratios[0], cur)
        self.norm1        = norm_layer(embed_dims[0])
        self.extra_norm1  = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2       = make_layer(depths[1], embed_dims[1], num_heads[1], mlp_ratios[1], sr_ratios[1], cur)
        self.extra_block2 = make_layer(depths[1], embed_dims[1], num_heads[1], mlp_ratios[1], sr_ratios[1], cur)
        self.norm2        = norm_layer(embed_dims[1])
        self.extra_norm2  = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3       = make_layer(depths[2], embed_dims[2], num_heads[2], mlp_ratios[2], sr_ratios[2], cur)
        self.extra_block3 = make_layer(depths[2], embed_dims[2], num_heads[2], mlp_ratios[2], sr_ratios[2], cur)
        self.norm3        = norm_layer(embed_dims[2])
        self.extra_norm3  = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4       = make_layer(depths[3], embed_dims[3], num_heads[3], mlp_ratios[3], sr_ratios[3], cur)
        self.extra_block4 = make_layer(depths[3], embed_dims[3], num_heads[3], mlp_ratios[3], sr_ratios[3], cur)
        self.norm4        = norm_layer(embed_dims[3])
        self.extra_norm4  = norm_layer(embed_dims[3])

        self.FRMs = nn.ModuleList([FRM(dim=d, reduction=1) for d in embed_dims])
        self.FFMs = nn.ModuleList([
            FFM(dim=embed_dims[i], reduction=1, num_heads=num_heads[i], norm_layer=norm_fuse)
            for i in range(4)
        ])

        self.apply(self._init_weights)

        if pretrained is not None:
            load_dualpath_model(self, pretrained)

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


# ── Model variants ───────────────────────────────────────────────
# window_size=16 ปลอดภัยกับ 512x512
# drop_rate=0.0 ตาม SegFormer paper

class mit_b0(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, window_size=16, **kwargs)

class mit_b1(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, window_size=16, **kwargs)

class mit_b2(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, window_size=16, **kwargs)

class mit_b3(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, window_size=16, **kwargs)

class mit_b4(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, window_size=16, **kwargs)

class mit_b5(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, window_size=16, **kwargs)
