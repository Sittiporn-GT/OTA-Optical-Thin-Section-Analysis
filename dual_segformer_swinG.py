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
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0,2,1).reshape(B,C,H,W).contiguous()
        x = self.dwconv(x)
        return x.flatten(2).transpose(1,2)


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
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0]*m.kernel_size[1]*m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0/fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x); x = self.dwconv(x,H,W); x = self.act(x)
        x = self.drop(x); x = self.fc2(x); x = self.drop(x)
        return x


def window_partition(x, window_size, H, W):
    """(B, N, C) → (B*nW, ws*ws, C)"""
    B, N, C = x.shape
    ws_h, ws_w = int(window_size[0]), int(window_size[1])
    x = x.view(B, H//ws_h, ws_h, W//ws_w, ws_w, C)
    return x.permute(0,1,3,2,4,5).contiguous().view(-1, ws_h*ws_w, C)


def window_reverse(windows, window_size, H, W):
    """(B*nW, ws*ws, C) → (B, N, C)"""
    ws_h, ws_w = int(window_size[0]), int(window_size[1])
    B = int(windows.shape[0] / (H*W/ws_h/ws_w))
    x = windows.view(B, H//ws_h, W//ws_w, ws_h, ws_w, -1)
    return x.permute(0,1,3,2,4,5).contiguous().view(B, H*W, -1)


def shift_attn_mask(H, W, window_size, shift_size, device, dtype=torch.float32):
    ws, ss = int(window_size), int(shift_size)
    assert ss < ws, \
        f"shift_size ({ss}) must be smaller than window_size ({ws})"
    img_mask = torch.zeros((1,H,W,1), device=device, dtype=dtype)
    cnt = 0
    for h in (slice(0,-ws), slice(-ws,-ss), slice(-ss,None)):
        for w in (slice(0,-ws), slice(-ws,-ss), slice(-ss,None)):
            img_mask[:,h,w,:] = cnt; cnt += 1
    mask_tokens  = img_mask.view(1, H*W, 1)
    mask_windows = window_partition(mask_tokens, [ws,ws], H, W).squeeze(-1)
    attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask!=0, -100.0).masked_fill(attn_mask==0, 0.0)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1, window_size=0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads  = num_heads
        self.scale      = qk_scale or (dim//num_heads)**-0.5
        self.window_size = window_size
        self.q  = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio  = sr_ratio
        if sr_ratio > 1:
            self.sr   = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        if window_size > 0:
            ws = window_size
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2*ws-1)*(2*ws-1), num_heads))
            trunc_normal_(self.relative_position_bias_table, std=.02)
            coords_h = torch.arange(ws); coords_w = torch.arange(ws)
            coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
            coords_flat = torch.flatten(coords, 1)
            rc = coords_flat[:,:,None] - coords_flat[:,None,:]
            rc = rc.permute(1,2,0).contiguous()
            rc[:,:,0] += ws-1; rc[:,:,1] += ws-1; rc[:,:,0] *= 2*ws-1
            self.register_buffer("relative_position_index", rc.sum(-1))
        else:
            self.relative_position_bias_table = None
            self.relative_position_index      = None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0]*m.kernel_size[1]*m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0/fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def _get_rel_pos_bias(self):
        ws2  = self.window_size * self.window_size
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(ws2, ws2, self.num_heads)
        return bias.permute(2,0,1).contiguous()

    def forward(self, x, H, W, attn_mask=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3)
        use_sr = (self.sr_ratio > 1) and (self.window_size == 0)
        if use_sr:
            x_ = self.norm(self.sr(x.permute(0,2,1).reshape(B,C,H,W)).reshape(B,C,-1).permute(0,2,1))
            kv = self.kv(x_).reshape(B,-1,2,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        else:
            kv = self.kv(x).reshape(B,-1,2,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        if self.relative_position_bias_table is not None:
            attn = attn + self._get_rel_pos_bias().unsqueeze(0)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device=attn.device, dtype=attn.dtype)
            nW, B_ = attn_mask.shape[0], attn.shape[0]
            Nq = attn.shape[2]
            attn = attn.view(B_//nW, nW, self.num_heads, Nq, Nq)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(B_, self.num_heads, Nq, Nq)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1, window_size=16, shift_size=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, attn_drop=attn_drop,
                               proj_drop=drop, sr_ratio=sr_ratio,
                               window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        self.mlp       = Mlp(dim, int(dim*mlp_ratio), act_layer=act_layer, drop=drop)
        self.window_size = int(window_size) if window_size else 0
        self.shift_size  = int(shift_size)  if shift_size  else 0
        self.register_buffer("attn_mask", None, persistent=False)
        self._mask_hw = None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0]*m.kernel_size[1]*m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0/fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        ws = self.window_size
        ss = self.shift_size
        x_norm = self.norm1(x)

        if ws > 0:
            x_2d = x_norm.view(B, H, W, C)

            pad_b = (ws - H % ws) % ws
            pad_r = (ws - W % ws) % ws
            if pad_b > 0 or pad_r > 0:
                x_2d = F.pad(x_2d, (0, 0, 0, pad_r, 0, pad_b))
            _, Hp, Wp, _ = x_2d.shape

            if ss > 0:
                x_shift = torch.roll(x_2d, shifts=(-ss,-ss), dims=(1,2))
                if self.attn_mask is None or self._mask_hw != (Hp, Wp):
                    self.attn_mask = shift_attn_mask(Hp, Wp, ws, ss, x.device)
                    self._mask_hw  = (Hp, Wp)
                attn_mask = self.attn_mask
            else:
                x_shift = x_2d; attn_mask = None

            # window partition on padded (Hp, Wp)
            x_seq     = x_shift.view(B, Hp*Wp, C)
            x_windows = window_partition(x_seq, [ws,ws], Hp, Wp)
            attn_out  = self.attn(x_windows, ws, ws, attn_mask=attn_mask)
            x_merge   = window_reverse(attn_out, [ws,ws], Hp, Wp)   # (B, Hp*Wp, C)

            if ss > 0:
                x_merge = torch.roll(
                    x_merge.view(B,Hp,Wp,C), shifts=(ss,ss), dims=(1,2)
                ).view(B, Hp*Wp, C)

            if pad_b > 0 or pad_r > 0:
                x_merge = x_merge.view(B,Hp,Wp,C)[:,:H,:W,:].contiguous().view(B,H*W,C)
        else:
            x_merge = self.attn(x_norm, H, W)

        x = x + self.drop_path(x_merge)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size   = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0]//2, patch_size[1]//2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0]*m.kernel_size[1]*m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0/fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x, H, W


class CrossModalGate(nn.Module):
    """SE-like cross-modal gate: gate x_target using global context of x_source.
    Tanh residual: output = x_target * (1 + delta), delta=0 at init → identity.
    NOTE: call _reset_gate_init() after self.apply() to preserve zero init.
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 16)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=False),    
            nn.Linear(hidden, dim),
            nn.Tanh()                  
        )
        nn.init.zeros_(self.fc[-2].weight)
        nn.init.zeros_(self.fc[-2].bias)

    def forward(self, x_target, x_source):
        B, C, H, W = x_source.shape
        ctx   = self.gap(x_source).view(B, C)
        delta = self.fc(ctx).view(B, C, 1, 1)
        return x_target * (1.0 + delta)


class RGBXTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dims=[64,128,256,512], num_heads=[1,2,4,8],
                 mlp_ratios=[4,4,4,4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, norm_fuse=nn.BatchNorm2d,
                 depths=[3,4,6,3], sr_ratios=[8,4,2,1],
                 window_size=16):
        super().__init__()
        self.num_classes = num_classes
        self.depths      = depths

        # ✅ configurable stage window/shift sizes (แทน hard-code)
        self.stage_windows = [8, 8, 4, 4]
        self.stage_shifts  = [4, 4, 2, 2]

        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(img_size,      7, 4, in_chans,     embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size//4,   3, 2, embed_dims[0],embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size//8,   3, 2, embed_dims[1],embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size//16,  3, 2, embed_dims[2],embed_dims[3])
        self.extra_patch_embed1 = OverlapPatchEmbed(img_size,      7, 4, in_chans,     embed_dims[0])
        self.extra_patch_embed2 = OverlapPatchEmbed(img_size//4,   3, 2, embed_dims[0],embed_dims[1])
        self.extra_patch_embed3 = OverlapPatchEmbed(img_size//8,   3, 2, embed_dims[1],embed_dims[2])
        self.extra_patch_embed4 = OverlapPatchEmbed(img_size//16,  3, 2, embed_dims[2],embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        def make_layer(depth, dim, nh, mlpr, sr_r, dpr_start, stage_ws, stage_ss):
            return nn.ModuleList([
                Block(dim=dim, num_heads=nh, mlp_ratio=mlpr,
                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate,
                      drop_path=dpr[dpr_start + i],
                      norm_layer=norm_layer, sr_ratio=sr_r,
                      window_size=stage_ws,
                      shift_size=0 if (i % 2 == 0) else stage_ss)
                for i in range(depth)])

        # ✅ ใช้ self.stage_windows แทน hard-code
        self.block1 = make_layer(depths[0],embed_dims[0],num_heads[0],mlp_ratios[0],sr_ratios[0],cur,self.stage_windows[0],self.stage_shifts[0])
        self.extra_block1 = make_layer(depths[0],embed_dims[0],num_heads[0],mlp_ratios[0],sr_ratios[0],cur,self.stage_windows[0],self.stage_shifts[0])
        self.norm1 = norm_layer(embed_dims[0]); self.extra_norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = make_layer(depths[1],embed_dims[1],num_heads[1],mlp_ratios[1],sr_ratios[1],cur,self.stage_windows[1],self.stage_shifts[1])
        self.extra_block2 = make_layer(depths[1],embed_dims[1],num_heads[1],mlp_ratios[1],sr_ratios[1],cur,self.stage_windows[1],self.stage_shifts[1])
        self.norm2 = norm_layer(embed_dims[1]); self.extra_norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3 = make_layer(depths[2],embed_dims[2],num_heads[2],mlp_ratios[2],sr_ratios[2],cur,self.stage_windows[2],self.stage_shifts[2])
        self.extra_block3 = make_layer(depths[2],embed_dims[2],num_heads[2],mlp_ratios[2],sr_ratios[2],cur,self.stage_windows[2],self.stage_shifts[2])
        self.norm3 = norm_layer(embed_dims[2]); self.extra_norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4 = make_layer(depths[3],embed_dims[3],num_heads[3],mlp_ratios[3],sr_ratios[3],cur,self.stage_windows[3],self.stage_shifts[3])
        self.extra_block4 = make_layer(depths[3],embed_dims[3],num_heads[3],mlp_ratios[3],sr_ratios[3],cur,self.stage_windows[3],self.stage_shifts[3])
        self.norm4 = norm_layer(embed_dims[3]); self.extra_norm4 = norm_layer(embed_dims[3])

        # Cross-modal gates (before FRM, per stage)
        self.cross_rgb   = nn.ModuleList([CrossModalGate(d) for d in embed_dims])
        self.cross_extra = nn.ModuleList([CrossModalGate(d) for d in embed_dims])

        self.FRMs = nn.ModuleList([FRM(dim=d, reduction=1) for d in embed_dims])
        self.FFMs = nn.ModuleList([
            FFM(dim=embed_dims[i], reduction=1, num_heads=num_heads[i], norm_layer=norm_fuse)
            for i in range(4)])

        self.apply(self._init_weights)
        # ✅ FIX: re-apply gate zero init AFTER apply() overwrites it
        self._reset_gate_init()

    def _reset_gate_init(self):
        """Re-zero CrossModalGate last linear after self.apply() overwrites it."""
        for gate in list(self.cross_rgb) + list(self.cross_extra):
            nn.init.zeros_(gate.fc[-2].weight)
            nn.init.zeros_(gate.fc[-2].bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0]*m.kernel_size[1]*m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0/fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        else:
            raise TypeError('pretrained must be a str or None')
        # NOTE: _reset_gate_init() is NOT called here intentionally
        # Calling it after resume checkpoint would overwrite learned gate weights

    def _run_stage(self, x_rgb, x_e, blocks_rgb, blocks_extra,
                   patch_embed_rgb, patch_embed_extra,
                   norm_rgb, norm_extra, gate_rgb, gate_extra,
                   frm, ffm, B):
        """Helper: run one stage (patch embed → blocks → norm → gate → FRM → FFM)"""
        x_rgb, H, W = patch_embed_rgb(x_rgb)
        x_e,   _, _ = patch_embed_extra(x_e)

        for blk in blocks_rgb:   x_rgb = blk(x_rgb, H, W)
        for blk in blocks_extra: x_e   = blk(x_e,   H, W)

        x_rgb = norm_rgb(x_rgb).reshape(B, H, W, -1).permute(0,3,1,2).contiguous()
        x_e   = norm_extra(x_e).reshape(B, H, W, -1).permute(0,3,1,2).contiguous()

        # ✅ CrossModalGate ก่อน FRM: ใช้ original ทั้งคู่ก่อน assign
        rgb_new   = gate_rgb(x_rgb, x_e)
        extra_new = gate_extra(x_e, x_rgb)
        x_rgb, x_e = rgb_new, extra_new

        x_rgb, x_e = frm(x_rgb, x_e)
        out = ffm(x_rgb, x_e)
        return x_rgb, x_e, out

    def forward_features(self, x_rgb, x_e):
        B    = x_rgb.shape[0]
        outs = []

        x_rgb, x_e, out = self._run_stage(
            x_rgb, x_e,
            self.block1, self.extra_block1,
            self.patch_embed1, self.extra_patch_embed1,
            self.norm1, self.extra_norm1,
            self.cross_rgb[0], self.cross_extra[0],
            self.FRMs[0], self.FFMs[0], B)
        outs.append(out)

        x_rgb, x_e, out = self._run_stage(
            x_rgb, x_e,
            self.block2, self.extra_block2,
            self.patch_embed2, self.extra_patch_embed2,
            self.norm2, self.extra_norm2,
            self.cross_rgb[1], self.cross_extra[1],
            self.FRMs[1], self.FFMs[1], B)
        outs.append(out)

        x_rgb, x_e, out = self._run_stage(
            x_rgb, x_e,
            self.block3, self.extra_block3,
            self.patch_embed3, self.extra_patch_embed3,
            self.norm3, self.extra_norm3,
            self.cross_rgb[2], self.cross_extra[2],
            self.FRMs[2], self.FFMs[2], B)
        outs.append(out)

        x_rgb, x_e, out = self._run_stage(
            x_rgb, x_e,
            self.block4, self.extra_block4,
            self.patch_embed4, self.extra_patch_embed4,
            self.norm4, self.extra_norm4,
            self.cross_rgb[3], self.cross_extra[3],
            self.FRMs[3], self.FFMs[3], B)
        outs.append(out)

        return outs

    def forward(self, x_rgb, x_e):
        return self.forward_features(x_rgb, x_e)


def load_dualpath_model(model, model_file):
    t_start = time.time()
    raw_state_dict = torch.load(model_file, map_location='cpu') \
        if isinstance(model_file, str) else model_file
    if 'model' in raw_state_dict:
        raw_state_dict = raw_state_dict['model']

    state_dict = {}
    for k, v in raw_state_dict.items():
        if 'patch_embed' in k:
            state_dict[k] = v
            state_dict[k.replace('patch_embed','extra_patch_embed')] = v
        elif 'block' in k:
            state_dict[k] = v
            state_dict[k.replace('block','extra_block')] = v
        elif 'norm' in k:
            state_dict[k] = v
            state_dict[k.replace('norm','extra_norm')] = v

    t_ioend = time.time()
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    t_end = time.time()
    logger.info("Load model, Time usage:\n\tIO: {:.3f}s, init: {:.3f}s".format(
        t_ioend-t_start, t_end-t_ioend))


class mit_b0(RGBXTransformer):
    def __init__(self, **kwargs):
        super().__init__(patch_size=4, embed_dims=[32,64,160,256], num_heads=[1,2,5,8],
            mlp_ratios=[4,4,4,4], qkv_bias=True, norm_layer=partial(nn.LayerNorm,eps=1e-6),
            depths=[2,2,2,2], sr_ratios=[8,4,2,1], drop_rate=0., drop_path_rate=0.1)

class mit_b1(RGBXTransformer):
    def __init__(self, **kwargs):
        super().__init__(patch_size=4, embed_dims=[64,128,320,512], num_heads=[1,2,5,8],
            mlp_ratios=[4,4,4,4], qkv_bias=True, norm_layer=partial(nn.LayerNorm,eps=1e-6),
            depths=[2,2,2,2], sr_ratios=[8,4,2,1], drop_rate=0., drop_path_rate=0.1)

class mit_b2(RGBXTransformer):
    def __init__(self, **kwargs):
        super().__init__(patch_size=4, embed_dims=[64,128,320,512], num_heads=[1,2,5,8],
            mlp_ratios=[4,4,4,4], qkv_bias=True, norm_layer=partial(nn.LayerNorm,eps=1e-6),
            depths=[3,4,6,3], sr_ratios=[8,4,2,1], drop_rate=0., drop_path_rate=0.1)

class mit_b3(RGBXTransformer):
    def __init__(self, **kwargs):
        super().__init__(patch_size=4, embed_dims=[64,128,320,512], num_heads=[1,2,5,8],
            mlp_ratios=[4,4,4,4], qkv_bias=True, norm_layer=partial(nn.LayerNorm,eps=1e-6),
            depths=[3,4,18,3], sr_ratios=[8,4,2,1], drop_rate=0., drop_path_rate=0.1)

class mit_b4(RGBXTransformer):
    def __init__(self, **kwargs):
        super().__init__(patch_size=4, embed_dims=[64,128,320,512], num_heads=[1,2,5,8],
            mlp_ratios=[4,4,4,4], qkv_bias=True, norm_layer=partial(nn.LayerNorm,eps=1e-6),
            depths=[3,8,27,3], sr_ratios=[8,4,2,1], drop_rate=0., drop_path_rate=0.1)

class mit_b5(RGBXTransformer):
    def __init__(self, **kwargs):
        super().__init__(patch_size=4, embed_dims=[64,128,320,512], num_heads=[1,2,5,8],
            mlp_ratios=[4,4,4,4], qkv_bias=True, norm_layer=partial(nn.LayerNorm,eps=1e-6),
            depths=[3,6,40,3], sr_ratios=[8,4,2,1], drop_rate=0., drop_path_rate=0.1)