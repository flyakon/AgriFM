import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from mmseg.models.builder import BACKBONES
from einops import reduce as einops_reduce
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from mmseg.registry.registry import MODELS, TRANSFORMS
from mmengine.model import BaseModule, BaseModel
from typing import Optional, Union, Dict, Iterator, Set, Tuple
from mmengine.runner import load_checkpoint


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
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
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_specific_module(checkpoint, module_name, replace_name,state_key=None):
    if isinstance(checkpoint, dict):
        state_dict = checkpoint
    elif isinstance(checkpoint, str):
        state_dict = torch.load(checkpoint)
    else:
        raise ValueError('checkpoint type error')
    if state_key is not None:
        state_dict = state_dict[state_key]
    # state_dict=torch.load(checkpoint)
    state_dict_s = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            state_dict_s[key.replace(module_name, replace_name)] = value
    return state_dict_s


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, downsample_step=(2, 2, 2),
                mean_frame_down=False):
        super().__init__()
        self.dim = dim
        # assert downsample_step[0]==2 or downsample_step[0]==1, "downsample_step[0] must be 1 or 2"

        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.downsample_step = downsample_step
        self.mean_frame_down = mean_frame_down

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        D_step, H_step, W_step = self.downsample_step
        if D_step>=2 and self.mean_frame_down and D%D_step!=0:
            x=F.pad(x,(0,0,0,0,0,0,0,D_step-D%D_step))
        if H_step==1:
            h=0
        else:
            h=1
        if W_step==1:
            w=0
        else:
            w=1
        x0 = x[:, ::D_step, 0::H_step, 0::W_step, :]  # B D H/2 W/2 C
        x1 = x[:, ::D_step, h::H_step, 0::W_step, :]  # B D H/2 W/2 C
        x2 = x[:, ::D_step, 0::H_step, w::W_step, :]  # B D H/2 W/2 C
        x3 = x[:, ::D_step, h::H_step, w::W_step, :]  # B D H/2 W/2 C
        if D_step>=2 and self.mean_frame_down:
            for i in range(1,D_step):
                x0 = x0 + x[:, i::D_step,  0::H_step, 0::W_step, :]
                x1 = x1 + x[:, i::D_step,  h::H_step, 0::W_step, :]
                x2 = x2 + x[:, i::D_step,  0::H_step, w::W_step, :]
                x3 = x3 + x[:, i::D_step,  h::H_step, w::W_step, :]
            x0 = x0 / D_step
            x1 = x1 / D_step
            x2 = x2 / D_step
            x3 = x3 / D_step
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 downsample_step=(2, 2, 2),
                 mean_frame_down=False,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.mean_frame_down = mean_frame_down

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, downsample_step=downsample_step,
                                         mean_frame_down=mean_frame_down)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


@MODELS.register_module()
class SpectralIsolatePatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(2, 4, 4), embed_dim=96,num_frames=16, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        # self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.proj = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.weight_proj= nn.Conv3d(embed_dim,1, kernel_size=3, stride=1, padding=1, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding

        x=x.permute(0,2,1,3,4)
        # x=x[:,:real_channels,:,:,:]
        _, _, D, H, W = x.size()
        assert D==self.num_frames, "D must be equal to num_frames: {}".format(D)
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        B, C, T, H, W = x.size()
        inputs_x=x
        x=rearrange(inputs_x, 'b c t h w -> (b c) 1 t h w')
        proj_x = self.proj(x)  # BC 128 T Wh Ww
        x=rearrange(proj_x, '(b c) d t h w -> b c d t h w', b=B,c=C)
        mean_x=x.mean(dim=1,keepdim=True)
        # weights_x=torch.unsqueeze(proj_x,dim=2)
        weights_x=x+mean_x
        weights_x=rearrange(weights_x, 'b c d t h w -> (b c) d t h w')
        weights = self.weight_proj(weights_x)  # BC 1 T Wh Ww
        weights=rearrange(weights, '(b c) 1 t h w -> b c 1 t h w', b=B,c=C)
        weights=torch.nn.functional.softmax(weights, dim=1)
        x = x * weights
        x = x.sum(dim=1, keepdim=False)  # B 128 T Wh Ww

        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class DynamicWeightGenerator(nn.Module):
    """
    根据输入数据动态生成卷积核权重的超网络
    输入: [B, C, D, H, W] 时序多光谱数据
    输出: 3D卷积核权重 [B, out_ch, in_ch, kt, kh, kw]
    """

    def __init__(self, embed_dim, patch_size, reduction=16,num_frames=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.kt, self.kh, self.kw = patch_size
        self.reduction = reduction
        self.num_frames=num_frames

        # 通道压缩层
        self.channel_compression = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool3d((1, 8, 8)))

        # 权重生成网络
        hidden_dim = embed_dim * self.kt * self.kh * self.kw // reduction
        self.weight_net = nn.Sequential(
            nn.Linear(8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * self.kt * self.kh * self.kw))

        # 偏置生成网络
        self.bias_net = nn.Sequential(
            nn.Linear(8 * 8, embed_dim // reduction),
            nn.ReLU(),
            nn.Linear(embed_dim // reduction, embed_dim))

    def forward(self, x):
        B, C,D, H, W = x.shape
        # x=x.permute(0, 2, 1, 3, 4)
        # assert D == self.num_frames, f"Expected {self.num_frames} frames, but got {D} frames."
        # 通道注意力权重
        # channel_weights = F.adaptive_avg_pool3d(x, (1, 1, 1))  # [B,C,1,1,1]
        # channel_weights = torch.sigmoid(channel_weights)

        # 对每个通道处理
        all_weights = []
        all_biases = []
        for c in range(C):
            # 提取单通道时空特征 [B,1,D,H,W]
            single_channel = x[:, c:c + 1, :, :, :]
            single_channel=torch.cat((single_channel, torch.mean(x,dim=1,keepdim=True)), dim=1)

            # 压缩空间维度 [B,1,D,8,8]
            compressed = self.channel_compression(single_channel)

            # 展平 [B,1,D*8*8] -> [B,D*8*8]
            flattened = compressed.view(B, -1)

            # 生成权重 [B, out*kt*kh*kw]
            weights = self.weight_net(flattened).view(
                B, self.embed_dim, 1, self.kt, self.kh, self.kw)

            # 生成偏置 [B, out]
            bias = self.bias_net(flattened)

            # 应用通道注意力
            # weights = weights * channel_weights[:, c:c + 1, :, :, :].unsqueeze(1)
            # bias = bias * channel_weights[:, c:c + 1, 0, 0, 0]

            all_weights.append(weights)
            all_biases.append(bias)

        # 合并所有通道的权重 [B, out, C, kt, kh, kw]
        final_weights = torch.cat(all_weights, dim=2)
        # 合并偏置 [B, out]
        final_bias = torch.sum(torch.stack(all_biases), dim=0)

        return final_weights, final_bias


@MODELS.register_module()
class FullyDynamicPatchEmbed(nn.Module):
    """
    完全动态的Patch Embedding层
    权重完全由输入数据决定
    """

    def __init__(self, embed_dim=96, patch_size=(2, 4, 4), num_frames=16,reduction=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = patch_size
        self.weight_generator = DynamicWeightGenerator(embed_dim, patch_size,num_frames=num_frames,reduction=reduction)
        self.num_frames=num_frames
        self.patch_size = patch_size
        # 可学习的位置编码
        # self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, D,C, H, W = x.shape
        x=x.permute(0, 2, 1, 3, 4)
        assert D == self.num_frames, f"Expected {self.num_frames} frames, but got {D} frames."

        # 动态生成权重 [B, out, in, kt, kh, kw]
        weights, bias = self.weight_generator(x)

        # 分组卷积处理(每个样本使用自己的权重)
        outputs = []
        for i in range(B):
            # [1, out, in, kt, kh, kw] -> [out, in, kt, kh, kw]
            sample_weight = weights[i].squeeze(0)
            # [1, out] -> [out]
            sample_bias = bias[i]

            # 3D卷积 [1, C, D, H, W] -> [1, E, D', H', W']
            out = F.conv3d(x[i:i + 1],
                           sample_weight,
                           sample_bias,
                           stride=self.kernel_size)
            outputs.append(out)

        # 合并结果 [B, E, D', H', W']
        x = torch.cat(outputs, dim=0)

        # # 展平时空维度并添加位置编码
        # x = x.flatten(2).transpose(1, 2)  # [B, L, E]
        # x = x + self.pos_embed

        return x


@MODELS.register_module()
class DynamicPatchEmbed(nn.Module):
    """
    完全动态的Patch Embedding层
    权重完全由输入数据决定
    """

    def __init__(self, embed_dim=96, patch_size=(2, 4, 4), num_frames=16,reduction=16,max_channels=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = patch_size
        self.weight_generator = DynamicWeightGenerator(embed_dim, patch_size,num_frames=num_frames,reduction=reduction)
        self.num_frames=num_frames
        self.patch_size = patch_size
        kt,kh,kw=patch_size
        self.weights_parameters= nn.Parameter(torch.rand(embed_dim,max_channels, kt,kh,kw), requires_grad=True)
        self.bias_parameters= nn.Parameter(torch.rand(embed_dim), requires_grad=True)
        self.max_channels=max_channels
        # 可学习的位置编码
        # self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, D,C, H, W = x.shape
        x=x.permute(0, 2, 1, 3, 4)
        assert D == self.num_frames, f"Expected {self.num_frames} frames, but got {D} frames."


        # 动态生成权重 [B, out, in, kt, kh, kw]
        weights, bias = self.weight_generator(x)

        # 分组卷积处理(每个样本使用自己的权重)
        outputs = []
        for i in range(B):
            # [1, out, in, kt, kh, kw] -> [out, in, kt, kh, kw]
            sample_weight = weights[i].squeeze(0)
            sample_weight=sample_weight+self.weights_parameters[:,:C,:,:,:]
            # [1, out] -> [out]
            sample_bias = bias[i]
            sample_bias=sample_bias+self.bias_parameters

            # 3D卷积 [1, C, D, H, W] -> [1, E, D', H', W']
            out = F.conv3d(x[i:i + 1],
                           self.weights_parameters[:,:C,:,:,:],
                           self.bias_parameters,
                           stride=self.kernel_size)
            outputs.append(out)

        # 合并结果 [B, E, D', H', W']
        x = torch.cat(outputs, dim=0)

        # # 展平时空维度并添加位置编码
        # x = x.flatten(2).transpose(1, 2)  # [B, L, E]
        # x = x + self.pos_embed

        return x


@MODELS.register_module()
class SwinPatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        x=x.permute(0,2,1,3,4)
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


@BACKBONES.register_module()
class SwinTransformer3D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(4, 4, 4),
                 # in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2, 7, 7),
                 mlp_ratio=4.,
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 downsample_steps=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                 frozen_stages=-1,
                 use_checkpoint=False, feature_fusion='cat',mean_frame_down=False,
                 reduce_feature_scale=None):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        # self.in_chans = in_chans
        self.mean_frame_down=mean_frame_down
        self.reduce_feature_scale = reduce_feature_scale

        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed3D(
        #     patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                downsample_step=downsample_steps[i_layer],
                mean_frame_down=mean_frame_down,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_fusion = feature_fusion
        # add a norm layer for each output
        self.norm = nn.LayerNorm(self.num_features)
        self.out_indices = out_indices
        self._freeze_stages()

    def _freeze_stages(self):
        # if self.frozen_stages >= 0:
        #     self.patch_embed.eval()
        #     for param in self.patch_embed.parameters():
        #         param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        # state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            if k not in self.state_dict():
                continue
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                # logger.warning(f"Error in loading {k}, passing")
                print(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        size=(2 * self.window_size[1] - 1, 2 * self.window_size[2] - 1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2,
                                                                                                                   L2).permute(
                        1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2 * wd - 1, 1)

        # if self.in_chans!=3:
        #     state_dict.pop('patch_embed.proj.weight')
        for key in list(state_dict.keys()):
            if key.startswith('patch_embed'):
                state_dict.pop(key)
        msg = self.load_state_dict(state_dict, strict=False)
        print(msg)
        # logger.info(msg)
        # logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            # logger.info(f'load model from: {self.pretrained}')
            print(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights()
            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained, strict=False)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        # x = self.patch_embed(x)
        if 0 in self.out_indices:

            if self.feature_fusion == 'cat':
                # feat=reduce(x,'b c d h w')
                # feat= torch.reshape(x, (B, C * D, H, W))
                feat = einops_reduce(x, 'b c (t s) h w -> b c t h w', 'mean', s=self.reduce_feature_scale) if self.reduce_feature_scale is not None else x
                B, C, D, H, W = feat.shape
                out_feats = [torch.reshape(feat, (B, C * D, H, W))]
            elif self.feature_fusion == 'mean':
                out_feats = [torch.mean(x, dim=2)]
            else:
                out_feats = [x]
            # out_feats = [torch.reshape(x, (B, C*D,H,W))]
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            x = layer(x.contiguous())
            if i + 1 in self.out_indices:
                B, C, D, H, W = x.shape
                if self.feature_fusion == 'cat':
                    if self.reduce_feature_scale is not None and D% self.reduce_feature_scale !=0:
                        feat=torch.nn.functional.pad(x,(0,0,0,0,0,self.reduce_feature_scale-D% self.reduce_feature_scale))
                    else:
                        feat=x
                    feat = einops_reduce(feat, 'b c (t s) h w -> b c t h w', 'mean', s=self.reduce_feature_scale) if self.reduce_feature_scale is not None else x
                    B, C, D, H, W = feat.shape
                    out_feats.append(torch.reshape(feat, (B, C * D, H, W)))
                elif self.feature_fusion == 'mean':
                    out_feats.append(torch.mean(x, dim=2))
                else:
                    out_feats.append(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')
        B, C, D, H, W = x.shape
        if self.feature_fusion == 'cat':
            feat = einops_reduce(x, 'b c (t s) h w -> b c t h w', 'mean', s=self.reduce_feature_scale) if self.reduce_feature_scale is not None else x
            B, C, D, H, W = feat.shape
            x = torch.reshape(feat, (B, C * D, H, W))
        elif self.feature_fusion == 'mean':
            x = torch.mean(x, dim=2)
        return {'encoder_features': x, 'features_list': out_feats}

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3D, self).train(mode)
        self._freeze_stages()

@MODELS.register_module()
class PretrainingSwinTransformer3DEncoder(BaseModel):
    def __init__(self, patch_emd_cfg,
                 backbone_cfg,
                 init_cfg=None, data_preprocessor=None,**kwargs):
        BaseModel.__init__(self, data_preprocessor, None)
        self.patch_emd = MODELS.build(patch_emd_cfg)
        self.backbone = MODELS.build(backbone_cfg)

        if init_cfg is not None:
            checkpoint = init_cfg.get('checkpoint', None)
            if checkpoint is None:
                print('checkpoint is None')
            else:
                revise_keys = init_cfg.get('revise_keys', [(r'^module\.', '')])
                strict=init_cfg.get('strict', False)
                if self.patch_emd.in_chans==6:
                    revise_keys.append(('HLSL30_patch_emd','patch_emd'))
                elif self.patch_emd.in_chans==7:
                    revise_keys.append(('Modis_patch_emd','patch_emd'))
                elif self.patch_emd.in_chans==10:
                    revise_keys.append(('S2_patch_emd','patch_emd'))
                load_checkpoint(self, checkpoint, strict=strict, revise_keys=revise_keys)
                print('load checkpoint success from:', checkpoint)
                self.is_init = True


    def forward(self, inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor'):
        shape=inputs.shape
        if len(shape)==4:
            inputs=inputs.unsqueeze(1)
        x = self.patch_emd(inputs)
        features_dict = self.backbone(x)
        return features_dict


