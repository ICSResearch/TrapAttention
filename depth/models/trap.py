# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of siam.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>
import timm.models.resnest

# from timm2.models.convnext import LayerNorm2d
# from encoder.xcit import *
from timm.models.layers.drop import DropBlock2d
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as torch_nn_func
import math
from timm.models.layers.patch_embed import PatchEmbed
from timm.models.layers.mlp import ConvMlp, Mlp
from timm.utils.model_ema import ModelEmaV2
import pathlib
from mmseg.ops import resize
from mmcv.cnn import ConvModule
from einops import rearrange
from timm.models.layers import DropPath
from .backbones.xcit import *

from collections import namedtuple

# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return torch_nn_func.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, dropout=0.1):
    """3x3 convolution + batch norm"""
    return torch.nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), dilation=dilation, stride=(stride, stride), groups=groups, padding=dilation, bias=False),
        nn.BatchNorm2d(out_planes, momentum=0.0025),
        # nn.InstanceNorm2d(out_planes),
        # nn.GroupNorm(32, out_planes),
        # nn.Dropout2d(dropout)
    )


def conv1x1(in_planes, out_planes, stride=1, groups=1, dropout=0.1):
    """3x3 convolution + batch norm"""
    return torch.nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(stride, stride), groups=groups, padding=0, bias=False),
        nn.BatchNorm2d(out_planes, momentum=0.0025),
        # nn.InstanceNorm2d(out_planes),
        # nn.GroupNorm(32, out_planes),
        # nn.Dropout2d(dropout)
    )



class LayerNorm2d(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return torch_nn_func.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        if m.kernel_size[0] > 1:
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # torch.nn.init.trunc_normal_(m.weight, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        else:
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        torch.nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, LayerNorm2d):
        m.reset_parameters()
    elif isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        # d = torch.log(depth_est[mask]+1e-6) - torch.log(depth_gt[mask]+1e-6)
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class h_loss(nn.Module):
    def __init__(self, variance_focus):
        super().__init__()
        self.variance_focus = variance_focus

    def forward(self, t, s, mask):
        t = t.detch()
        d = torch.log(t[mask == False]) - torch.log(s[mask == False])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


class BottleNeck(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=LayerNorm2d, drop=0.,
            ls_init_value=1., trap=True, drop_path=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dw_conv = nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm1 = norm_layer(in_features) if norm_layer else nn.Identity()
        self.norm2 = norm_layer(in_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)
        self.drop_path = nn.Dropout(drop_path)
        # self.gamma2 = nn.Parameter(ls_init_value * torch.ones(in_features)) if ls_init_value > 0 else None
        # self.gamma3 = nn.Parameter(ls_init_value * torch.ones(out_features)) if ls_init_value > 0 else None
        self.gamma2 = nn.Parameter(ls_init_value * torch.ones(out_features)) if ls_init_value > 0 else None

        self.trap = trap
        if self.trap:
            # self.attn_conv = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
            # self.attn_conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=2, padding=1, groups=in_features)
            self.downsample = nn.PixelUnshuffle(2)
            self.attn_conv = nn.Conv2d(in_features*4, in_features, kernel_size=3, padding=1, groups=in_features)
            # self.attn_conv = nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)
            # self.pool = nn.AvgPool2d(2, 2)
            self.norm1 = norm_layer(in_features*4) if norm_layer else nn.Identity()
            self.gamma1 = nn.Parameter(ls_init_value * torch.ones(in_features)) if ls_init_value > 0 else None

    def forward(self, x):
        x = self.dw_conv(x) + x
        if self.trap:
            shortcut1 = x
            x = trapped_inter(self.downsample(x))
            x = self.norm1(x)
            x = self.attn_conv(x)
            x = x.mul(self.gamma1.reshape(1, -1, 1, 1))
            x = self.drop_path(x) + shortcut1

        shortcut2 = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        if self.gamma2 is not None:
            x = x.mul(self.gamma2.reshape(1, -1, 1, 1))

        return self.drop_path(x) + shortcut2


def trapped_inter(x):
    B, C, H, W = x.shape
    mask1 = torch.round(torch.abs(torch.sin(x)))
    mask2 = torch.round(torch.abs(torch.cos(x)))
    mask3 = torch.round(torch.abs(2*torch.sin(x)*torch.cos(x)))
    mask4 = torch.round(torch.sin(x)**2)

    x1 = mask1 * x
    x2 = mask2 * x
    x3 = mask3 * x
    x4 = mask4 * x
    x = torch.cat([x1, x3, x2, x4], dim=1)
    x = x.view(B, 2, 2*C, H, W)
    x = x.permute(0, 2, 3, 1, 4).flatten(2).contiguous()
    x = x.view(B, 2*C, H * 2, W)
    x = x.view(B, 2, C, H * 2, W)
    x = x.permute(0, 2, 3, 4, 1).flatten(-1).contiguous()
    x = x.view(B, C, H * 2, W * 2)
    return x


class CatBlocks(nn.Module):
    def __init__(self, scale=2, align_corners=False, squeeze_dims=None):
        super(CatBlocks, self).__init__()
        self.squeeze = nn.Identity()
        if squeeze_dims:
            self.squeeze = conv1x1(squeeze_dims[0], squeeze_dims[1])

    def forward(self, x):
        B, C, H, W = x.shape
        # x = x.view(B, 2, 2 * C, H, W)
        x = x.view(B, 2, C//2, H, W)
        x = x.permute(0, 2, 3, 1, 4).flatten(2).contiguous()
        # x = x.view(B, 2 * C, H * 2, W)
        x = x.view(B, C//2, H * 2, W)

        x = x.view(B, 2, C//4, H * 2, W)
        x = x.permute(0, 2, 3, 4, 1).flatten(-1).contiguous()
        x = x.view(B, C//4, H * 2, W * 2)
        x = self.squeeze(x)

        return x


class BlocksChosen(nn.Module):
    def __init__(self, chunks=4, align_corners=False, in_dim=None, hide_dim=None, out_dim=None):
        super(BlocksChosen, self).__init__()
        self.chunks = chunks


    def forward(self, xs):
        x_max = xs[0]
        for x_b in xs[1:]:
            x_max = torch.maximum(x_max, x_b)

        return x_max





class PPM(nn.ModuleList):
    def __init__(self, in_channels, channels, pool_scales=(1, 2, 3, 6),
                 act=nn.GELU, align_corners=False):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels

        for pool_scale in pool_scales:
        #     if pool_scale==1:
        #         self.append(
        #             nn.Sequential(
        #                 nn.AdaptiveAvgPool2d(pool_scale),
        #                 LayerNorm2d(in_channels),
        #                 nn.Conv2d(in_channels, channels, kernel_size=(1, 1)),
        #                 # nn.GroupNorm(1, channels),
        #                 # act()
        #             ))
        #     else:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    LayerNorm2d(in_channels),
                    nn.Conv2d(in_channels, channels, kernel_size=(1, 1)),
                    # act()
                   ))

    def forward(self, x):
        ppm_outs = [x]
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        ppm_outs = torch.cat(ppm_outs, dim=1)
        return ppm_outs


class ASPPModule(nn.ModuleList):
    def __init__(self, dilations, in_channels, channels, act=nn.GELU):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.append(nn.Sequential(
                conv1x1(in_channels, channels),
                act()))
        for dilation in dilations:
            self.append(
                nn.Sequential(
                conv3x3(in_channels, channels, dilation=dilation),
                act()))

    def forward(self, x):
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        aspp_outs = torch.cat(aspp_outs, dim=1)
        return aspp_outs


class siam(nn.Module):
    def __init__(self, params, size=(480, 640)):
        super(siam, self).__init__()
        self.params = params
        dim = 512
        self.h = size[0] // 8
        # self.w = size[1] // 8
        block_dim = 384
        # decode_dim = block_dim
        decode_dim = 288
        # self.trap1 = TrappedAttention(block_dim,  decode_dim, decode_dim*8, block_dim * 4,
        #                               inter_stride=2, shrink_stride=1)
        # self.trap2 = TrappedAttention(block_dim,  decode_dim, decode_dim*8, block_dim*4)
        # self.trap3 = TrappedAttention(block_dim,  decode_dim, decode_dim*4, block_dim)
        # self.trap4 = TrappedAttention(block_dim//4,  block_dim, decode_dim*4)
        self.block1 = BottleNeck(block_dim, block_dim*4, trap=False)
        self.block2 = BottleNeck(block_dim, block_dim*4)
        self.block3 = BottleNeck(block_dim, block_dim*4)
        self.block4 = BottleNeck(block_dim, block_dim*4)
        self.block5 = BottleNeck(block_dim, block_dim*4)

        self.up1 = nn.Sequential(
                                 # nn.PixelShuffle(2),
                                 # nn.ConvTranspose2d(block_dim, block_dim, groups=block_dim, kernel_size=(2, 2), stride=(2, 2)),
                                 LayerNorm2d(block_dim),
                                 # nn.Conv2d(block_dim, decode_dim//4, kernel_size=(1, 1), stride=(1, 1)),
                                 nn.ConvTranspose2d(block_dim, block_dim, kernel_size=(2, 2), stride=(2, 2)),
                                 )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            # nn.Conv2d(block_dim, block_dim, kernel_size=(2, 2), stride=(2, 2), groups=block_dim),
            # LayerNorm2d(block_dim),
            # nn.Conv2d(block_dim, decode_dim*4, kernel_size=(2, 2), stride=(2, 2)),
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(4, 4),
            # nn.Conv2d(block_dim, block_dim, kernel_size=(4, 4), stride=(4, 4), groups=block_dim),
            # LayerNorm2d(block_dim),
            # nn.Conv2d(block_dim, decode_dim * 4, kernel_size=(2, 2), stride=(2, 2)),
            # nn.GELU(),
            # nn.Conv2d(decode_dim * 4, decode_dim, kernel_size=(2, 2), stride=(2, 2)),
        )

        # self.get_depth = torch.nn.Sequential(BottleNeck(block_dim, block_dim*4, 1),
        #                                      nn.Sigmoid())
        self.act = nn.GELU()
        self.sig = nn.Sigmoid()
        self.get_depth = torch.nn.Sequential(
                                                # nn.GELU(),
                                                # nn.Conv2d(decode_dim // 8, decode_dim // 8, kernel_size=7, stride=1,
                                                #           groups=decode_dim // 8, padding=3),
                                                LayerNorm2d(block_dim),
                                                # nn.Conv2d(decode_dim//8, 1, 3, 1, 1),
                                                nn.Conv2d(block_dim, 1, 3, 1, 1),
                                                # nn.Conv2d(block_dim // 4, 1, 3, 1, 1, bias=False),
                                                nn.Sigmoid())

        # 43000-2 0.878,0.983,0.996,0.119,0.064,0.385,0.141,10.799,0.049
        # self.blocks_start = [3, 6, 8, 10]
        # self.blocks_end = [6, 8, 10, 13]

        # 60000,0.872,0.982,0.996,0.126,0.071,0.397,0.146,10.704,0.051
        # self.blocks_start = [6, 8, 9, 11]
        # self.blocks_end = [8, 10, 11, 13]

        self.blocks_start = [3, 5, 7, 11]
        self.blocks_end = [5, 7, 9, 13]

        self.blocks_chosen1 = BlocksChosen(self.blocks_end[0]-self.blocks_start[0], in_dim=block_dim, hide_dim=block_dim//4,
                                           out_dim=block_dim)
        self.blocks_chosen2 = BlocksChosen(self.blocks_end[1]-self.blocks_start[1], in_dim=block_dim, hide_dim=block_dim//4,
                                           out_dim=block_dim)
        self.blocks_chosen3 = BlocksChosen(self.blocks_end[2]-self.blocks_start[2], in_dim=block_dim, hide_dim=block_dim//4,
                                           out_dim=block_dim)
        self.blocks_chosen4 = BlocksChosen(self.blocks_end[3]-self.blocks_start[3], in_dim=block_dim, hide_dim=block_dim//4,
                                           out_dim=block_dim)

        # self.blocks_select = BlocksSelection()
        # self.cat_blocks4x = CatBlocks(4)
        # self.cat_blocks = nn.PixelShuffle(2)
        # self.avg = nn.AvgPool2d(2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.norm = nn.InstanceNorm2d(block_dim)

        self.apply(self.init_weight)
        self.use_checkpoint = True

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.kaiming_normal_(m.weight)
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()

    def forward(self, features, focal):
        f4, f8, f16, f32 = features[self.blocks_start[0]:self.blocks_end[0]], features[self.blocks_start[1]:self.blocks_end[1]], \
                           features[self.blocks_start[2]:self.blocks_end[2]], features[self.blocks_start[3]:self.blocks_end[3]]

        f4 = self.blocks_chosen1(f4)
        f8 = self.blocks_chosen2(f8)
        f16 = self.blocks_chosen3(f16)
        f32 = self.blocks_chosen4(f32)

        f4 = rearrange(f4, 'B (H W) C -> B C H W', H=self.h)
        f8 = rearrange(f8, 'B (H W) C -> B C H W', H=self.h)
        f16 = rearrange(f16, 'B (H W) C -> B C H W', H=self.h)
        f32 = rearrange(f32, 'B (H W) C -> B C H W', H=self.h)

        if self.use_checkpoint:
            f4 = checkpoint(self.up1, f4)
            f16 = self.down1(f16)
            f32 = self.down2(f32)

            f32 = checkpoint(self.block1, f32)
            f16 = f16 + trapped_inter(f32)
            # f16 = f16 + self.upsample(f32)
            f16 = checkpoint(self.block2, f16)
            f8 = f8 + trapped_inter(f16)
            # f8 = f8 + self.upsample(f16)
            f8 = self.block3(f8)
            f4 = f4 + trapped_inter(f8)
            # f4 = f4 + self.upsample(f8)
            f4 = checkpoint(self.block4, f4)

            fusion = torch_nn_func.interpolate(f32, scale_factor=8, mode='bilinear', align_corners=False) + \
                     torch_nn_func.interpolate(f16, scale_factor=4, mode='bilinear', align_corners=False) + \
                     torch_nn_func.interpolate(f8, scale_factor=2, mode='bilinear', align_corners=False) + \
                     f4

            # fusion = checkpoint(self.block5, fusion)
            fusion = self.block5(fusion)
            final_depth = self.params.max_depth * checkpoint(self.get_depth, fusion)
        else:
            f4 = self.up1(f4)
            # f8 = self.chan_seq(f8)
            f16 = self.down1(f16)
            f32 = self.down2(f32)
            # f4 = self.up1(f4)

            f32 = self.block1(f32)
            f16 = f16 + trapped_inter(f32)
            # f16 = f16 + self.upsample(f32)
            f16 = self.block2(f16)
            f8 = f8 + trapped_inter(f16)
            # f8 = f8 + self.upsample(f16)
            f8 = self.block3(f8)
            f4 = f4 + trapped_inter(f8)
            # f4 = f4 + self.upsample(f8)
            f4 = self.block4(f4)

            fusion = torch_nn_func.interpolate(f32, scale_factor=8, mode='bilinear', align_corners=False) + \
                     torch_nn_func.interpolate(f16, scale_factor=4, mode='bilinear', align_corners=False) + \
                     torch_nn_func.interpolate(f8, scale_factor=2, mode='bilinear', align_corners=False) + \
                     f4

            fusion = self.block5(fusion)

            final_depth = self.params.max_depth * self.get_depth(fusion)
        # final_depth = self.ps4(final_depth)
        final_depth = torch_nn_func.interpolate(final_depth, scale_factor=4, mode='bilinear', align_corners=False)
        # final_depth = self.params.max_depth * self.sig(output)

        if self.params.dataset == 'kitti':
            final_depth = final_depth * focal.view(-1, 1, 1, 1).float() / 715.0873

        return final_depth



class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        # from timm2.models.xcit import xcit_small_12_p8_224, xcit_small_12_p8_384_dist
        # from encoder.xcit import xcit_small_12_p8_224, xcit_small_12_p8_384_dist

        # self.base_model = xcit_small_12_p8_224(pretrained=True)
        # self.base_model = xcit_small_12_p8_384_dist(pretrained=True, drop_path_rate=0.05)
        self.base_model = xcit_small_12_p8_384_dist(pretrained=True)
        # self.base_model = xcit_small_12_p8_384_dist(pretrained=True, use_checkpoint=True)
        # self.base_model = xcit_small_24_p8_384_dist(pretrained=True)
        # self.base_model = xcit_small_24_p8_384_dist(pretrained=True, drop_path_rate=0.15)
        # self.base_model = xcit_small_24_p8_384_dist()
        # self.base_model.load_state_dict(torch.load(r'E:\PyProject\DE\bts-master\pytorch\encoder\xcit.pth'))
        # self.base_model = nn.ModuleList()
        # for n, m in xcit.named_children():
        del self.base_model.cls_attn_blocks
        del self.base_model.cls_token
        del self.base_model.norm
        del self.base_model.head
        # for n, m in self.base_model.named_children():
        #     print(f'11{n}')
        #     if 'cls_attn_blocks' in n:
        #         del m
        #     if 'embed' in n or 'blocks' in n:
            #     self.base_model.add_module(n, m)

        # del xcit


        # self.base_model = xcit_medium_24_p16_384_dist(pretrained=True)
        # print(self.base_model)
        # self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
        # self.feat_out_channels = [64, 64, 128, 256, 1024]
        # self.block_guide = nn.Parameter(torch.normal(0, 0.02, [1, 1, 384]))

    def forward(self, x):
        # feature = x
        feats = []
        Hp, Wp = 0, 0
        # block_guide = self.block_guide.repeat(x.shape[0], 1, 1)
        for k, m in self.base_model._modules.items():
            if 'blocks' == k:
                # print(block_guide.shape)
                # x = torch.cat([x, block_guide], dim=1)
                for k2, m2 in m._modules.items():
                    x = m2(x, Hp, Wp)
                    # block_guide = m2(block_guide, 2, 2)
                    # if k2 in [0, 3, 4, 5, 6, 7, 12, 11]:
                    # if int(k2) > 7:
                    # feats.append(x.unsqueeze(0))
                    feats.append(x)
                    # print(rearrange(x, 'B (Hp Wp) C -> B C Hp Wp', Hp=Hp).shape)
            elif 'patch_embed' == k:
                x, (Hp, Wp) = m(x)
                # Hp, Wp = Hp+1, Wp+1
                # block_guide, _ = m(block_guide)
                # x = torch.cat([x, block_guide], dim=1)
            elif 'pos_embed' == k:
                pos_encoding = m(x.shape[0], Hp, Wp).reshape(x.shape[0], -1, x.shape[1]).permute(0, 2, 1)
                x = x + pos_encoding
                # block_guide = block_guide + m(1, 1, 1).reshape(1, -1, 3).permute(0, 2, 1)
            # elif 'cls_attn' in k:
            #     block_guide = self.base_model.cls_token.expand(x.shape[0], -1, -1)
                # x = torch.cat((block_guide, x), dim=1)
                # for _, m2 in m._modules.items():
                #     x = m2(x)
        # feats = torch.cat(feats, dim=-1)
        # print(feats.shape)
        # feats = rearrange(feats, 'B (Hp Wp) C -> B C Hp Wp', Hp=Hp)
        # feats = rearrange(feats, 'B N (H W) -> B C H W', Hp=64)
        # print(feats.shape)
        # print(x.shape)
        return feats, x[:, 0:1, :]
        # return feats




class SiamModel(nn.Module):
    def __init__(self, params):
        super(SiamModel, self).__init__()
        self.encoder = encoder()
        # self.decoder_s = siam(params, self.encoder.feat_out_channels, params.siam_size)
        self.decoder = siam(params)
        # self.decoder_t = ModelEmaV2(self.decoder_s)

    def forward(self, x, focal, only_s=True):
        feats, block_guide = self.encoder(x)
        # feats = self.encoder(x)
        # return self.decoder(feats, focal, block_guide)
        return self.decoder(feats, focal)


if __name__ == '__main__':
    import argparse
    import torch
    import torch.nn as nn

    import torch.distributed as dist

    # from siam_dataloader import *


    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    parser = argparse.ArgumentParser(description='siam PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--mode', type=str, help='train or test', default='train')
    parser.add_argument('--model_name', type=str, help='model name', default='siam_eigen_v2')
    parser.add_argument('--siam_size', type=int, help='initial num_filters in siam', default=512)
    parser.add_argument('--encoder', type=str, default='resnet101')
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--dataset', type=float, help='maximum depth in estimation', default=10)

    args = parser.parse_args()
    # model = SiamModel(args)
    model = SiamModel(args)
    # in_t = torch.zeros([1, 3, 480, 640])
    in_t = torch.normal(mean=0, std=1, size=[1, 3, 480, 640])
    # print(model(in_t, in_t))
    # out = model(in_t, in_t)
    for n, m in model.named_parameters():
        if 'encoder' in n:
            print(n)
    # print(len(model.parameters()))
    import numpy as np
    print(sum([np.prod(p.size()) for p in model.parameters()]))
    out = model(in_t, in_t, only_s=False)
    print(out.shape)
    # _, out = model(in_t, in_t, only_s=False)

    print(torch.max(out))
    print(torch.min(out))


