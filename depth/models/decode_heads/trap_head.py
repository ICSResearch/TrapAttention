import mmcv
import math
import torch
import torch.nn as nn

import torch.nn.functional as torch_nn_func
from depth.models.builder import HEADS
from .decode_head import DepthBaseDecodeHead
from timm.models.layers.drop import DropPath

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
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


class BottleNeck(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, norm_layer=LayerNorm2d, drop=0.1,
            ls_init_value=1e-5, trap=True, drop_path=0.
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
        self.gamma2 = nn.Parameter(ls_init_value * torch.ones(out_features)) if ls_init_value > 0 else None

        self.trap = trap
        if self.trap:
            self.downsample = nn.PixelUnshuffle(2)
            self.attn_conv = nn.Conv2d(in_features*4, in_features, kernel_size=3, padding=1, groups=in_features)
            self.norm1 = norm_layer(in_features*4) if norm_layer else nn.Identity()
            self.gamma1 = nn.Parameter(ls_init_value * torch.ones(in_features)) if ls_init_value > 0 else None

        self.shortcut = out_features == in_features
        self.drop = nn.Identity()
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = self.drop_path(self.dw_conv(x)) + x
        if self.trap:
            shortcut1 = x
            x = trapped_inter(self.downsample(x))
            x = self.norm1(x)
            x = self.attn_conv(x)
            x = x.mul(self.gamma1.reshape(1, -1, 1, 1))
            x = self.drop_path(x) + shortcut1

        if self.shortcut:
            shortcut2 = x
        x = self.norm2(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.gamma2 is not None:
            x = x.mul(self.gamma2.reshape(1, -1, 1, 1))

        if self.shortcut:
            x = self.drop_path(x) + shortcut2

        return x


def trapped_inter(x):
    B, C, H, W = x.shape
    mask1 = torch.round(torch.abs(torch.sin(x)))
    mask2 = torch.round(torch.abs(2*torch.sin(x)*torch.cos(x)))
    mask3 = torch.round(torch.abs(torch.cos(x)))
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
    # x = torch.cat([x1, x3, x2, x4], dim=1)
    # x = torch.pixel_shuffle(torch.channel_shuffle(x, 4), 2)

    return x




@HEADS.register_module()
class TrappedHead(DepthBaseDecodeHead):
    def __init__(self, final_norm, post_process_channels=[64, 256, 512, 1024, 2048], drop_path_rate=0.05, **kwargs):
        super(TrappedHead, self).__init__(**kwargs)
        self.final_norm = final_norm
        self.channels = post_process_channels
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(post_process_channels)+1)]


        self.block1 = BottleNeck(self.channels[-1], self.channels[-1]*2, self.channels[-2], trap=False, drop_path=dpr[0])
        self.block2 = BottleNeck(self.channels[-2], self.channels[-2]*2, self.channels[-3], drop_path=dpr[1])
        self.block3 = BottleNeck(self.channels[-3], self.channels[-3]*2, self.channels[-4], drop_path=dpr[2])
        self.block4 = BottleNeck(self.channels[-4], self.channels[-4]*2, self.channels[-5], drop_path=dpr[3])
        self.block5 = BottleNeck(self.channels[-5], self.channels[-5]*2, self.channels[-5]//2, drop_path=dpr[4])
        self.block6 = BottleNeck(self.channels[-5]//2, self.channels[-5]*2, self.channels[-5]//2,
                                 drop_path=dpr[5]
                                 )

        self.fusion1 = nn.Sequential(
            LayerNorm2d(self.channels[-2]),
            nn.Conv2d(self.channels[-2], self.channels[-5] // 2, kernel_size=1),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        )

        self.fusion2 = nn.Sequential(
            LayerNorm2d(self.channels[-3]),
            nn.Conv2d(self.channels[-3], self.channels[-5] // 2, kernel_size=1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        )

        self.fusion3 = nn.Sequential(
            LayerNorm2d(self.channels[-4]),
            nn.Conv2d(self.channels[-4], self.channels[-5] // 2, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

        self.fusion4 = nn.Sequential(
            LayerNorm2d(self.channels[-5]),
            nn.Conv2d(self.channels[-5], self.channels[-5] // 2, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.act = nn.GELU()
        self.sig = nn.Sigmoid()

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

    def forward(self, features, img_metas):
        cam_params = [img_meta['cam_intrinsic'] for img_meta in img_metas]
        focal = [cam_param[0][0] for cam_param in cam_params]

        f2, f4, f8, f16, f32 = features

        f32 = self.block1(f32)
        f16 = f16 + trapped_inter(f32)
        f16 = self.block2(f16)
        f8 = f8 + trapped_inter(f16)
        f8 = self.block3(f8)
        f4 = f4 + trapped_inter(f8)
        f4 = self.block4(f4)
        f2 = f2 + trapped_inter(f4)
        f2 = self.block5(f2)

        fusion = self.fusion1(f32) + self.fusion2(f16) + self.fusion3(f8) + self.fusion4(f4) + f2

        fusion = self.block6(fusion)

        final_depth = self.depth_pred(fusion)

        final_depth = torch_nn_func.interpolate(final_depth, scale_factor=2, mode='bilinear', align_corners=False)

        if self.final_norm:
            final_depth = final_depth * focal.view(-1, 1, 1, 1).float() / 715.0873

        return final_depth
