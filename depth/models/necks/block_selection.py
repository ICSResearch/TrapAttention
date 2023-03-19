# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, xavier_init

from depth.ops import resize
from depth.models.builder import NECKS

import math
import torch
import torch.nn.functional as F


from mmcv.runner import BaseModule, auto_fp16


class BlocksChosen(nn.Module):
    def __init__(self):
        super(BlocksChosen, self).__init__()

    def forward(self, xs):
        x_max = xs[0]
        for x_b in xs[1:]:
            x_max = torch.maximum(x_max, x_b)

        return x_max

def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x

@NECKS.register_module()
class BlockSelectionNeck(BaseModule):
    def __init__(self,
                 in_channels=[384]*5,
                 out_channels=[48, 96, 192, 384, 768],
                 start=[3, 5, 7, 9, 11],
                 end=[5, 7, 9, 11, 12],
                 scales=[4, 2, 1, .5, .25]):
        super(BlockSelectionNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.start = start
        self.end = end
        self.num_outs = len(scales)

        self.blocks_selection = BlocksChosen()
        self.trans_proj = nn.ModuleList()
        for i in range(len(scales)):
            if scales[i] > 1:
                self.trans_proj.append(nn.Sequential(
                                 nn.ConvTranspose2d(in_channels[i], in_channels[i], kernel_size=scales[i], stride=scales[i],
                                                    groups=in_channels[i]),
                                 LayerNorm2d(in_channels[i]),
                                 nn.Conv2d(in_channels[i], out_channels[i], kernel_size=(1, 1), stride=(1, 1)),
                                 ))
            elif scales[i] == 1:
                if in_channels[i] == out_channels[i]:
                    self.trans_proj.append(nn.Identity())
                else:
                    self.trans_proj.append(nn.Sequential(
                                     LayerNorm2d(in_channels[i]),
                                     nn.Conv2d(in_channels[i], out_channels[i], kernel_size=(1, 1), stride=(1, 1))
                                     ))
            elif scales[i] < 1:
                self.trans_proj.append(nn.Sequential(
                                 nn.Conv2d(in_channels[i], in_channels[i], kernel_size=int(1/scales[i]), stride=int(1/scales[i])),
                                 LayerNorm2d(in_channels[i]),
                                 nn.Conv2d(in_channels[i], out_channels[i], kernel_size=(1, 1), stride=(1, 1)),
                                 ))




    # init weight
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # xavier_init(m, distribution='uniform')
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, LayerNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs):
        assert len(inputs) >= self.end[-1] - 1
        outs = []
        # for indices_start, indices_end in zip(self.start, self.end):
        for i in range(len(self.start)):
            feature = self.blocks_selection(inputs[self.start[i]:self.end[i]])
            outs.append(self.trans_proj[i](feature))

        return tuple(outs)

