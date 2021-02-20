import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (
    ConvModule,
    build_activation_layer,
    build_conv_layer,
    build_norm_layer,
    constant_init,
    normal_init,
)
from mmcv.runner import load_checkpoint
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


def conv2d_sample_by_sample(
    x: torch.Tensor,
    weight: torch.Tensor,
    oup: int,
    inp: int,
    ksize: int,
    stride: int,
    padding: int,
    groups: int,
) -> torch.Tensor:
    batch_size = x.shape[0]
    if batch_size == 1:
        out = F.conv2d(
            x,
            weight=weight.view(oup, inp, ksize, ksize),
            stride=stride,
            padding=padding,
            groups=groups,
        )
    else:
        out = F.conv2d(
            x.view(1, -1, x.shape[2], x.shape[3]),
            weight.view(batch_size * oup, inp, ksize, ksize),
            stride=stride,
            padding=padding,
            groups=groups * batch_size,
        )
        out = out.view(batch_size, oup, out.shape[2], out.shape[3])
    return out


class WeightNetConv(nn.Module):
    r"""Applies WeightNet to a standard convolution.
    The grouped fc layer directly generates the convolutional kernel,
    this layer has M*inp inputs, G*oup groups and oup*inp*ksize*ksize outputs.
    M/G control the amount of parameters.
    """

    def __init__(self, inp, oup, ksize, stride, M=2, G=2):
        super().__init__()
        inp_gap = max(16, inp // 16)
        self.inp = inp
        self.oup = oup
        self.ksize = ksize
        self.stride = stride
        self.padding = ksize // 2

        self.wn_fc1 = nn.Conv2d(inp_gap, M * oup, 1, 1, 0, groups=1, bias=True)
        self.wn_fc2 = nn.Conv2d(
            M * oup, oup * inp * ksize * ksize, 1, 1, 0, groups=G * oup, bias=False
        )

    def forward(self, x, x_gap):
        x_w = self.wn_fc1(x_gap)
        x_w = torch.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)
        return conv2d_sample_by_sample(
            x, x_w, self.oup, self.inp, self.ksize, self.stride, self.padding, 1
        )


class WeightNetConvDW(nn.Module):
    r"""Here we show a grouping manner when we apply WeightNet to a depthwise convolution.
    The grouped fc layer directly generates the convolutional kernel, has fewer parameters while achieving comparable results.
    This layer has M/G*inp inputs, inp groups and inp*ksize*ksize outputs.
    """

    def __init__(self, inp, ksize, stride, M=2, G=2):
        super().__init__()
        inp_gap = max(16, inp // 16)
        self.inp = inp
        self.ksize = ksize
        self.stride = stride
        self.padding = ksize // 2

        self.wn_fc1 = nn.Conv2d(inp_gap, M // G * inp, 1, 1, 0, groups=1, bias=True)
        self.wn_fc2 = nn.Conv2d(M // G * inp, inp * ksize * ksize, 1, 1, 0, groups=inp, bias=False)

    def forward(self, x, x_gap):
        x_w = self.wn_fc1(x_gap)
        x_w = torch.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)
        return conv2d_sample_by_sample(
            x, x_w, self.inp, 1, self.ksize, self.stride, self.padding, self.inp
        )


# https://github.com/megvii-model/WeightNet/blob/master/shufflenet_v2.py
class InvertedResidual(nn.Module):
    """InvertedResidual block for WeightNet (adapted from ShuffleNetV2) backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 convolution layer. Default: 1
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU"),
        with_cp=False,
    ):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.with_cp = with_cp
        self.reduce = build_conv_layer(
            conv_cfg,
            in_channels,
            max(16, in_channels // 16),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        branch_features = out_channels // 2

        if self.stride > 1:
            self.wnet_proj_1 = WeightNetConvDW(in_channels, 3, self.stride)
            _, self.norm_proj_1 = build_norm_layer(norm_cfg, in_channels)
            self.wnet_proj_2 = WeightNetConv(in_channels, in_channels, 1, 1)
            _, self.norm_proj_2 = build_norm_layer(norm_cfg, in_channels)
            self.relu_proj_2 = build_activation_layer(act_cfg)

        self.wnet1 = WeightNetConv(in_channels, branch_features, 1, 1)
        _, self.norm1 = build_norm_layer(norm_cfg, branch_features)
        self.relu1 = build_activation_layer(act_cfg)

        self.wnet2 = WeightNetConvDW(branch_features, 3, self.stride)
        _, self.norm2 = build_norm_layer(norm_cfg, branch_features)

        self.wnet3 = WeightNetConv(branch_features, out_channels - in_channels, 1, 1)
        _, self.norm3 = build_norm_layer(norm_cfg, out_channels - in_channels)
        self.relu3 = build_activation_layer(act_cfg)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % 4 == 0
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

    def forward(self, x):
        def _inner_forward(old_x):
            if self.stride == 1:
                x_proj, x = self.channel_shuffle(old_x)
            elif self.stride == 2:
                x_proj, x = old_x, old_x
            x_gap = self.reduce(x.mean(dim=[2, 3], keepdim=True))

            x = self.wnet1(x, x_gap)
            x = self.norm1(x)
            x = self.relu1(x)
            x = self.wnet2(x, x_gap)
            x = self.norm2(x)
            x = self.wnet3(x, x_gap)
            x = self.norm3(x)
            x = self.relu3(x)

            if self.stride == 2:
                x_proj = self.wnet_proj_1(x_proj, x_gap)
                x_proj = self.norm_proj_1(x_proj)
                x_proj = self.wnet_proj_2(x_proj, x_gap)
                x_proj = self.norm_proj_2(x_proj)
                x_proj = self.relu_proj_2(x_proj)
            return torch.cat((x_proj, x), 1)

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


@BACKBONES.register_module()
class WeightNet(BaseBackbone):
    """WeightNet (adapted from ShuffleNetV2) backbone.

    Args:
        widen_factor (float): Width multiplier - adjusts the number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(
        self,
        widen_factor=1.0,
        out_indices=(3,),
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU"),
        norm_eval=False,
        with_cp=False,
    ):
        super(WeightNet, self).__init__()
        self.stage_blocks = [4, 8, 4]
        for index in out_indices:
            if index not in range(0, 4):
                raise ValueError(
                    "the item in out_indices must in " f"range(0, 4). But received {index}"
                )

        if frozen_stages not in range(-1, 4):
            raise ValueError(
                "frozen_stages must be in range(-1, 4). " f"But received {frozen_stages}"
            )
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        if widen_factor == 0.5:
            channels = [48, 96, 192, 1024]
        elif widen_factor == 1.0:
            channels = [116, 232, 464, 1024]
        elif widen_factor == 1.5:
            channels = [176, 352, 704, 1024]
        elif widen_factor == 2.0:
            channels = [244, 488, 976, 2048]
        else:
            raise ValueError(
                "widen_factor must be in [0.5, 1.0, 1.5, 2.0]. " f"But received {widen_factor}"
            )

        self.in_channels = 24
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            layer = self._make_layer(channels[i], num_blocks)
            self.layers.append(layer)

        output_channels = channels[-1]
        self.layers.append(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=output_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        )

    def _make_layer(self, out_channels, num_blocks):
        """Stack blocks to make a layer.

        Args:
            out_channels (int): out_channels of the block.
            num_blocks (int): number of blocks.
        """
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(
                    InvertedResidual(
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        stride=2,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        with_cp=self.with_cp,
                    )
                )
                self.in_channels = out_channels
            else:
                layers.append(
                    InvertedResidual(
                        in_channels=self.in_channels // 2,
                        out_channels=out_channels,
                        stride=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        with_cp=self.with_cp,
                    )
                )

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    if "conv1" in name:
                        normal_init(m, mean=0, std=0.01)
                    else:
                        normal_init(m, mean=0, std=1.0 / m.weight.shape[1])
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m.weight, val=1, bias=0.0001)
                    if isinstance(m, _BatchNorm):
                        if m.running_mean is not None:
                            nn.init.constant_(m.running_mean, 0)
        else:
            raise TypeError("pretrained must be a str or None. But received " f"{type(pretrained)}")

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(WeightNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
