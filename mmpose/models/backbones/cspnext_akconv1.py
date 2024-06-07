# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
from typing import Optional, Sequence, Tuple

import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from mmengine.model import constant_init, kaiming_init, normal_init
from .base_backbone import BaseBackbone
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.registry import MODELS
from mmpose.utils.typing import ConfigType
from ..utils.csp_layer import CSPLayer
from ..utils.csp_layer1 import CSPLayer1
from .csp_darknet import SPPBottleneck
from ..cnn import AKConv
from ..cnn.dcnv4 import yolo_Conv,DCNV4_Conv
from ..cnn.coordconv import CoordConv


########################### SPPELAN ##############################################
class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)
 
    def forward(self, x):
        return self.m(x)
     
class SPPELAN(nn.Module):
    # spp-elan
    def __init__(self, c1, c2, c3):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3
        self.cv1 = yolo_Conv(c1, c3, 1, 1)
        self.cv2 = SP(5)
        self.cv3 = SP(5)
        self.cv4 = SP(5)
        self.cv5 = yolo_Conv(4 * c3, c2, 1, 1)
 
    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))
########################### SPPELAN END ##############################################


########################### Bottleneck##############################################
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = yolo_Conv(c1, c_, k[0], 1)
        self.cv2 = yolo_Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
########################### Bottleneck END##############################################

########################### SPPELAN DCNv4 #################################################
class Bottleneck_DCNV4(Bottleneck):
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNV4_Conv(c_, c2, k[1])

class SPPELAN_DCNV4(nn.Module):
    # spp-elan
    def __init__(self, c1, c2, c3):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3
        self.cv1 = yolo_Conv(c1, c3, 1, 1)
        self.cv2 = SP(5)
        self.cv3 = SP(5)
        self.cv4 = SP(5)
        self.DCNV4 = Bottleneck_DCNV4(4 * c3, c2)
 
    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.DCNV4(torch.cat(y, 1))
########################### SPPELAN DCNv4 END##############################################


########################### CSPNeXt_AKconv ##############################################

@MODELS.register_module()
class CSPNeXt_AKconv1(BaseBackbone):
    """CSPNeXt backbone used in RTMDet.

    Args:
        arch (str): Architecture of CSPNeXt, from {P5, P6}.
            Defaults to P5.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        use_akconv: bool = False,   #是否在主干网络中使用 akconv
        use_dcnv4: bool = False,    #是否在CSPLayer的CSPNeXtBlock中使用 dcnv4
        use_sppelan: bool = False,  #是否在主干网络中使用 sppelan 替换 SPPF
        channel_attention_type: str = 'CA',  #CSPLayer的通道注意力类型
        arch_ovewrite (list): Overwrite default arch settings.
            Defaults to None.
        spp_kernel_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Defaults to (5, 9, 13).
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. Defaults to dict(type='BN', requires_grad=True).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(
        self,
        arch: str = 'P5',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        use_akconv: bool = False,   #是否在主干网络中使用 akconv
        use_dcnv4: bool = False,    #是否在CSPLayer的CSPNeXtBlock中使用 dcnv4
        use_sppelan: bool = False,  #是否在主干网络中使用 sppelan 替换 SPPF
        expand_ratio: float = 0.5,
        arch_ovewrite: dict = None,
        spp_kernel_sizes: Sequence[int] = (5, 9, 13),
        channel_attention: bool = True,
        channel_attention_type: str = 'CA',  #CSPLayer的通道注意力类型
        conv_cfg: Optional[ConfigType] = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU'),
        norm_eval: bool = False,
        init_cfg: Optional[ConfigType] = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        #判断是否使用AK_Conv
        #conv = AKConv if use_akconv else ConvModule
        self.stem = nn.Sequential(
            ConvModule(
                3,
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
        # self.stem = nn.Sequential(
        #     CoordConv(
        #         3,
        #         int(arch_setting[0][0] * widen_factor // 2),
        #         3,
        #         padding=1,
        #         stride=2,
        #         norm_cfg=norm_cfg,
        #         act_cfg=act_cfg),
            # ConvModule(
            #     int(arch_setting[0][0] * widen_factor // 2),
            #     int(arch_setting[0][0] * widen_factor // 2),
            #     3,
            #     padding=1,
            #     stride=1,
            #     norm_cfg=norm_cfg,
            #     act_cfg=act_cfg),
            CoordConv(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            c3 = int(out_channels * 0.5)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            #判断是否使用AK_Conv
            if use_akconv:
                conv_layer = conv(
                in_channels,
                out_channels,
                9,
                2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
                stage.append(conv_layer) 
            else:
                conv_layer = conv(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(conv_layer)
            if use_spp:
                if use_sppelan:
                    spp = SPPELAN(
                        out_channels,
                        out_channels,
                        c3,                        
                        )
                    # spp = SPPELAN_DCNV4(
                    #     out_channels,
                    #     out_channels,
                    #     c3,                        
                    #     )
                    stage.append(spp)
                else:
                    spp = SPPBottleneck(
                        out_channels,
                        out_channels,
                        kernel_sizes=spp_kernel_sizes,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        )
                    stage.append(spp)
            csp_layer = CSPLayer1(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                use_cspnext_block=True,
                use_dcnv4=use_dcnv4,
                expand_ratio=expand_ratio,
                channel_attention=channel_attention,
                channel_attention_type = channel_attention_type,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                )
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True) -> None:
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
    
########################### CSPNeXt_AKconv END##############################################