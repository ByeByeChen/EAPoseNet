import torch
import torch.nn as nn
import math
from typing import Optional, Sequence, Tuple
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
from mmpose.registry import MODELS
from mmengine.model import constant_init, kaiming_init, normal_init
from mmpose.utils.typing import ConfigType

'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret

class CoordConv(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3,padding=1,stride=1,with_r=False,  conv_cfg: Optional[ConfigType] = None, norm_cfg: ConfigType = dict(type='SyncBN', momentum=0.03, eps=0.001),act_cfg: ConfigType = dict(type='SiLU'), init_cfg: Optional[ConfigType] = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')) -> None:
        super().__init__(init_cfg=init_cfg)

        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        #self.conv = nn.Conv2d(in_size, out_channels,k,padding=padding,stride=stride)
        self.conv = nn.Sequential(nn.Conv2d(in_size, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,bias=None),
                                  nn.BatchNorm2d(out_channels),
                                  nn.SiLU())
        #self.conv = ConvModule(in_channels,out_channels,k,padding=padding,stride=stride,norm_cfg=norm_cfg,act_cfg=act_cfg),
    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret