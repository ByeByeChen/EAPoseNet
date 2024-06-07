# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch.nn as nn
from mmengine.model import constant_init, kaiming_init, normal_init
from DCNv4.modules.dcnv4 import DCNv4

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


########################### yolo_Conv ##############################################
class yolo_Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # if d > 1:
        #     k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
        # if p is None:
        #     p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        #self.conv = nn.Conv2d(c1, c2, k, s, groups=g, dilation=d, bias=False)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        
        # kaiming_init(self.conv,a=math.sqrt(5),distribution='uniform',mode='fan_in',nonlinearity='leaky_relu')
        # print("初始化SPPELAN的Conv2d,初始值为："+str(self.conv.weight))
        # constant_init(self.bn, 1)
        # print("初始化SPPELAN的BatchNorm2d,初始值为："+str(self.bn.weight))       
        
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

########################### yolo_Conv END##############################################



########################### DCNv4 #################################################
class DCNV4_Conv(nn.Module):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
 
        if inc != ouc:
            self.stem_conv = yolo_Conv(inc, ouc, k=1)
        self.dcnv4 = DCNv4(ouc, kernel_size=k, stride=s, pad=autopad(k, p, d), group=g, dilation=d)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = yolo_Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        kaiming_init(self.dcnv4,a=math.sqrt(5),distribution='uniform',mode='fan_in',nonlinearity='leaky_relu')
        #print("初始化DCNv4的DCNv4,初始值为："+str(self.dcnv4.weight))
        constant_init(self.bn, 1)
        print("初始化DCNv4的BatchNorm2d,初始值为："+str(self.bn.weight))   

    def forward(self, x):
        if hasattr(self, 'stem_conv'):
            x = self.stem_conv(x)
        x = self.dcnv4(x, (x.size(2), x.size(3)))
        x = self.act(self.bn(x))
        return x
########################### DCNv4 END##############################################