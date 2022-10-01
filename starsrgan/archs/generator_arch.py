from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F
from basicsr.archs.arch_util import default_init_weights, make_layer, pixel_unshuffle
import torch


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DynamicConv2d(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 if_bias=True,
                 K=5,
                 init_weight=False):
        super(DynamicConv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.if_bias = if_bias
        self.K = K

        self.weight = nn.Parameter(
            torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size), requires_grad=True)
        if self.if_bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes), requires_grad=True)
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])
            if self.if_bias:
                nn.init.constant_(self.bias[i], 0)

    def forward(self, inputs):
        x = inputs['x']
        softmax_attention = inputs['weights']
        batch_size, in_planes, height, width = x.size()
        x = x.contiguous().view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size,
                                                                    self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=aggregate_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size)
        else:
            output = F.conv2d(
                x,
                weight=aggregate_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


class StarResidualDenseBlock(nn.Module):
    """Star Residual Dense Block.

    Used in StarRRDB block in StarSRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32, num_networks=5):
        super(StarResidualDenseBlock, self).__init__()
        self.conv1x1 = conv1x1(num_feat, num_grow_ch)
        self.conv1 = DynamicConv2d(num_feat, num_grow_ch, 3, K=num_networks)
        self.conv2 = DynamicConv2d(num_feat + num_grow_ch, num_grow_ch, 3, K=num_networks)
        self.conv3 = DynamicConv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, K=num_networks)
        self.conv4 = DynamicConv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, K=num_networks)
        self.conv5 = DynamicConv2d(num_feat + 4 * num_grow_ch, num_feat, 3, K=num_networks)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x2 = x2 + self.conv1x1(x)
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x4 = x4 + x2
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class StarRRDB(nn.Module):
    """Star Residual in Residual Dense Block.

    Used in StarSRNet in StarSRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32, num_networks=5):
        super(StarRRDB, self).__init__()
        self.star_rdb1 = StarResidualDenseBlock(num_feat, num_grow_ch, num_networks)
        self.star_rdb2 = StarResidualDenseBlock(num_feat, num_grow_ch, num_networks)
        self.star_rdb3 = StarResidualDenseBlock(num_feat, num_grow_ch, num_networks)

    def forward(self, x):
        out = self.star_rdb1(x)
        out = self.star_rdb2(out)
        out = self.star_rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class StarSRNet(nn.Module):
    """Networks consisting of Star Residual in Residual Dense Block, which is used in StarSRGAN.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32, num_networks=5):
        super(StarSRNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = DynamicConv2d(num_in_ch, num_feat, 3, K=num_networks)
        self.body = make_layer(
            StarRRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch, num_networks=num_networks)
        self.conv_body = DynamicConv2d(num_feat, num_feat, 3, K=num_networks)
        # upsample
        self.conv_up1 = DynamicConv2d(num_feat, num_feat, 3, K=num_networks)
        self.conv_up2 = DynamicConv2d(num_feat, num_feat, 3, K=num_networks)
        self.conv_hr = DynamicConv2d(num_feat, num_feat, 3, K=num_networks)
        self.conv_last = DynamicConv2d(num_feat, num_out_ch, 3, K=num_networks)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, weights):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first({'x': feat, 'weights': weights})
        body_feat = self.conv_body(self.body({'x': feat, 'weights': weights}))['x']
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1({'x': F.interpolate(feat, scale_factor=2, mode='nearest'), 'weights': weights}))
        feat = self.lrelu(self.conv_up2({'x': F.interpolate(feat, scale_factor=2, mode='nearest'), 'weights': weights}))
        out = self.conv_last(self.lrelu(self.conv_hr({'x': feat, 'weights': weights})))
        return out


# @ARCH_REGISTRY.register()
# class SRVGGNetCompact(nn.Module):
#     """A compact VGG-style network structure for super-resolution.

#     It is a compact network structure, which performs upsampling in the last layer and no convolution is
#     conducted on the HR feature space.

#     Args:
#         num_in_ch (int): Channel number of inputs. Default: 3.
#         num_out_ch (int): Channel number of outputs. Default: 3.
#         num_feat (int): Channel number of intermediate features. Default: 64.
#         num_conv (int): Number of convolution layers in the body network. Default: 16.
#         upscale (int): Upsampling factor. Default: 4.
#         act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
#     """

#     def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
#         super(SRVGGNetCompact, self).__init__()
#         self.num_in_ch = num_in_ch
#         self.num_out_ch = num_out_ch
#         self.num_feat = num_feat
#         self.num_conv = num_conv
#         self.upscale = upscale
#         self.act_type = act_type

#         self.body = nn.ModuleList()
#         # the first conv
#         self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
#         # the first activation
#         if act_type == 'relu':
#             activation = nn.ReLU(inplace=True)
#         elif act_type == 'prelu':
#             activation = nn.PReLU(num_parameters=num_feat)
#         elif act_type == 'leakyrelu':
#             activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#         self.body.append(activation)

#         # the body structure
#         for _ in range(num_conv):
#             self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
#             # activation
#             if act_type == 'relu':
#                 activation = nn.ReLU(inplace=True)
#             elif act_type == 'prelu':
#                 activation = nn.PReLU(num_parameters=num_feat)
#             elif act_type == 'leakyrelu':
#                 activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#             self.body.append(activation)

#         # the last conv
#         self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
#         # upsample
#         self.upsampler = nn.PixelShuffle(upscale)

#     def forward(self, x):
#         out = x
#         for i in range(0, len(self.body)):
#             out = self.body[i](out)

#         out = self.upsampler(out)
#         # add the nearest upsampled image, so that the network learns the residual
#         base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
#         out += base
#         return out
