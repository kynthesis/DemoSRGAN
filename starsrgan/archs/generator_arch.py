from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F
import torch

from basicsr.archs.arch_util import default_init_weights, make_layer, pixel_unshuffle


class GaussianNoise(nn.Module):

    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float).to(torch.device('cuda'))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


def get_activation_function(act_type='relu', num_feat=64, negative_slope=0.1, inplace=True):
    if act_type == 'relu':
        return nn.ReLU(inplace)
    elif act_type == 'prelu':
        return nn.PReLU(num_feat)
    elif act_type == 'leakyrelu':
        return nn.LeakyReLU(negative_slope, inplace)
    elif act_type == 'mish':
        return nn.Mish(inplace)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class StarResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: leakyrelu.
        use_noise (bool): Inject Gaussian noise. Default: True.
    """

    def __init__(self, num_feat=64, num_grow_ch=32, act_type='leakyrelu', use_noise=True):
        super(StarResidualDenseBlock, self).__init__()
        self.noise = GaussianNoise() if use_noise else nn.Identity()
        self.conv1x1 = conv1x1(num_feat, num_grow_ch)
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.act = get_activation_function(act_type, num_feat, negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), 1)))
        x2 = x2 + self.conv1x1(x)
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x4 = x4 + x2
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return self.noise(x5 * 0.2 + x)


class StarRRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: leakyrelu.
        use_noise (bool): Inject Gaussian noise. Default: True.
    """

    def __init__(self, num_feat=64, num_grow_ch=32, act_type='leakyrelu', use_noise=True):
        super(StarRRDB, self).__init__()
        self.rdb1 = StarResidualDenseBlock(num_feat, num_grow_ch, act_type, use_noise)
        self.rdb2 = StarResidualDenseBlock(num_feat, num_grow_ch, act_type, use_noise)
        self.rdb3 = StarResidualDenseBlock(num_feat, num_grow_ch, act_type, use_noise)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class StarSRNet(nn.Module):
    """StarSRGAN: Synthetic Technical Accelerated Receptive Super-Resolution Generative Adversarial Networks.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: leakyrelu.
        use_noise (bool): Inject Gaussian noise. Default: True.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 scale=4,
                 num_feat=64,
                 num_block=23,
                 num_grow_ch=32,
                 act_type='leakyrelu',
                 use_noise=True):
        super(StarSRNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            StarRRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch, act_type=act_type, use_noise=use_noise)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.act = get_activation_function(act_type, num_feat, negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.act(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.act(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.act(self.conv_hr(feat)))
        return out


@ARCH_REGISTRY.register()
class LiteSRNet(nn.Module):
    """LiteSRGAN: Lightweight Interpolative Temporal Efficient Super-Resolution Generative Adversarial Networks.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
        use_noise (bool): Inject Gaussian noise. Default: True.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_conv=16,
                 upscale=4,
                 act_type='prelu',
                 use_noise=True):
        super(LiteSRNet, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type
        self.use_noise = use_noise

        self.noise = GaussianNoise() if use_noise else nn.Identity()
        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        activation = get_activation_function(act_type, num_feat, negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return self.noise(out)
