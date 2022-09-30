import torch
from torch import nn as nn
from torchvision.models import resnet as resnet

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY


class ResNetLoss(nn.Module):

    def __init__(self, cnn, feature_layers=3):
        super(ResNetLoss, self).__init__()
        self.conv1 = cnn.conv1
        self.bn1 = cnn.bn1
        self.relu = cnn.relu
        self.maxpool = cnn.maxpool
        self.feature_layers = feature_layers
        assert feature_layers <= 4
        self.blocks = nn.ModuleList()
        for _, layer in zip(range(4 + feature_layers), cnn.children()):
            if isinstance(layer, nn.Sequential):
                self.blocks.append(layer)
        self.l1 = nn.L1Loss().cuda()

    def forward(self, predict_features, target_features, weights=[1]):
        if len(weights) == 1:
            weights = weights * self.feature_layers
        x = predict_features
        x = self.bn1(self.conv1(x))
        x = self.maxpool(self.relu(x))

        y = target_features
        y = self.bn1(self.conv1(y))
        y = self.maxpool(self.relu(y))

        losses = []
        for block in self.blocks:
            x = block(x)
            y = block(y)
            losses.append(self.l1(x, y))
        total_loss = 0
        for weight, loss in zip(weights, losses):
            total_loss += loss * weight
        return total_loss


@LOSS_REGISTRY.register()
class DualPerceptualLoss(nn.Module):
    """Dual Perceptual Loss with commonly used style losses.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(DualPerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        # start: resnet loss (ESRGAN-DP)
        self.resnet = resnet.resnet50()
        self.resnet_loss = ResNetLoss(self.resnet).cuda()
        # end: resnet loss (ESRGAN-DP)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # # start: calculate normal perceptual loss
        # if self.perceptual_weight > 0:
        #     percep_loss = 0
        #     for k in x_features.keys():
        #         if self.criterion_type == 'fro':
        #             percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
        #         else:
        #             percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
        #     percep_loss *= self.perceptual_weight
        # else:
        #     percep_loss = None
        # # end: calculate normal perceptual loss

        # start: calculate dual perceptual loss (ESRGAN-DP)
        percep_loss1 = self.resnet_loss(x, gt.detach())

        if self.perceptual_weight > 0:
            percep_loss2 = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss2 += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss2 += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss2 *= self.perceptual_weight
        else:
            percep_loss2 = None

        a = percep_loss1.item()
        b = percep_loss2.item()
        mu = (1 / 0.5)
        DP_loss = mu * (b / a) * percep_loss1 + percep_loss2
        # end: calculate dual perceptual loss (ESRGAN-DP)

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        # # normal perceptual loss return
        # return percep_loss2, style_loss

        # dual perceptual loss return (ESRGAN-DP)
        return DP_loss, style_loss, percep_loss1, percep_loss2

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
