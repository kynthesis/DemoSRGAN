from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn as nn


@ARCH_REGISTRY.register()
class DegradationPredictor(nn.Module):

    def __init__(self, num_in_ch=3, num_feat=64, num_params=100, num_networks=5, use_bias=True):
        super(DegradationPredictor, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(num_in_ch, num_feat, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat, kernel_size=5, stride=2, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_params, kernel_size=5, stride=1, padding=2, bias=use_bias),
            nn.LeakyReLU(0.2, True),
        ])

        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))

        self.MappingNet = nn.Sequential(*[
            nn.Linear(num_params, 15),
            nn.Linear(15, num_networks),
        ])

    def forward(self, input):
        conv = self.ConvNet(input)
        flat = self.globalPooling(conv)
        out_params = flat.view(flat.size()[:2])
        mapped_weights = self.MappingNet(out_params)
        return out_params, mapped_weights
