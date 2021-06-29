import torch
import torch.nn as nn

# Implementation for pix2pix
# For generator we use segmentation-models pytorch library and create a backbone using efficientnet-b1
# Implementation of Discriminator.
class PatchDiscriminator(nn.Module):
    """Discriminator model."""

    def __init__(self, input_channels, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_channels, num_filters, norm=False)]
        model += [
            self.get_layers(
                num_filters * 2 ** i,
                num_filters * 2 ** (i + 1),
                s=1 if i == (n_down - 1) else 2,
            )
            for i in range(n_down)
        ]
        model += [
            self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)
        ]
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm:
            layers += [nn.BatchNorm2d(nf)]
        if act:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
