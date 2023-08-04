from enum import Enum, auto

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    VGG16_Weights,
    resnet18,
    resnet50,
    vgg16,
)
from torchvision.models._utils import IntermediateLayerGetter

from fast.model.custom_layers import Upscale


class StyleNetwork(Enum):
    RESNET18 = auto()
    RESNET50 = auto()
    VGG16 = auto()


class ResNet:
    def __init__(
        self,
        model_type: StyleNetwork = StyleNetwork.RESNET18,
    ) -> None:
        """_summary_

        Args:
            model_type (StyleNetwork, optional): _description_. Defaults to StyleNetwork.RESNET18.
            use_gpu (bool, optional): _description_. Defaults to torch.cuda.is_available().
        """
        self.return_layers = {
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
        }

        if model_type == StyleNetwork.RESNET18:
            self._resnet18()
        else:
            self._resnet50()

    def _resnet18(self):
        self.architecture = resnet18(weights=ResNet18_Weights.DEFAULT)

    def _resnet50(self):
        self.architecture = resnet50(weights=ResNet50_Weights.DEFAULT)


def build_encoder(
    model_type: StyleNetwork = StyleNetwork.RESNET18,
):
    """_summary_

    Args:
        model_type (StyleNetwork, optional): _description_. Defaults to StyleNetwork.RESNET18.

    Returns:
        _type_: _description_
    """

    model = ResNet(model_type=model_type)
    backbone, return_layers = model.architecture, model.return_layers

    encoder = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # Lock layers of the encoder. Only the decoder should be trained for this model.
    for param in encoder.parameters():
        param.requires_grad = False

    return encoder


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        scale_factor: int = 2,
        n_layers: int = 5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_layers = n_layers

        up_layers = [
            Upscale(in_channels=in_channels // 2**i, scale_factor=scale_factor)
            for i in range(n_layers)
        ]

        self.upscaling_layers = nn.ModuleList(up_layers)

        self.out = nn.Conv2d(
            in_channels=in_channels // 2 ** (n_layers),
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding="same",
        )

    def forward(self, input):
        x = input
        for i in range(self.n_layers):
            x = self.upscaling_layers[i](x)

        x = self.out(x)

        return x
