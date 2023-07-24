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


class StyleNetwork(Enum):
    RESNET18 = auto()
    RESNET50 = auto()
    VGG16 = auto()


class ResNet:
    def __init__(
        self,
        model_type: StyleNetwork = StyleNetwork.RESNET18,
        use_gpu: bool = torch.cuda.is_available(),
    ) -> None:
        self.return_layers = {
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
        }

        self.device = torch.device("cuda:0" if use_gpu else "cpu")

        if model_type == StyleNetwork.RESNET18:
            self._resnet18()
        else:
            self._resnet50()

    def _resnet18(self):
        self.architecture = resnet18(weights=ResNet18_Weights.DEFAULT).to(self.device)

    def _resnet50(self):
        self.architecture = resnet50(weights=ResNet50_Weights.DEFAULT).to(self.device)


def build_style_model(
    model_type: StyleNetwork = StyleNetwork.RESNET18,
    use_gpu: bool = torch.cuda.is_available(),
):
    """_summary_

    Args:
        model_type (StyleNetwork, optional): _description_. Defaults to StyleNetwork.RESNET18.

    Returns:
        _type_: _description_
    """

    model = ResNet(model_type=model_type, use_gpu=use_gpu)
    backbone, return_layers = model.architecture, model.return_layers

    style_model = IntermediateLayerGetter(backbone, return_layers=return_layers)

    return style_model
