import numpy as np
import torch
import torch.nn as nn

from fast.model.custom_layers import AdaIN


class StyleLoss(nn.Module):
    def __init__(self, epsilon=1e-5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        loss = []

        for key, value in y_true.items():
            var_true, mean_true = torch.var_mean(y_true[key], dim=(2, 3), keepdim=True)
            std_true = torch.sqrt(var_true + self.epsilon)

            var_pred, mean_pred = torch.var_mean(y_pred[key], dim=(2, 3), keepdim=True)
            std_pred = torch.sqrt(var_pred + self.epsilon)

            mean_loss = torch.sum(torch.square(mean_true - mean_pred))
            std_loss = torch.sum(torch.square(std_true - std_pred))

            loss.append(mean_loss + std_loss)

        return torch.mean(torch.stack(loss))


class ContentLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, output_features, adain_output):
        """_summary_

        Args:
            output_features (_type_): _description_
            adain_output (_type_): _description_
        """

        # print(f"output features: {adain_output.shape}")

        return torch.sum((output_features - adain_output) ** 2)
