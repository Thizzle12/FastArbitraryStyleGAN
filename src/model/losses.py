import torch
import torch.nn as nn

from src.model.custom_layers import AdaIN


class StyleLoss(nn.Module):
    def __init__(self, epsilon=1e-5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def forward(self, y_true, y_pred):
        n_features = len(y_true)

        loss = []

        for i in range(n_features):
            mean_true, var_true = 0, 0
            std_true = torch.sqrt(var_true + self.epsilon)

            mean_pred, var_pred = 0, 0
            std_pred = torch.sqrt(var_pred + self.epsilon)

            mean_loss = torch.sum(torch.square(mean_true - mean_pred))
            std_loss = torch.sum(torch.square(std_true - std_pred))

            loss.append(mean_loss + std_loss)

        return torch.mean(loss)


class ContentLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.adain = AdaIN()

    def forward(self, output_features):
        adain_output = self.adain(output_features)

        torch.sum((output_features[-1] - adain_output) ** 2)
        pass
