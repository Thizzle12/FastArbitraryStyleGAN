import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(
        self,
        epsilon=1e-5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def forward(self, inputs):
        x = inputs[0]  # Content
        y = inputs[1]  # Style

        std_x = torch.sqrt(var_x + self.epsilon)
        std_y = torch.sqrt(var_y + self.epsilon)
        output = std_y * (x - mean_x) / (std_x) + mean_y

        return output
