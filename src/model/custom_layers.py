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

    def forward(self, content: torch.Tensor, style: torch.Tensor):
        x = content  # Content
        y = style  # Style

        std_x = torch.sqrt(var_x + self.epsilon)
        std_y = torch.sqrt(var_y + self.epsilon)
        output = std_y * (x - mean_x) / (std_x) + mean_y

        return output


class Upscale(nn.Module):
    def __init__(
        self,
        in_channels: int,
        scale_factor: int = 2,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // 2,
            out_channels=in_channels // 2 * scale_factor**2,
            kernel_size=3,
            stride=1,
            padding="same",
        )

        # in_channels * 4, H, W -> in_channels, H * 2, W * 2
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.PReLU(in_channels // 2)

    def forward(self, input):
        x = self.conv(input)
        x = self.conv2(x)
        x = self.pixel_shuffle(x)
        x = self.activation(x)
        return x
