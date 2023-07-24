import torch
import torch.nn as nn


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
            out_channels=in_channels * scale_factor**2,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        # in_channels * 4, H, W -> in_channels, H * 2, W * 2
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.PReLU(in_channels)

    def foward(self, input):
        x = self.conv(input)
        x = self.pixel_shuffle(x)
        x = self.activation(x)
        return x


class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d()

    def forward(self, input):
        pass


class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        pass
