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
        var_content, mean_content = torch.var_mean(content, dim=(2, 3), keepdim=True)
        var_style, mean_style = torch.var_mean(style, dim=(2, 3), keepdim=True)

        std_content = torch.sqrt(var_content + self.epsilon)
        std_style = torch.sqrt(var_style + self.epsilon)
        output = std_style * (content - mean_content) / (std_content) + mean_style

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

        # self.bn = nn.BatchNorm2d(in_channels // 2 * scale_factor**2)

        # in_channels * 4, H, W -> in_channels, H * 2, W * 2
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.PReLU(in_channels // 2)

    def forward(self, input):
        x = self.conv(input)
        x = self.conv2(x)
        # x = self.bn(x)
        x = self.pixel_shuffle(x)
        x = self.activation(x)
        return x


class CustomConv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_relu: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_relu = use_relu

        self.padding = nn.ReflectionPad2d(1)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="valid",
            stride=1,
        )

        self.act = nn.PReLU(out_channels)

    def forward(self, inputs):
        # Pad
        x = self.padding(inputs)
        x = self.conv(x)

        if self.use_relu:
            x = self.act(x)

        return x
