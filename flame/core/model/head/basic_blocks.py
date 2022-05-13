import math
import torch
from torch import nn
from typing import Tuple, Union


__all__ = ['ConvBNReLU', 'SeparableConvBNSwish', 'StaticSamePaddingConv2D', 'Swish']


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        use_activation: bool = False,
        use_batch_norm: bool = False,
    ) -> None:
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False
            ),
            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU() if use_activation else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)

        return output


class SeparableConvBNSwish(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_batch_norm: bool = False,
        use_activation: bool = False,
    ):
        super(SeparableConvBNSwish, self).__init__()
        self.depthwise_conv = StaticSamePaddingConv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=1, bias=False,
        )

        self.pointwise_conv = StaticSamePaddingConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1, stride=1
        )

        self.bn = nn.BatchNorm2d(
            num_features=out_channels, momentum=0.01, eps=1e-3
        ) if use_batch_norm else nn.Identity()

        self.act = Swish() if use_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.act(self.bn(x))

        return x


class StaticSamePaddingConv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        groups: int = 1,
        **kwargs
    ):
        super(StaticSamePaddingConv2D, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            groups=groups,
            **kwargs
        )

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        extra_W = (math.ceil(W / self.stride[1]) - 1) * self.stride[1] - W + self.kernel_size[1]
        extra_H = (math.ceil(H / self.stride[0]) - 1) * self.stride[0] - H + self.kernel_size[0]

        left = extra_W // 2
        right = extra_W - left
        top = extra_H // 2
        bot = extra_H - top

        x = nn.functional.pad(x, [left, right, top, bot])
        x = self.conv2d(x)

        return x


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.sigmoid()
