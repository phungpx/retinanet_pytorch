class SeparableConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_batch_norm: bool = False,
        use_activation: bool = False,
    ):
        super(SeparableConvBlock, self).__init__()
        self.depthwise_conv = Conv2dStaticSamePadding(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=in_channels,
            bias=False
        )

        self.pointwise_conv = Conv2dStaticSamePadding(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1, stride=1, padding=0
        )

        self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3) if use_batch_norm else nn.Identity()
        self.act = Swish() is use_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.act(self.bn(x))

        return x


class Conv2dStaticSamePadding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        groups: int = 1,
        dilation: int = 1,
        **kwargs
    ):
        super(Conv2dStaticSamePadding, self).__init__()
        self.conv = nn.Conv2d(
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

        if isinstance(self.kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        extra_H = (math.ceil(W / self.stride[1]) - 1) * self.stride[1] - W + self.kernel_size[1]
        extra_V = (math.ceil(H / self.stride[0]) - 1) * self.stride[0] - H + self.kernel_size[0]

        left = extra_H // 2
        right = extra_H - left
        top = extra_V // 2
        bottom = extra_V - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)

        return x


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
