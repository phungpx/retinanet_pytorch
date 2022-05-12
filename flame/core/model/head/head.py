import torch
from torch import nn
from typing import List, Tuple, Union


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


class Regressor(nn.Module):
    def __init__(
        self,
        FPN_out_channels: int = 256,
        num_anchors: int = 9,
        num_layers: int = 4,
        num_pyramid_levels: int = 5
    ):
        super(Regressor, self).__init__()
        self.convs = nn.ModuleList(
            [
                ConvBNReLU(
                    in_channels=FPN_out_channels,
                    out_channels=FPN_out_channels,
                    kernel_size=3,
                    use_batch_norm=False,
                    use_activation=False,
                )
                for _ in range(num_layers)
            ]
        )

        self.batch_norms = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.BatchNorm2d(num_features=FPN_out_channels, momentum=0.01, eps=1e-3)
                        for _ in range(num_layers)
                    ]
                )
                for _ in range(num_pyramid_levels)
            ]
        )

        self.act = nn.ReLU()
        self.header = nn.Conv2d(
            in_channels=FPN_out_channels,
            out_channels=num_anchors * 4,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, pyramid_features: List[torch.Tensor]) -> torch.Tensor:
        '''
            Args:
                pyramid_features: (P3, P4, P5, P6, P7)
                    P3: B, C_FPN, H / 2 ^ 3, W / 2 ^ 3
                    P4: B, C_FPN, H / 2 ^ 4, W / 2 ^ 4
                    P5: B, C_FPN, H / 2 ^ 5, W / 2 ^ 5
                    P6: B, C_FPN, H / 2 ^ 6, W / 2 ^ 6
                    P7: B, C_FPN, H / 2 ^ 7, W / 2 ^ 7
            Outputs:
                x: Tensor [B, (H3 * W3 * n_anchors
                               + H4 * W4 * n_anchors
                               + H5 * W5 * n_anchors
                               + H6 * W6 * n_anchors
                               + H7 * W7 * n_anchors), 4]
        '''
        features = []
        for x, norms in zip(pyramid_features, self.batch_norms):
            for conv, bn in zip(self.convs, norms):
                x = self.act(bn(conv(x)))

            x = self.header(x)
            x = x.permute(0, 2, 3, 1).contiguous()  # B x H x W x (num_anchors * 4)
            x = x.view(x.shape[0], -1, 4).contiguous()  # B x (H * W * num_anchors) x 4

            features.append(x)

        x = torch.cat(features, dim=1)

        return x


class Classifier(nn.Module):
    def __init__(
        self,
        FPN_out_channels: int = 256,
        num_anchors: int = 9,
        num_classes: int = 80,
        num_layers: int = 4,
        num_pyramid_levels: int = 5
    ):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.convs = nn.ModuleList(
            [
                ConvBNReLU(
                    in_channels=FPN_out_channels,
                    out_channels=FPN_out_channels,
                    kernel_size=3,
                    use_batch_norm=False,
                    use_activation=False,
                )
                for _ in range(num_layers)
            ]
        )

        self.batch_norms = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.BatchNorm2d(num_features=FPN_out_channels, momentum=0.01, eps=1e-3)
                        for _ in range(num_layers)
                    ]
                )
                for _ in range(num_pyramid_levels)
            ]
        )

        self.act = nn.ReLU()

        self.header = nn.Conv2d(
            in_channels=FPN_out_channels,
            out_channels=num_anchors * num_classes,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, pyramid_features: List[torch.Tensor]) -> torch.Tensor:
        '''
            Args:
                pyramid_features: (P3, P4, P5, P6, P7)
                    P3: B, C_FPN, H / 2 ^ 3, W / 2 ^ 3
                    P4: B, C_FPN, H / 2 ^ 4, W / 2 ^ 4
                    P5: B, C_FPN, H / 2 ^ 5, W / 2 ^ 5
                    P6: B, C_FPN, H / 2 ^ 6, W / 2 ^ 6
                    P7: B, C_FPN, H / 2 ^ 7, W / 2 ^ 7
            Outputs:
                x: Tensor [B, (H3 * W3 * n_anchors
                               + H4 * W4 * n_anchors
                               + H5 * W5 * n_anchors
                               + H6 * W6 * n_anchors
                               + H7 * W7 * n_anchors), num_classes]
        '''
        features = []
        for x, norms in zip(pyramid_features, self.batch_norms):
            for conv, bn in zip(self.convs, norms):
                x = self.act(bn(conv(x)))

            x = self.header(x)
            x = x.permute(0, 2, 3, 1).contiguous()  # B x H x W x (num_anchors * num_classes)

            B, H, W, C = x.shape
            x = x.view(B, H, W, self.num_anchors, self.num_classes).contiguous()
            x = x.view(B, H * W * self.num_anchors, self.num_classes).contiguous()  # B x (H * W * num_anchors) x num_classes

            features.append(x)

        x = torch.cat(features, dim=1)
        x = nn.Sigmoid()(x)

        return x
