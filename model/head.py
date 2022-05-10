import torch
from torch import nn
from typing import List, Tuple, Union


# Trandition Conv Block with Attention
class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        conv_repeat: int = 2,
        resconnect: bool = False,
        use_attention: bool = False,
    ) -> None:
        super(ConvBNReLU, self).__init__()
        if conv_repeat < 1:
            raise ValueError('Convolution must be repeated more than one time.')

        self.resconnect = resconnect
        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            *[
                nn.Conv2d(
                    in_channels=out_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ] * (conv_repeat - 1)
        )
        # self.attention = CBAMBlock(in_channels=out_channels) if use_attention else nn.Identity()
        self.attention = nn.Identity()

        if self.resconnect:
            self.res_transform = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.convs(x)

        if self.resconnect:
            output += self.res_transform(x)

        output = self.attention(output)

        return output


class Regressor(nn.Module):
    def __init__(self, FPN_out_channels: int = 256, mid_out_channels: int = 256, num_anchors: int = 9):
        super(Regressor, self).__init__()

        self.conv = ConvBNReLU(
            in_channels=FPN_out_channels,
            out_channels=mid_out_channels,
            kernel_size=3,
            conv_repeat=4
        )

        self.head_conv = nn.Conv2d(
            in_channels=mid_out_channels,
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
        for x in pyramid_features:
            x = self.conv(x)
            x = self.head_conv(x)
            x = x.permute(0, 2, 3, 1).contiguous()  # B x H x W x (num_anchors * 4)
            x = x.view(x.shape[0], -1, 4).contiguous()  # B x (H * W * num_anchors) x 4

            features.append(x)

        x = torch.cat(features, dim=1)

        return x


class Classifier(nn.Module):
    def __init__(self, FPN_out_channels: int = 256, mid_out_channels: int = 256, num_anchors: int = 9, num_classes: int = 80):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv = ConvBNReLU(
            in_channels=FPN_out_channels,
            out_channels=mid_out_channels,
            kernel_size=3,
            conv_repeat=4
        )

        self.head_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=mid_out_channels,
                out_channels=num_anchors * num_classes,
                kernel_size=3, stride=1, padding=1
            ),
            nn.Sigmoid(),
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
        for x in pyramid_features:
            x = self.conv(x)
            x = self.head_conv(x)
            x = x.permute(0, 2, 3, 1).contiguous()  # B x H x W x (num_anchors * num_classes)

            B, H, W, C = x.shape
            x = x.view(B, H, W, self.num_anchors, self.num_classes).contiguous()
            x = x.view(B, H * W * self.num_anchors, self.num_classes).contiguous()  # B x (H * W * num_anchors) x num_classes

            features.append(x)

        x = torch.cat(features, dim=1)

        return x
