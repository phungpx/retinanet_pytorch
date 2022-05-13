import torch
from torch import nn
from typing import List
from .basic_blocks import Swish, SeparableConvBNSwish


__all__ = ['Regressor', 'Classifier']


class Regressor(nn.Module):
    def __init__(
        self,
        FPN_out_channels: int = 256,  # out_channels of all feature maps of FPN
        num_anchors: int = 9,  # num_scales * num_aspect_ratios
        num_layers: int = 4,  # number of convolution layers in regression head
        num_pyramid_levels: int = 5  # number of pyramid feautures which are output of FPN (neck) # P3 -> P7
    ):
        super(Regressor, self).__init__()
        self.conv_layers = nn.ModuleList(
            [
                SeparableConvBNSwish(
                    in_channels=FPN_out_channels,
                    out_channels=FPN_out_channels,
                    kernel_size=3,
                    use_batch_norm=False,
                    use_activation=False,
                )
                for _ in range(num_layers)
            ]
        )

        self.batch_norms_levels = nn.ModuleList(
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

        self.act = Swish()
        self.header = SeparableConvBNSwish(
            in_channels=FPN_out_channels,
            out_channels=num_anchors * 4,
            kernel_size=3,
            use_batch_norm=False,
            use_activation=False,
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
        for x, batch_norm_layers in zip(pyramid_features, self.batch_norms_levels):
            for conv, batch_norm in zip(self.conv_layers, batch_norm_layers):
                x = self.act(batch_norm(conv(x)))

            x = self.header(x)
            x = x.permute(0, 2, 3, 1).contiguous()  # B x H x W x (num_anchors * 4)
            x = x.view(x.shape[0], -1, 4).contiguous()  # B x (H * W * num_anchors) x 4

            features.append(x)

        x = torch.cat(features, dim=1)

        return x


class Classifier(nn.Module):
    def __init__(
        self,
        FPN_out_channels: int = 256,  # out_channels of all feature maps of FPN
        num_anchors: int = 9,  # num_scales * num_aspect_ratios
        num_classes: int = 80,  # num_classes of dataset (excluding background, just setting for forceground)
        num_layers: int = 4,  # number of convolution layers in classification head
        num_pyramid_levels: int = 5  # number of pyramid feautures which are output of FPN (neck) # P3 -> P7
    ):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv_layers = nn.ModuleList(
            [
                SeparableConvBNSwish(
                    in_channels=FPN_out_channels,
                    out_channels=FPN_out_channels,
                    kernel_size=3,
                    use_batch_norm=False,
                    use_activation=False,
                )
                for _ in range(num_layers)
            ]
        )

        self.batch_norms_levels = nn.ModuleList(
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

        self.act = Swish()

        self.header = SeparableConvBNSwish(
            in_channels=FPN_out_channels,
            out_channels=num_anchors * num_classes,
            kernel_size=3,
            use_batch_norm=False,
            use_activation=False,
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
        for x, batch_norm_layers in zip(pyramid_features, self.batch_norms_levels):
            for conv, batch_norm in zip(self.conv_layers, batch_norm_layers):
                x = self.act(batch_norm(conv(x)))

            x = self.header(x)
            x = x.permute(0, 2, 3, 1).contiguous()  # B x H x W x (num_anchors * num_classes)

            B, H, W, C = x.shape
            x = x.view(B, H, W, self.num_anchors, self.num_classes).contiguous()
            x = x.view(B, H * W * self.num_anchors, self.num_classes).contiguous()  # B x (H * W * num_anchors) x num_classes

            features.append(x)

        x = torch.cat(features, dim=1)
        x = x.sigmoid()

        return x


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pyramid_features = (
        torch.rand(size=[1, 256, 64, 64], dtype=torch.float32, device=device),  # P3
        torch.rand(size=[1, 256, 32, 32], dtype=torch.float32, device=device),  # P4
        torch.rand(size=[1, 256, 16, 16], dtype=torch.float32, device=device),  # P5
        torch.rand(size=[1, 256, 8, 8], dtype=torch.float32, device=device),  # P6
        torch.rand(size=[1, 256, 4, 4], dtype=torch.float32, device=device),  # P7
    )

    classifier = Classifier(num_classes=1, num_anchors=9)
    regressor = Regressor(num_anchors=9)

    with torch.no_grad():
        cls_preds = classifier(pyramid_features)
        reg_preds = regressor(pyramid_features)

    print(f"Parameters of Classifier: {sum((p.numel() for p in classifier.parameters() if p.requires_grad))}")
    print(f"Parameters of Regressor: {sum((p.numel() for p in regressor.parameters() if p.requires_grad))}")
    print(f"Cls Preds Shape: {cls_preds.shape}")
    print(f"Reg Preds Shape: {cls_preds.shape}")
