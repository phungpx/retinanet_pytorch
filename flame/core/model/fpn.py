import torch
from torch import nn
from typing import List


class FPN(nn.Module):
    def __init__(
        self,
        C3_out_channels: int,
        C4_out_channels: int,
        C5_out_channels: int,
        FPN_out_channels: int = 256
    ) -> None:
        super(FPN, self).__init__()

        # M5 = conv1x1(C5) -> P5 = conv3x3(M5)
        self.C5_conv1x1 = nn.Conv2d(
            in_channels=C5_out_channels,
            out_channels=FPN_out_channels,
            kernel_size=1, stride=1, padding=0
        )
        self.M5_conv3x3 = nn.Conv2d(
            in_channels=FPN_out_channels,
            out_channels=FPN_out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.M5_2x = nn.Upsample(scale_factor=2, mode='nearest')

        # M4 = conv1x1(C4) + upsample(M5) -> P4 = conv3x3(M4)
        self.C4_conv1x1 = nn.Conv2d(
            in_channels=C4_out_channels,
            out_channels=FPN_out_channels,
            kernel_size=1, stride=1, padding=0
        )
        self.M4_conv3x3 = nn.Conv2d(
            in_channels=FPN_out_channels,
            out_channels=FPN_out_channels,
            kernel_size=3, stride=1, padding=1,
        )
        self.M4_2x = nn.Upsample(scale_factor=2, mode='nearest')

        # M3 = conv1x1(C3) + upsample(M4) -> P3 = conv3x3(M3)
        self.C3_conv1x1 = nn.Conv2d(
            in_channels=C3_out_channels,
            out_channels=FPN_out_channels,
            kernel_size=1, stride=1, padding=0
        )
        self.M3_conv3x3 = nn.Conv2d(
            in_channels=FPN_out_channels,
            out_channels=FPN_out_channels,
            kernel_size=3, stride=1, padding=1
        )

        # P6 = conv3x3(C5), with stride=2
        self.P6_conv3x3 = nn.Conv2d(
            in_channels=C5_out_channels,
            out_channels=FPN_out_channels,
            kernel_size=3, stride=2, padding=1
        )

        # P7 = conv3x3(ReLU(P6)), with stride=2
        self.P7_ReLU = nn.ReLU()
        self.P7_conv3x3 = nn.Conv2d(
            in_channels=FPN_out_channels,
            out_channels=FPN_out_channels,
            kernel_size=3, stride=2, padding=1
        )

        self.C3_out_channels = C3_out_channels
        self.C4_out_channels = C4_out_channels
        self.C5_out_channels = C5_out_channels

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        '''
            Args: features outputed from backbone (C3, C4, C5)
                # C3: B x C3 x (H / 2 ^ 3) x (W / 2 ^ 3)
                # C4: B x C4 x (H / 2 ^ 4) x (W / 2 ^ 4)
                # C5: B x C5 x (H / 2 ^ 5) x (W / 2 ^ 5)
            Outputs:
                # C3: B x C_FPN x (H / 2 ^ 3) x (W / 2 ^ 3)
                # C4: B x C_FPN x (H / 2 ^ 4) x (W / 2 ^ 4)
                # C5: B x C_FPN x (H / 2 ^ 5) x (W / 2 ^ 5)
                # C6: B x C_FPN x (H / 2 ^ 6) x (W / 2 ^ 5)
                # C7: B x C_FPN x (H / 2 ^ 7) x (W / 2 ^ 5)
        '''
        C3, C4, C5 = features

        assert C3.shape[1] == self.C3_out_channels, 'out_channels is not compatible.'
        assert C4.shape[1] == self.C4_out_channels, 'out_channels is not compatible.'
        assert C5.shape[1] == self.C5_out_channels, 'out_channels is not compatible.'

        M5 = self.C5_conv1x1(C5)
        P5 = self.M5_conv3x3(M5)

        M4 = self.C4_conv1x1(C4) + self.M5_2x(M5)
        P4 = self.M4_conv3x3(M4)

        M3 = self.C3_conv1x1(C3) + self.M4_2x(M4)
        P3 = self.M3_conv3x3(M3)

        # compute P6 directly from C5 of backbone
        P6 = self.P6_conv3x3(C5)

        # compute P7 directly from P6
        P7 = self.P7_conv3x3(self.P7_ReLU(P6))

        return [P3, P4, P5, P6, P7]


if __name__ == "__main__":
    import time
    import torch
    from backbone import load_backbone

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone, out_channels = load_backbone(backbone_name='resnet34', pretrained=False)
    fpn = FPN(
        C3_out_channels=out_channels['C3'],
        C4_out_channels=out_channels['C4'],
        C5_out_channels=out_channels['C5'],
        FPN_out_channels=256,
    )

    backbone = backbone.to(device)
    fpn = fpn.to(device)

    dummy_input = torch.rand(size=[1, 3, 224, 224], dtype=torch.float32, device=device)

    with torch.no_grad():
        t1 = time.time()
        _, _, C3, C4, C5 = backbone(dummy_input)
        features = fpn([C3, C4, C5])
        t2 = time.time()

    print(f"Input Shape: {dummy_input.shape}")
    print(f"Prams of Backbone: {sum((p.numel() for p in backbone.parameters() if p.requires_grad))}")
    print(f"Params of FPN: {sum((p.numel() for p in fpn.parameters() if p.requires_grad))}")
    print(f"Processing Time: {t2 - t1}s")
    print(f"Features Shape:")
    for i, feature in enumerate(features, 0):
        print(f'P#{i} - Shape: {feature.shape}')
