from .resnet import ResNet
from .densenet import DenseNet
from typing import Optional


def load_backbone(backbone_name: str = 'resnet18', pretrained: bool = False, num_layers: Optional[int] = None):
    resnet_layers_channels = {
        'resnet18': {'C1': 64, 'C2': 64, 'C3': 128, 'C4': 256, 'C5': 512},
        'resnet34': {'C1': 64, 'C2': 64, 'C3': 128, 'C4': 256, 'C5': 512},
        'resnet50': {'C1': 64, 'C2': 256, 'C3': 512, 'C4': 1024, 'C5': 2048},
        'resnet101': {'C1': 64, 'C2': 256, 'C3': 512, 'C4': 1024, 'C5': 2048},
        'resnet152': {'C1': 64, 'C2': 256, 'C3': 512, 'C4': 1024, 'C5': 2048},
    }
    densenet_layers_channels = {
        'densenet121': {'C1': 64, 'C2': 256, 'C3': 512, 'C4': 1024, 'C5': 1024},
        'densenet161': {'C1': 96, 'C2': 384, 'C3': 768, 'C4': 2112, 'C5': 2208},
        'densenet169': {'C1': 64, 'C2': 256, 'C3': 512, 'C4': 1280, 'C5': 1664},
        'densenet201': {'C1': 64, 'C2': 256, 'C3': 512, 'C4': 1792, 'C5': 1920},
    }

    if backbone_name in resnet_layers_channels:
        backbone = ResNet(backbone_name, pretrained=pretrained)
        layers_channels = resnet_layers_channels[backbone_name]
    elif backbone_name in densenet_layers_channels:
        backbone = DenseNet(backbone_name, pretrained=pretrained)
        layers_channels = densenet_layers_channels[backbone_name]
    else:
        raise ValueError(f'Not supported backbone {backbone_name}')

    if num_layers:
        layers_channels = layers_channels[:num_layers]

    return backbone, layers_channels
