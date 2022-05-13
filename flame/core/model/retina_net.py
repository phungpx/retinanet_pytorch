import math
import torch
from torch import nn
from torchvision.ops.boxes import batched_nms
from typing import Tuple, List, Dict, Optional

from .neck.fpn import FPN
from .backbone import load_backbone
# from .head.head import Regressor, Classifier
from .head.efficient_head import Regressor, Classifier
from .anchor_generator import AnchorGenerator
from .box_transform.box_decoder import BoxDecoder
from .box_transform.box_clipper import BoxClipper


class RetinaNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        backbone_name: str = 'resnet50',
        backbone_pretrained: bool = False,
        FPN_out_channels: int = 256,
        scales: List[float] = [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)],
        aspect_ratios: List[float] = [0.5, 1, 2.],
        iou_threshold: float = 0.2,
        score_threshold: float = 0.2,
    ) -> None:
        super(RetinaNet, self).__init__()
        self.num_classes = num_classes
        # for postprocessing
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        # for anchor generation
        self.scales = scales
        self.aspect_ratios = aspect_ratios

        # anchor generator
        self.anchor_generator = AnchorGenerator(
            anchor_scale=4,  # scale for C1 and C2 layers. (each layers downsize 2 times of size)
            scales=scales,
            aspect_ratios=aspect_ratios
        )

        # backbone
        self.backbone, backbone_out_channels = load_backbone(
            backbone_name=backbone_name, pretrained=backbone_pretrained
        )

        # neck
        self.fpn = FPN(
            C3_out_channels=backbone_out_channels['C3'],
            C4_out_channels=backbone_out_channels['C4'],
            C5_out_channels=backbone_out_channels['C5'],
            FPN_out_channels=FPN_out_channels,
        )

        # head
        self.regressor = Regressor(
            FPN_out_channels=FPN_out_channels,
            # mid_out_channels=FPN_out_channels,
            num_anchors=len(scales) * len(aspect_ratios),
        )

        self.classifier = Classifier(
            FPN_out_channels=FPN_out_channels,
            # mid_out_channels=FPN_out_channels,
            num_anchors=len(scales) * len(aspect_ratios),
            num_classes=num_classes,
        )

        # using for inference to find predicted bounding boxes
        self.box_decoder = BoxDecoder()
        self.box_clipper = BoxClipper()

        # initialize model
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

        # prior = 0.01
        # self.classifier.header.weight.data.fill_(0)
        # self.classifier.header.bias.data.fill_(-math.log((1.0 - prior) / prior))
        # self.regressor.header.weight.data.fill_(0)
        # self.regressor.header.bias.data.fill_(0)

        self.freeze_batchnorm

    @property
    def freeze_batchnorm(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, C3, C4, C5 = self.backbone(x=inputs)
        pyramid_features = self.fpn(features=[C3, C4, C5])
        reg_preds = self.regressor(pyramid_features=pyramid_features)
        cls_preds = self.classifier(pyramid_features=pyramid_features)

        anchors = self.anchor_generator(inputs, pyramid_features)

        return cls_preds, reg_preds, anchors

    def predict(self, inputs: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        _, _, C3, C4, C5 = self.backbone(inputs)
        pyramid_features = self.fpn([C3, C4, C5])
        reg_preds = self.regressor(pyramid_features)  # B x all_anchors x 4
        cls_preds = self.classifier(pyramid_features)  # B x all_anchors x num_classes

        anchors = self.anchor_generator(inputs, pyramid_features)  # 1 x all_anchors x 4

        # get predicted boxes which are decoded from reg_preds and anchors
        batch_boxes = self.box_decoder(anchors=anchors, regression=reg_preds)  # B x all_anchors x 4, decoded boxes
        batch_boxes = self.box_clipper(boxes=batch_boxes, image_width=inputs.shape[3], image_height=inputs.shape[2])

        batch_scores, batch_classes = torch.max(cls_preds, dim=2)  # B x all_anchors, choose class with max confidence in each box.
        batch_scores_over_threshold = (batch_scores > self.score_threshold)  # B x all_anchors, remove class of box with confidence lower than threshold.

        preds = []
        batch_size = inputs.shape[0]
        for i in range(batch_size):  # loop for each sample.
            sample_scores_over_threshold = batch_scores_over_threshold[i]  # all_anchors, sample_scores_over_threshold
            if sample_scores_over_threshold.sum() == 0:  # sample has no valid boxes.
                preds.append(
                    {
                        'boxes': torch.FloatTensor([[0, 0, 1, 1]]),  # 1 pixel.
                        'labels': torch.FloatTensor([-1]),
                        'scores': torch.FloatTensor([0])
                    }
                )
                continue

            sample_boxes = batch_boxes[i]  # all_anchors x 4
            sample_scores = batch_scores[i]  # all_anchors
            sample_classes = batch_classes[i]  # all_anchors

            valid_boxes = sample_boxes[sample_scores_over_threshold, :]  # n_valid_scores x 4
            valid_scores = sample_scores[sample_scores_over_threshold]  # n_valid_scores
            valid_classes = sample_classes[sample_scores_over_threshold]  # n_valid_scores

            # determind what boxes will be kept by nms algorithm
            keep_indices = batched_nms(
                boxes=valid_boxes,
                scores=valid_scores,
                idxs=valid_classes,
                iou_threshold=self.iou_threshold
            )

            if keep_indices.shape[0] != 0:
                kept_boxes = valid_boxes[keep_indices, :]  # num_keep_boxes x 4
                kept_scores = valid_scores[keep_indices]  # num_keep_boxes
                kept_classes = valid_classes[keep_indices]  # num_keep_boxes

                preds.append({'boxes': kept_boxes, 'labels': kept_classes, 'scores': kept_scores})
            else:
                preds.append(
                    {
                        'boxes': torch.FloatTensor([[0, 0, 1, 1]]),
                        'labels': torch.FloatTensor([-1]),
                        'scores': torch.FloatTensor([0])
                    }
                )

        return preds


class Model(nn.Module):
    def __init__(
        self,
        pretrained_weight: Optional[str] = None,
        num_classes: int = 80,
        backbone_name: str = 'resnet50',
        backbone_pretrained: bool = False,
        FPN_out_channels: int = 256,
        scales: List[float] = [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)],
        aspect_ratios: List[float] = [0.5, 1, 2.],
        iou_threshold: float = 0.2,
        score_threshold: float = 0.2,
    ) -> None:
        super(Model, self).__init__()
        self.retina_net = RetinaNet(
            num_classes=num_classes,
            backbone_name=backbone_name,
            backbone_pretrained=backbone_pretrained,
            FPN_out_channels=FPN_out_channels,
            scales=scales,
            aspect_ratios=aspect_ratios,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )

        if pretrained_weight is not None:
            state_dict = torch.load(pretrained_weight, map_location='cpu')
            state_dict.pop('classifier.header.weight')
            state_dict.pop('classifier.header.bias')
            state_dict.pop('regressor.header.weight')
            state_dict.pop('regressor.header.bias')
            self.retina_net.load_state_dict(state_dict, strict=False)

    def state_dict(self):
        return self.retina_net.state_dict()

    def load_state_dict(self, state_dict):
        self.retina_net.load_state_dict(state_dict)

    def forward(self, inputs):
        return self.retina_net(inputs)
            

    def predict(self, inputs):
        return self.retina_net.predict(inputs)


if __name__ == "__main__":
    import time
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    retina_net = Model(
        pretrained_weight=None,
        num_classes=1,
        backbone_name='resnet18',
        backbone_pretrained=False,
    ).to(device)

    retina_net.train()

    dummy_input = torch.rand(size=[1, 3, 224, 224], dtype=torch.float32, device=device)

    t1 = time.time()
    outputs = retina_net(dummy_input)
    t2 = time.time()

    print(f"Input Shape: {dummy_input.shape}")
    print(f"Number of parameters: {sum((p.numel() for p in retina_net.parameters() if p.requires_grad))}")
    print(f"Processing Time: {t2 - t1}s")
    # print(f"output: {outputs}")
