import torch
from torch import nn
from typing import List, Tuple, Dict


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.,
        lamda: float = 50.,
        device: str = 'cpu',
        mean: Tuple[float, float, float, float] = [0., 0., 0., 0.],
        std: Tuple[float, float, float, float] = [0., 0., 0., 0.],
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # this paper set 0.25
        self.gamma = gamma  # this paper set 2.0
        self.lamda = lamda  # this paper set 50
        self.device = device

        # using for normalizing regression boxes (num_boxes x 4)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 4)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 4)

        self.positive_iou_threshold = 0.5
        self.negative_iou_threshold = 0.4

    def _compute_iou(self, anchors: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        '''
        args:
            anchors: [num_anchors, 4]
            box_type: [x1, y1, x2, y2]
            target_boxes: [num_boxes, 4]
            box_type: [x1, y1, x2, y2]
        output:
            ious: [num_anchors, num_boxes]
        references: https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
        '''
        # calculate intersection areas of anchors and target boxes
        inter_width = torch.min(anchors[:, 2].unsqueeze(dim=1), target_boxes[:, 2]) - torch.max(anchors[:, 0].unsqueeze(dim=1), target_boxes[:, 0])
        inter_height = torch.min(anchors[:, 3].unsqueeze(dim=1), target_boxes[:, 3]) - torch.max(anchors[:, 1].unsqueeze(dim=1), target_boxes[:, 1])
        inter_width = torch.clamp(inter_width, min=0.)  # num_anchors x num_boxes
        inter_height = torch.clamp(inter_height, min=0.)  # num_anchors x num_boxes
        inter_areas = inter_width * inter_height  # num_anchors x num_boxes
        # calculate union areas of anchors and target boxes
        area_anchors = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])  # num_anchors
        area_boxes = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])  # num_boxes
        union_areas = area_anchors.unsqueeze(dim=1) + area_boxes - inter_width * inter_height  # num_anchors x num_boxes
        union_areas = torch.clamp(union_areas, min=1e-8)
        # calculate ious of anchors and target boxes
        ious = inter_areas / union_areas
        return ious

    def _xyxy2xywh(self, boxes: torch.Tensor) -> torch.Tensor:
        '''convert box type from x1, y1, x2, y2 to ctr_x, ctr_y, w, h
        args:
            boxes: [num_boxes, 4]
            type of box: (x1, y1, x2, y2)
        output:
            converted_boxes: [num_boxes, 4]
            type of box: (cx, cy, w, h)
        '''
        w = torch.clamp(boxes[:, 2] - boxes[:, 0], min=1.)
        h = torch.clamp(boxes[:, 3] - boxes[:, 1], min=1.)
        cx = boxes[:, 0] + w / 2.
        cy = boxes[:, 1] + h / 2.

        return torch.stack([cx, cy, w, h], dim=1)

    def _regress_bounding_boxes(
        self, anchors: torch.Tensor, boxes: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        ''' find bounding box regression
        - dx = (x - xa) / wa
        - dy = (y - ya) / ha
        - dw = log(w / wa)
        - dh = log(h / ha)
        args:
            anchors: num_anchors x 4, box type: (x1, y1, x2, y2)
            boxes: num_anchors x 4, box type: (x1, y1, x2, y2)
        outputs:
            boxes_regression: num_anchors x 4, box type: (dx, dy, dw, dh)
        '''
        assert anchors.shape == boxes.shape, f'boxes shape: {boxes.shape}, anchors shape: {anchors.shape}'
        xywh_boxes = self._xyxy2xywh(boxes=boxes)  # num_anchors x 4
        xywh_anchors = self._xyxy2xywh(boxes=anchors)  # num_anchors x 4

        dx = (xywh_boxes[:, 0] - xywh_anchors[:, 0]) / xywh_anchors[:, 2]  # num_anchors
        dy = (xywh_boxes[:, 1] - xywh_anchors[:, 1]) / xywh_anchors[:, 3]  # num_anchors
        dw = torch.log(xywh_boxes[:, 2] / xywh_anchors[:, 2])  # num_anchors
        dh = torch.log(xywh_boxes[:, 3] / xywh_anchors[:, 3])  # num_anchors

        reg_target = torch.stack((dx, dy, dw, dh), dim=1)  # num_anchors x 4 
        reg_target = (reg_target - mean) / std

        return reg_target

    def _smooth_l1_loss(self, input: torch.Tensor, target: torch.Tensor, reduction: str = 'none', beta: float = 1.) -> torch.Tensor:
        '''smooth l1 for calculating regression loss,
            reference: https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
        '''
        assert input.shape == target.shape, f'boxes shape: {input.shape}, anchors shape: {target.shape}'
        diff = torch.abs(input - target)
        loss = torch.where(torch.le(diff, beta), 0.5 * diff.pow(2) / beta, diff - 0.5 * beta)
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError(f'invalid reduction mode: {reduction}')
        return loss

    def _focal_loss(self, cls_pred: torch.Tensor, cls_target: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
        assert cls_pred.shape == cls_target.shape, f'expected target batch {cls_target.shape} to match target batch {cls_pred.shape}'
        positive_classes_loss = - self.alpha * (1. - cls_pred).pow(self.gamma) * torch.log(cls_pred)
        positive_classes_loss = cls_target * positive_classes_loss  # FL(pos_cls) = - alpha * [(1 - pred) ** gamma] * log(pred)
        positive_classes_loss = positive_classes_loss
        negative_classes_loss = - (1. - self.alpha) * cls_pred.pow(self.gamma) * torch.log(1. - cls_pred)
        negative_classes_loss = (1. - cls_target) * negative_classes_loss  # FL(neg_cls) = - (1 - alpha) * [pred ** gamma] * log(1 - pred)
        negative_classes_loss = negative_classes_loss
        loss = cls_target * positive_classes_loss + (1. - cls_target) * negative_classes_loss
        loss = torch.where(torch.ne(cls_target, -1.), loss, torch.zeros_like(loss, device=self.device))  # remove ignored values
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'none':
            loss = loss
        else:
            raise NotImplementedError(f'invalid reduction mode: {reduction}')
        return loss

    def loss_fn(
        self,
        cls_preds: torch.Tensor,
        reg_preds: torch.Tensor,
        anchors: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''calculate object detection loss
        args:
            cls_preds: [batch_size, num_anchors, num_classes]
            reg_preds: [batch_size, num_anchors, 4] (box_type: x1, y1, x2, y2)
            annotations:
                list of annotation:
                - annotation['labels']: [num_labels]
                - annotation['boxes']: [num_boxes, 4]
                - annotation['image_id']: [idx]
            anchors: [1, num_anchors, 4] (box_type: x1, y1, x2, y2)
        output:
            cls_loss: torch.Tensor (1)
            reg_loss: torch.Tensor (1)
        '''
        anchors = anchors[0, :, :]
        cls_losses, reg_losses = [], []
        for i, target in enumerate(targets):
            cls_pred = cls_preds[i, :, :]  # num_anchors x num_classes
            cls_pred = torch.clamp(cls_pred, min=1e-4, max=1.0 - 1e-4)
            reg_pred = reg_preds[i, :, :]  # num_anchors x 4
            target_boxes = target['boxes']
            target_labels = target['labels']

            ious = self._compute_iou(anchors, target_boxes)  # num_anchors x num_boxes
            iou_maxes, iou_argmaxes = torch.max(ious, dim=1)  # num_anchors x 1

            # find indice of positive anchors (>= positive_iou_threshold)
            positive_indices = torch.ge(iou_maxes, self.positive_iou_threshold)  # num_anchors
            num_positive_anchors = positive_indices.sum()

            # find indice of negative anchors (< negative_iou_threshold)
            negative_indices = torch.lt(iou_maxes, self.negative_iou_threshold)  # num_anchors

            # attach label for each anchor which has max iou value with boxes
            anchor_labels = target_labels[iou_argmaxes]
            anchor_boxes = target_boxes[iou_argmaxes, :]  # ground truth box correspones each anchors

            # initial cls_targets with all ignored values
            cls_target = -1 * torch.ones(cls_pred.shape, device=self.device)  # num_anchors x num_classes
            # add negative values
            cls_target[negative_indices, :] = 0
            # add positive values
            cls_target[positive_indices, :] = 0
            cls_target[positive_indices, anchor_labels[positive_indices].to(torch.int64)] = 1

            # compute classification loss
            cls_loss = self._focal_loss(cls_pred, cls_target, reduction='sum')
            cls_losses.append(cls_loss / torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute regression loss
            if num_positive_anchors > 0:
                _reg_pred = reg_pred[positive_indices, :]
                positive_anchors = anchors[positive_indices, :]
                positive_boxes = anchor_boxes[positive_indices, :]
                reg_target = self._regress_bounding_boxes(
                    anchors=positive_anchors,
                    boxes=positive_boxes,
                    mean=self.mean,
                    std=self.std,
                )
                _reg_target = reg_target.to(self.device)
                reg_loss = self._smooth_l1_loss(_reg_pred, _reg_target, reduction='mean', beta=1. / 9.)
                reg_losses.append(reg_loss)
            else:
                reg_losses.append(torch.tensor(0).float().to(self.device))

        cls_loss = torch.stack(cls_losses).mean()
        reg_loss = torch.stack(reg_losses).mean()

        return cls_loss, reg_loss

    def forward(
        self,
        cls_preds: torch.Tensor,
        reg_preds: torch.Tensor,
        anchors: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_loss, reg_loss = self.loss_fn(cls_preds, reg_preds, anchors, targets)
        return cls_loss, self.lamda * reg_loss
