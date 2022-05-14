# RetinaNet Pytorch

# 1. References
- [1] [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)
- [2] https://github.com/yhenon/pytorch-retinanet

# 2. Project Structure
* [Backbone using all variants of DenseNet and ResNet](https://github.com/phungpx/retinanet_pytorch/tree/main/flame/core/model/backbone)
```
# investigate backbone (input shape and output shape), I will update more backbone for experiments (next step: efficient net B0-B7)
cd flame/core/model/backbone
python resnet.py --version <resnet18 -> resnet110> --pretrained <if use pretrained weight>
python densenet.py --version <densenet121 -> densenet201> --pretrained <if use pretrained weight>
```
* [FPN](https://github.com/phungpx/retinanet_pytorch/blob/main/flame/core/model/fpn.py)
```bash
# investigate fpn (input shape and output shape)
cd flame/core/model/
python fpn.py
```
* [Regression and Classification Head](https://github.com/phungpx/retinanet_pytorch/tree/main/flame/core/model/head)
```bash
# investigate head (input shape and output shape)
cd flame/core/model/head
python efficient_head.py
python head.py
```
* [Retina Net](https://github.com/phungpx/retinanet_pytorch/tree/main/flame/core/model)
* [Focal Loss](https://github.com/phungpx/retinanet_pytorch/blob/main/flame/core/loss/focal_loss.py)
* [Anchor Generation](https://github.com/phungpx/retinanet_pytorch/blob/main/flame/core/model/anchor_generator.py)
```bash
cd flame/core/model/head
python efficient_head.py
python head.py
```
* [mAP for evaluating model](https://github.com/phungpx/retinanet_pytorch/tree/main/flame/core/metric/)
* [Visualization for predicting results](https://github.com/phungpx/retinanet_pytorch/blob/main/flame/handlers/region_predictor.py)

# 3. Usage
* Training: training with Focal loss on train_set and evaluating with Focal Loss on both train_set (again) and valid_set.
```bash
CUDA_VISIBLE_DEVICES=<gpu indice> python -m flame configs/PASCAL/pascal_training.yaml
```
* Evaluation: evaluating with mAP metric and visualizing all predictions for test_set.
```bash
CUDA_VISIBLE_DEVICES=<gpu indice> python -m flame configs/PASCAL/pascal_testing.yaml
```

# 4. Experiments
<In progress ...>
