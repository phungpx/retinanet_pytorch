# RetinaNet Pytorch

# 1. References
- [1] [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)
- [2] https://github.com/yhenon/pytorch-retinanet

# 2. Retina Net
## 2.1 Retina Net Architecture
<div align="center">
	<img src="https://user-images.githubusercontent.com/61035926/168408849-69bbbf54-89ab-4787-a5a5-987fd0350e68.png" width="800">
</div>

* [Backbone using all variants of DenseNet and ResNet](https://github.com/phungpx/retinanet_pytorch/tree/main/flame/core/model/backbone)
```python3
# investigate backbone (input shape and output shape), I will update more backbone for experiments (next step: efficient net B0-B7)
cd flame/core/model/backbone
python resnet.py --version <resnet18 -> resnet110> --pretrained <if use pretrained weight>
python densenet.py --version <densenet121 -> densenet201> --pretrained <if use pretrained weight>
```
* [FPN](https://github.com/phungpx/retinanet_pytorch/blob/main/flame/core/model/fpn.py)
```python3
# investigate fpn (input shape and output shape)
cd flame/core/model/
python fpn.py
```
* [Regression and Classification Head](https://github.com/phungpx/retinanet_pytorch/tree/main/flame/core/model/head)
```python3
# investigate head (input shape and output shape)
cd flame/core/model/head
python efficient_head.py
python head.py
```
* [Anchor Generation](https://github.com/phungpx/retinanet_pytorch/blob/main/flame/core/model/anchor_generator.py)
```python3
# investigate anchor generator (input shape and output shape)
cd flame/core/model/
python anchor_generator.py
```
* [Retina Net: combinating all parts together including Resnet/Desenet (backbone), FPN (neck), Regressor&Classifier(head) and Anchors](https://github.com/phungpx/retinanet_pytorch/tree/main/flame/core/model)

## 2.2 Loss
* Focal Loss: for computating loss of classification head.
<div align="center">
	<img src="https://user-images.githubusercontent.com/61035926/168409172-8cf14d1c-9c69-4109-856b-e95086731001.png" width="500">
</div>

* Smooth L1: for computating loss of regression head.
<div align="center">
	<img src="https://user-images.githubusercontent.com/61035926/168409111-7cd3507c-0596-4605-a560-7aa106628079.png" width="600">
</div>

* [Focal Loss](https://github.com/phungpx/retinanet_pytorch/blob/main/flame/core/loss/focal_loss.py)

## 2.3 mAP (mean Average Precision)
This is my technical note for mAP [here](https://docs.google.com/presentation/d/1OHEU1H1EaBNGDofxkptDZvnNXrhPo2rzbMj7mugHgGs/edit#slide=id.ge19bca7490_0_429) borrowed heavily knowledge in awsome [repo](https://github.com/rafaelpadilla/Object-Detection-Metrics).
* [mAP for evaluating model](https://github.com/phungpx/retinanet_pytorch/tree/main/flame/core/metric/)

## 2.4 Visualization
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
## 4.1 VOC2007 and VOC2012
<In progress ...>
## 4.2 COCO
<In progress ...>
## 4.3 Labelme format Dataset
<In progress ...>
