from __future__ import annotations

from typing import Optional

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes: int = 2, pretrained: bool = True, pretrained_backbone: Optional[bool] = None):
    """
    Create a Faster R-CNN detection model.

    Args:
        num_classes: number of classes including background. For apples: 2 (background, apple)
        pretrained: load COCO-pretrained weights.
        pretrained_backbone: optional override for backbone pretraining.
    """
    if pretrained:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
