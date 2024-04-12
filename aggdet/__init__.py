# Copyright (c) Facebook, Inc. and its affiliates.
from .modeling.meta_arch import custom_rcnn
from .modeling.roi_heads import aggdet_roi_heads
from .modeling.roi_heads import res5_roi_heads
from .modeling.backbone import swintransformer
from .modeling.backbone import timm
from .modeling.backbone import eva
from .modeling.rpn import customrpn

from .data.datasets import lvis_v1
from .data.datasets import imagenet
from .data.datasets import cc
from .data.datasets import objects365
from .data.datasets import oid
from .data.datasets import coco_zeroshot

try:
    from .modeling.meta_arch import d2_deformable_detr
except:
    pass