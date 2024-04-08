# This code is modified from https://github.com/facebookresearch/Detic/blob/main/detic/modeling/roi_heads/zero_shot_classifier.py
# Modified by YH Zheng
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) YH Zheng.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec
import json
import numpy as np

class ZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        zs_weight_path: str,
        detection_weight_path: str,
        zs_weight_dim: int = 512,
        use_bias: float = 0.0, 
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
        alpha: float = 0.0,
        visual_prototype: str = '',
        categories_info: str = '',
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature
        self.alpha = alpha

        with open(categories_info, 'r') as f:
            categories = json.load(f)
        self.unseen_cls = np.array(categories['novel'])
        self.seen_cls = np.array(categories['base'])
        self.unused_cls = np.array(categories['unused'])

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.linear = nn.Linear(input_size, zs_weight_dim)
        
        if zs_weight_path == 'rand':
            zs_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            zs_weight = torch.tensor(
                np.load(zs_weight_path), 
                dtype=torch.float32).permute(1, 0).contiguous() # D x C
        
        if detection_weight_path == 'rand':
            detection_weight = torch.randn((zs_weight_dim, num_classes))
            nn.init.normal_(zs_weight, std=0.01)
        else:
            detection_weight = torch.tensor(
                np.load(detection_weight_path),
                dtype=torch.float32).permute(1, 0).contiguous()  # D x C

        detection_weight = torch.cat(
            [detection_weight, detection_weight.new_zeros((zs_weight_dim, 1))],
            dim=1)  # D x (C + 1)

        if self.norm_weight:
            zs_weight = F.normalize(zs_weight, p=2, dim=0)
            detection_weight = F.normalize(detection_weight, p=2, dim=0)

        if zs_weight_path == 'rand':
            self.zs_weight = nn.Parameter(zs_weight)
        else:
            self.register_buffer('zs_weight', zs_weight)
            self.register_buffer('detection_weight', detection_weight)

        assert self.detection_weight.shape[1] == num_classes + 1, self.detection_weight.shape
        
        if visual_prototype is not '':
            visual_prototype = torch.from_numpy(np.load(visual_prototype))
        else:
            visual_prototype = torch.tensor(np.load(zs_weight_path), dtype=torch.float32)
        
        self.register_buffer('visual_prototype', visual_prototype)
       

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'detection_weight_path': cfg.MODEL.ROI_BOX_HEAD.DETECTION_WEIGHT_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
            'alpha': cfg.MODEL.ALPHA,
            'visual_prototype': cfg.MODEL.VISUAL_PROTOTYPE,
            'categories_info': cfg.MODEL.CATEGORIES_INFO
        }

    def forward(self, x, classifier=None):
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)
        scores2 = x @ self.visual_prototype.T

        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous()  # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            zs_weight = self.detection_weight
        
        x = torch.mm(x, zs_weight)
        
        x[:, self.unseen_cls] =  self.norm_temperature * (x[:, self.unseen_cls]  + self.alpha * (scores2[:, self.unseen_cls]))
        x[:, self.seen_cls] = self.norm_temperature * x[:, self.seen_cls]
        x[:, self.unused_cls] = self.norm_temperature * x[:, self.unused_cls]
        
        if self.use_bias:
            x = x + self.cls_bias
        return x
    