# Reproduce Instructions

## main results
To reproduce the main results:

OV-COCO
```shell
## Detic on the OV-COCO with a ResNet-50 backbone
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml

## CoDet on the OV-COCO with a ResNet-50 backbone
python train_net.py --eval-only --config-file configs/CoDet_RN50_COCO.yaml
```

OV-LVIS
```shell
## Detic on the OV-LVIS with a SwinB backbone
python train_net.py --eval-only --config-file configs/Detic_SwinB_LVIS.yaml

## CoDet on the OV-LVIS with a ResNet-50 backbone
python train_net.py --eval-only --config-file configs/CoDet_RN50_LVIS.yaml

## Codet on the OV-LVIS with a SwinB backbone
python train_net.py --eval-only --config-file configs/CoDet_SwinB_LVIS.yaml

## Codet on the OV-LVIS with an EVA02 backbone
python train_net.py --eval-only --config-file configs/CoDet_EVA02_LVIS.yaml
```

## ablation studies
To reproduce the ablation studies, you can set the `OVERLAP_TOPK`, `ALPHA` and `BETA` to `0` to disable the `ARP LQ`, `AOC VS` and `AOC LQ` respectively.
```shell
## baseline
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=0 MODEL.ALPHA 0.0 MODEL.BETA 0.0

## ARP LQ (Aggregated Region-Proposal -- Localization Quality Estimation)
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=3 MODEL.ALPHA 0.0 MODEL.BETA 0.0

## AOC VS (Aggregated Object-Classification -- Visual Similarity Estimation)
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=0 MODEL.ALPHA 0.05 MODEL.BETA 0.0

## AOC LQ (Aggregated Object-Classification -- Localization Quality Estimation)
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=0 MODEL.ALPHA 0.0 MODEL.BETA 0.75

## ARP LQ + AOC VS
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=3 MODEL.ALPHA 0.05 MODEL.BETA 0.0

## ARP LQ + AOC LQ
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=3 MODEL.ALPHA 0.0 MODEL.BETA 0.75

### AOC VS + AOC LQ
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=0 MODEL.ALPHA 0.05 MODEL.BETA 0.75
```

## core functions
The core functions of `ARP LQ`, `AOC VS` and `AOC LQ`are as follows:

- `ARP LQ` for Faster R-CNN: [customrpn.py L235-L240](../aggdet/modeling/rpn/customrpn.py#L235)
- `ARP LQ` for CenterNet2: [centernet.py L659-L675](../third_party/CenterNet2/centernet/modeling/dense_heads/centernet.py#L659)
- `AOC VS` (unified) : [zero_shot_classifier.py L121-L125](../aggdet/modeling/roi_heads/zero_shot_classifier.py#L121)
- `AOC LQ` for Faster R-CNN: [aggdet_fast_rcnn.py L111-L117](../aggdet/modeling/roi_heads/aggdet_fast_rcnn.py#L111)
- `AOC LQ` for CenterNet2: [aggdet_roi_heads.py L162-L164](../aggdet/modeling/roi_heads/aggdet_roi_heads.py#L162)
