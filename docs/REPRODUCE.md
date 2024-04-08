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
To reproduce the ablation studies, you can set the `OVERLAP_TOPK`, `ALPHA` and `BETA` to `0` to disable the `ARP LQ`, `VS` and `AOC LQ` respectively.
```shell
## baseline
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=0 MODEL.ALPHA 0.0 MODEL.BETA 0.0

## ARP LQ
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=3 MODEL.ALPHA 0.0 MODEL.BETA 0.0

## VS
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=0 MODEL.ALPHA 0.05 MODEL.BETA 0.0

## AOC LQ
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=0 MODEL.ALPHA 0.0 MODEL.BETA 0.75

## ARP LQ + VS
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=3 MODEL.ALPHA 0.05 MODEL.BETA 0.0

## ARP LQ + AOC LQ
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=3 MODEL.ALPHA 0.0 MODEL.BETA 0.75

### VS + AOC LQ
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=0 MODEL.ALPHA 0.05 MODEL.BETA 0.75
```