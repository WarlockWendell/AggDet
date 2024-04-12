# AggDet
This repo is the implementation of [Training-free Boost for Open-Vocabulary Object Detection with Confidence Aggregation]()

## Abstract
Open-vocabulary object detection (OVOD) aims at localizing and recognizing visual objects from novel classes unseen at the training time.
Whereas, empirical studies reveal that advanced detectors generally assign lower scores to those novel instances, which are inadvertently suppressed during inference by commonly adopted greedy strategies like Non-Maximum Suppression (NMS), leading to sub-optimal detection performance for novel classes. 
This paper systematically investigates this problem with the commonly-adopted two-stage OVOD paradigm.
Specifically, in the region-proposal stage, proposals that contain novel instances showcase lower objectness scores, since they are treated as background proposals during the training phase.
Meanwhile, in the object-classification stage, novel objects share lower region-text similarities (i.e., classification scores) due to the biased visual-language alignment by seen training samples.
To alleviate this problem, this paper introduces two advanced measures to adjust confidence scores and conserve erroneously dismissed objects: (1) a class-agnostic localization quality estimate via overlap degree of region/object proposals, and (2) a text-guided visual similarity estimate with proxy prototypes for novel classes.
Integrated with adjusting techniques specifically designed for the region-proposal and object-classification stages, this paper derives the aggregated confidence estimate for the open-vocabulary object detection paradigm AggDet.

![framewroks](./assets/framework.png)

## Preparations
- Installation

    Following the [Installation instructions](https://github.com/CVMI-Lab/CoDet/blob/main/README.md#installation) of [CoDet](https://github.com/CVMI-Lab/CoDet) to setup environment.

    Setup environment
    ```shell
    conda create --name aggdet python=3.8 -y && conda activate aggdet
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    git clone https://github.com/WarlockWendell/AggDet.git
    ```
    Install [Apex](https://github.com/NVIDIA/apex#linux) and [xFormer](https://github.com/facebookresearch/xformers#installing-xformers) (You can skip this part if you do not use EVA-02 backbone)
    ```shell script
    pip install ninja
    pip install -v -U git+https://github.com/facebookresearch/xformers.git@7e05e2caaaf8060c1c6baadc2b04db02d5458a94
    git clone https://github.com/NVIDIA/apex && cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && cd ..
    ```
    Install detectron2 and other dependencies
    ```shell
    cd AggDet/third_party/detectron2
    pip install -e .
    cd ../..
    pip install -r requirements.txt
    ```

- Datasets
  
  Please refer to [DATA.md](./docs/DATA.md) for more details.

- Pretrained weights

  You can download the pre-trained weights from the official github repos of [Detic](https://github.com/facebookresearch/Detic/blob/main/docs/MODEL_ZOO.md) and [CoDet](https://github.com/CVMI-Lab/CoDet?tab=readme-ov-file#model-zoo), and put them under the `<AGGDET_ROOT>/ckpt/models` directory.
  
  |model|dataset|download|
  |:---:|:---:|:---:|
  |[Detic_RN50](./configs/Detic_RN50_COCO.yaml) | COCO |[model](https://dl.fbaipublicfiles.com/detic/Detic_OVCOCO_CLIP_R50_1x_max-size_caption.pth) |
  |[CoDet_RN50](./configs/CoDet_RN50_COCO.yaml) | COCO |[model](https://drive.google.com/file/d/1uYX7Jm61TghEtop94fMymBS6AUR66T8k/view?usp=sharing) |
  |[Detic_SwinB](./configs/Detic_SwinB_LVIS.yaml) | LVIS | [model](https://dl.fbaipublicfiles.com/detic/Detic_LbaseI_CLIP_SwinB_896b32_4x_ft4x_max-size.pth) |
  |[CoDet_RN50](./configs/CoDet_RN50_LVIS.yaml) | LVIS | [model](https://drive.google.com/file/d/1-chsmrh5fahOOSa4G2o5Mi6W2mGuMtG-/view?usp=sharing)|
  |[CoDet_SwinB](./configs/CoDet_SwinB_LVIS.yaml) | LVIS|[model](https://drive.google.com/file/d/1ut1K8IsdD2A4uK0xVtPRDg1r4FubH8Pq/view?usp=sharing) |
  |[CoDet_EVA02](./configs/CoDet_EVA02_LVIS.yaml)|LVIS| [model](https://drive.google.com/file/d/1oILkFkIlbEgCCLqCLyJJ5ZDHG1bd0aWN/view?usp=sharing)|

## Inference
Take Detic with a ResNet50 backbone on the OV-COCO dataset as an example.
```shell
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml
```

You can modify the fllowing parameters in the [yaml file](./configs/Detic_RN50_COCO.yaml#L456) to adjust the parameters described in the paper.
```yaml
OVERLAP_TOPK: 3
ALPHA: 0.05
BETA: 0.75
```
For example, use the following command to test the baseline model:
```shell
python train_net.py --eval-only --config-file configs/Detic_RN50_COCO.yaml  \
MODEL.OVERLAP_TOPK=0 MODEL.ALPHA 0.0 MODEL.BETA 0.0
```

You can change the `config-file` to change the model and dataset. Refer to [REPRODUCE.md](./docs/REPRODUCE.md) for more details.

## Citation
```
@article{
  title={Training-free Boost for Open-Vocabulary Object Detection with Confidence Aggregation},
  author={Yanhao Zheng, Kai Liu},
  journal={arXiv preprint arXiv:xxxx.xxxx},
  year={2024}
}
```

## Acknowledgment
AggDet is built upon the awesome works [Codet](https://github.com/CVMI-Lab/CoDet), [EVA](https://github.com/baaivision/EVA/tree/master) and [Detic](https://github.com/facebookresearch/Detic). Many thanks for their wonderful works. 

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](./docs/LICENSE) file for details.