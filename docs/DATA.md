# Prepare datasets for AggDet

Following the [Prepare Datasets](https://github.com/CVMI-Lab/CoDet?tab=readme-ov-file#prepare-datasets) of [CoDet](https://github.com/CVMI-Lab/CoDet) to prepare the COCO and LVIS datasets.

First, download COCO and LVIS data place them in the following way:
```
lvis/
    lvis_v1_train.json
    lvis_v1_val.json
coco/
    train2017/
    val2017/
    annotations/
        captions_train2017.json
        instances_train2017.json 
        instances_val2017.json
```


Then we follow [OVR-CNN](https://github.com/alirezazareian/ovr-cnn/blob/master/ipynb/003.ipynb) to create the open-vocabulary COCO split. The converted files should be like:
```
coco/
    zero-shot/
        instances_train2017_seen_2.json
        instances_val2017_all_2.json
```

We further follow [Detic](https://github.com/facebookresearch/Detic/tree/main) to pre-process the annotation format for easier evaluation:
```
python tools/get_coco_zeroshot_oriorder.py --data_path datasets/coco/zero-shot/instances_train2017_seen_2.json
python tools/get_coco_zeroshot_oriorder.py --data_path datasets/coco/zero-shot/instances_val2017_all_2.json
```
And process the category infomation:
```
python tools/get_lvis_cat_info.py --ann datasets/coco/zero-shot/instances_train2017_seen_2_oriorder.py
```

Next, prepare the open-vocabulary LVIS training set using
```
python tools/remove_lvis_rare.py --ann datasets/lvis/lvis_v1_train.json
```
This will generate `datasets/lvis/lvis_v1_train_norare.json`.

Then generate `datasets/lvis/lvis_v1_train_norare_cat_info.json` by running
```
python tools/get_lvis_cat_info.py --ann datasets/lvis/lvis_v1_train_norare.json --add_freq
```

After the preparation is complete, your directories should be organized as follows:
```
datasets/
├── coco
│   ├── annotations
│   ├── train2017
│   ├── val2017
│   └── zero-shot
├── lvis
│   ├── lvis_v1_train.json
│   ├── lvis_v1_train_norare_cat_info.json
│   ├── lvis_v1_train_norare.json
│   └── lvis_v1_val.json
├── metadata
│   ├── coco_categories_info.json
│   ├── codet_cc3m_clip_a+cname.npy
│   ├── codet_cococap_clip_a+cname.npy
│   ├── codet_coco_clip_a+cname.npy
│   ├── codet_lvis_v1_clip_a+cname.npy
│   ├── codet_o365_clip_a+cnamefix.npy
│   ├── detic_coco_clip_a+cname.npy
│   ├── detic_lvis_v1_clip_a+cname.npy
│   ├── lvis_categories_info.json
│   └── lvis_v1_train_cat_info.json
└── prototypes
    ├── CoDet_COCO_RN50.npy
    ├── CoDet_LVIS_EVA02.npy
    ├── CoDet_LVIS_RN50.npy
    ├── CoDet_LVIS_SWINB.npy
    ├── Detic_COCO_RN50.npy
    └── Detic_LVIS_SWINB.npy
```
where the prototypes directory contains pre-extracted visual prototypes, and you can use [`extract_training_set_features.py`](./extract_training_set_features.py) to generate them.
```shell
### For Detic with a ResNet50 backbone on the OV-COCO dataset.
python extract_training_set_features.py --dataset coco --detection-weight datasets/metadata/detic_coco_clip_a+cname.npy --config-file config/Detic_RN50_COCO.py
```