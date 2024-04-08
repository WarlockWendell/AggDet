# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) YH Zheng.
import logging
import os
import sys
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.structures import ImageList, Instances, Boxes

from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
    LVISEvaluator,
    COCOEvaluator,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader
from detectron2.utils.logger import setup_logger
from torch.cuda.amp import GradScaler
import numpy as np
import json

# sys.path.insert(0, 'third_party/CenterNet2/')
from third_party.CenterNet2.centernet.config import add_centernet_config

from aggdet.config import add_aggdet_config
from aggdet.data.custom_build_augmentation import build_custom_augmentation
from aggdet.data.custom_dataset_dataloader import  build_custom_train_loader
from aggdet.data.custom_dataset_mapper import CustomDatasetMapper, DetrDatasetMapper
from aggdet.custom_solver import build_custom_optimizer
from aggdet.evaluation.oideval import OIDEvaluator
from aggdet.evaluation.custom_coco_eval import CustomCOCOEvaluator
from aggdet.modeling.utils import reset_cls_test
from aggdet.data.custom_dataset_dataloader import build_custom_test_loader
from tqdm import tqdm
import torch.nn.functional as F

logger = logging.getLogger("detectron2")


def l2_norm(x):
    return F.normalize(x, p=2, dim=-1)


def load_text_embeddings(path):
    text_embeddings = torch.from_numpy(np.load(path)).float()
    print(text_embeddings.shape)
    print(f'text_embeddings mean: {torch.mean(text_embeddings)}, std {torch.std(text_embeddings)}')
    return l2_norm(text_embeddings)


def get_prototype(text_embeddings, visual_embeddings, seen_cls, unseen_cls):
    base_text_embeddings = text_embeddings[seen_cls]
    novel_text_embeddings = text_embeddings[unseen_cls]
    base_visual_embeddings = visual_embeddings[seen_cls]
    text_base_delta = base_text_embeddings[:, None, :] - base_text_embeddings[None, :, :]
    miu_t, sigma_t = torch.mean(text_base_delta), torch.std(text_base_delta)
    visual_base_delta = base_visual_embeddings[:, None, :] - base_visual_embeddings[None, :, :]
    miu_v, sigma_v = torch.mean(visual_base_delta), torch.std(visual_base_delta)
    x = (novel_text_embeddings[:, None, :] - base_text_embeddings[None, :, :] - miu_t) / sigma_t * sigma_v + base_visual_embeddings + miu_v
    novel_visual_embeddings = F.normalize(torch.mean(x, dim=1), dim=1)
    visual_cluster = torch.zeros(visual_embeddings.shape, dtype=torch.float32)
    visual_cluster[unseen_cls] = novel_visual_embeddings
    visual_cluster[seen_cls] = base_visual_embeddings
    return visual_cluster


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_aggdet_config(cfg)
    print(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, \
        distributed_rank=comm.get_rank(), name="aggdet")
    return cfg


def extract_rpn(model, data_loader, cache_dir, feat_log, score_log, label_log):
    model.eval()
    model.cuda()
    st = 0
    for batch_idx, inputs in enumerate(tqdm(data_loader)):
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        boxes = [Boxes(x['instances'].gt_boxes.tensor.cuda()) for x in inputs]
        box_features = model.roi_heads._shared_roi_transform([features[f] for f in features], boxes).mean(dim=[2, 3])
        visual_embedding = model.roi_heads.box_predictor.cls_score.linear(box_features)
        scores = model.roi_heads.box_predictor(box_features)[0]
        feat_log[st:st+box_features.shape[0], :] = visual_embedding.data.cpu().numpy()
        score_log[st:st+box_features.shape[0], :] = scores[:, :-1].data.cpu().numpy()
        label_log[st:st+box_features.shape[0]] = inputs[0]['instances'].gt_classes.data.cpu().numpy()
        st = st+box_features.shape[0]
    
    os.makedirs(f'{cache_dir}/cls_feat_score', exist_ok=True)
    for cls in tqdm(range(score_log.shape[-1])):
        index = label_log == cls
        feat_origin = feat_log[index]
        score = score_log[index]
        np.save(f'{cache_dir}/cls_feat_score/{cls}_origin_feat.npy', feat_origin)
        np.save(f'{cache_dir}/cls_feat_score/{cls}_score.npy', score)


def extract_centernet2(model, data_loader, cache_dir, feat_log, score_log, label_log):
    model.eval()
    model.cuda()
    st = 0
    roi_heads = model.roi_heads
    for batch_idx, inputs in enumerate(tqdm(data_loader)):
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        boxes = [Boxes(x['instances'].gt_boxes.tensor.cuda()) for x in inputs]
        features = [features[f] for f in roi_heads.box_in_features]
        image_sizes = [tuple(inputs[0]['image'].shape[1:])]
        box_features = roi_heads.box_pooler(features, boxes)
        for k in range(roi_heads.num_cascade_stages):
            box_embeddings = roi_heads.box_head[k](box_features)
            visual_embedding = roi_heads.box_predictor[k].cls_score.linear(box_embeddings)
            scores = roi_heads.box_predictor[k](box_embeddings)[0][:, :-1]
            feat_log[st:st+box_features.shape[0], k, :] = visual_embedding.data.cpu().numpy()
            score_log[st:st+box_features.shape[0], k, :] = scores.data.cpu().numpy()
        label_log[st:st+box_features.shape[0]] = inputs[0]['instances'].gt_classes.data.cpu().numpy()
        st = st+box_features.shape[0]
    os.makedirs(f'{cache_dir}/cls_feat_score', exist_ok=True)
    for cls in tqdm(range(score_log.shape[-1])):
        index = label_log == cls
        feat_origin = feat_log[index]
        score = score_log[index]
        np.save(f'{cache_dir}/cls_feat_score/{cls}_origin_feat.npy', feat_origin[:, -1])
        np.save(f'{cache_dir}/cls_feat_score/{cls}_score.npy', score[:, -1, cls])


def get_cluster(score_log, feat_log, cache_dir):
    visual_embeddings = torch.zeros((score_log.shape[-1], feat_log.shape[-1]), dtype=torch.float32)
    for cls in range(score_log.shape[-1]):
        feat_origin = torch.from_numpy(np.load(f'{cache_dir}/cls_feat_score/{cls}_origin_feat.npy')).float()
        if feat_origin.shape[0] < 10:
            continue
        index = np.random.choice(len(feat_origin), 300, True)
        visual_embeddings[cls, :] = F.normalize(torch.mean(feat_origin[index], dim=0), dim=-1)
    return visual_embeddings


def do_extraction(cfg, model, data_loader, cache_dir, text_embeddings):
    feat_dim = 512 # CLIP text embedding dim
    gt_len = 1264884 if 'LVIS' in cache_dir else 656231 # total number of objects
    num_classes = 1203 if 'LVIS' in cache_dir else 80 # number of categories
    categories_info = 'datasets/metadata/lvis_categories_info.json' if 'LVIS' in cache_dir else 'datasets/metadata/coco_categories_info.json'
    with open(categories_info, 'r') as f:
        categories = json.load(f)
    unseen_cls = np.array(categories['novel'])
    seen_cls = np.array(categories['base'])

    if cfg.MODEL.PROPOSAL_GENERATOR.NAME == 'CenterNet':
        feat_log = np.memmap(f'{cache_dir}/feat.mmap', dtype=float, mode='w+', shape=(gt_len, 3, feat_dim))
        score_log = np.memmap(f'{cache_dir}/score.mmap', dtype=float, mode='w+', shape=(gt_len, 3, num_classes))
        label_log = np.memmap(f'{cache_dir}/label.mmap', dtype=float, mode='w+', shape=(gt_len))
        extract_centernet2(model, data_loader, cache_dir, feat_log, score_log, label_log)
    elif 'RPN' in cfg.MODEL.PROPOSAL_GENERATOR.NAME:
        feat_log = np.memmap(f'{cache_dir}/feat.mmap', dtype=float, mode='w+', shape=(gt_len, feat_dim))
        score_log = np.memmap(f'{cache_dir}/score.mmap', dtype=float, mode='w+', shape=(gt_len, num_classes))
        label_log = np.memmap(f'{cache_dir}/label.mmap', dtype=float, mode='w+', shape=(gt_len))
        # feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(gt_len, feat_dim))
        # score_log = np.memmap(f'{cache_dir}/score.mmap', dtype=float, mode='r', shape=(gt_len, num_classes))
        # label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(gt_len))
        extract_rpn(model, data_loader, cache_dir, feat_log, score_log, label_log)
    
    visual_embeddings = get_cluster(score_log, feat_log, cache_dir)
    visual_prototypes = get_prototype(text_embeddings, visual_embeddings, seen_cls, unseen_cls)
    np.save(f'{cache_dir}/visual_prototype.npy', visual_prototypes.cpu().numpy())


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    if args.dataset == 'coco':
        dataset_name = 'coco_zeroshot_train_oriorder'
    elif args.dataset == 'lvis':
        dataset_name = 'lvis_v1_train_norare'
    mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' \
        else DatasetMapper(
            cfg, False, augmentations=build_custom_augmentation(cfg, False))
    data_loader = build_custom_test_loader(cfg, dataset_name, mapper=mapper)

    prefix = os.path.basename(args.config_file).split('.')[0]
    cache_dir = f'./tempdata/{prefix}'
    os.makedirs(cache_dir, exist_ok=True)

    text_embeddings = load_text_embeddings(args.detection_weight)
    do_extraction(cfg, model, data_loader, cache_dir, text_embeddings)


if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument('--dataset', type=str, default='coco', choices=['coco', 'lvis'])
    args.add_argument('--detection-weight', type=str, default='datasets/metadata/detic_coco_clip_a+cname.npy')
    args = args.parse_args()

    if args.num_machines == 1:
        args.dist_url = 'tcp://127.0.0.1:{}'.format(
            torch.randint(11111, 60000, (1,))[0].item())
    else:
        if args.dist_url == 'host':
            args.dist_url = 'tcp://{}:12345'.format(
                os.environ['SLURM_JOB_NODELIST'])
        elif not args.dist_url.startswith('tcp'):
            tmp = os.popen(
                    'echo $(scontrol show job {} | grep BatchHost)'.format(
                        args.dist_url)
                ).read()
            tmp = tmp[tmp.find('=') + 1: -1]
            args.dist_url = 'tcp://{}:12345'.format(tmp)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )