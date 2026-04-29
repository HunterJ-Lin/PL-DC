from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOPanopticEvaluator
from detectron2.data import MetadataCatalog
from detectron2.data.dataset_mapper import DatasetMapper

from detrex.data.dataset_mappers import MaskFormerPanopticDatasetMapper,maskformer_semantic_transform_gen

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="ade20k_panoptic_train"),
    mapper=L(MaskFormerPanopticDatasetMapper)(
        augmentation=L(maskformer_semantic_transform_gen)(
            min_size_train=[int(x * 0.1 * 640) for x in range(5, 21)],
            max_size_train=2560,
            min_size_train_sampling="choice",
            enabled_crop=True,
            crop_params=dict(
              crop_type="absolute",
              crop_size=(640,640),
              single_category_max_area=1.0,
            ),
            color_aug_ssd=True,
            img_format="RGB",
        ),
        meta=MetadataCatalog.get("ade20k_panoptic_train"),
        is_train=True,
        image_format="RGB",
        size_divisibility=640,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="ade20k_panoptic_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=640,
                max_size=2560,
            ),
        ],
        is_train=False,
        image_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOPanopticEvaluator)(
    dataset_name="${..test.dataset.names}",
)
