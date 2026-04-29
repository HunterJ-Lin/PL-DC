from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.evaluation import COCOEvaluator

from detrex.data.dataset_mappers.mask_former_instance_dataset_mapper import MaskFormerInstanceDatasetMapper,build_transform_gen
dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="cityscapes_fine_instance_seg_train_10sup"),
    mapper=L(MaskFormerInstanceDatasetMapper)(
        augmentations=L(build_transform_gen)(
            min_size_train=[int(x * 0.1 * 1024) for x in range(5, 21)],
            max_size_train=4096,
            min_size_train_sampling="choice",
            enabled_crop=True,
            crop_type="absolute",
            crop_size=(512,1024),
            color_aug_ssd=True,
            img_format="RGB",
            is_train=True,
        ),
        is_train=True,
        image_format="RGB",
        size_divisibility=-1,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="cityscapes_fine_instance_seg_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=1024,
                max_size=2048,
                sample_style="choice",
            ),
        ],
        is_train=False,
        image_format="RGB",
    ),
    num_workers=4,
)

# from detectron2.evaluation.cityscapes_evaluation import CityscapesInstanceEvaluator
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
# dataloader.evaluator = L(CityscapesInstanceEvaluator)(
#     dataset_name="${..test.dataset.names}",
# )
