# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
from PIL import Image
import torch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.structures import Instances, Boxes, PolygonMasks
from pycocotools import mask as coco_mask
from .detection_utils import build_strong_augmentation

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class DatasetMapperTwoCrop(DatasetMapper):
    def __init__(
        self,
        is_train=True,
        *,
        augmentation,
        image_format,
    ):
        self.augmentation = augmentation
        logging.getLogger(__name__).info(
            "[COCO_Instance_LSJ_Augment_Dataset_Mapper] Full TransformGens used in training: {}".format(str(self.augmentation))
        )
        self.strong_augmentation = build_strong_augmentation(is_train)
        self.img_format = image_format
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # Load image.
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        padding_mask = np.ones(image.shape[:2],dtype=np.uint8)
        image, transforms = T.apply_transform_gens(self.augmentation, image)

        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        # add strong augmentations later.
        image1 = Image.fromarray(image.astype("uint8"), "RGB")
        image_aug = np.array(self.strong_augmentation(image1))
        image_aug = torch.as_tensor(np.ascontiguousarray(image_aug).transpose(2, 0, 1))

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if not instances.has('gt_masks'): 
                instances.gt_masks = PolygonMasks([])  # for negative examples
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
            # import ipdb; ipdb.set_trace()
            dataset_dict["instances"] = instances

        dataset_dict_strong = copy.deepcopy(dataset_dict)
        dataset_dict_strong["image"] = image_aug

        return dataset_dict, dataset_dict_strong