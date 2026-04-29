# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
import multiprocessing as mp
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import functools
import io
import logging
import json
import numpy as np
from itertools import chain
import pycocotools.mask as mask_util
from PIL import Image

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

logger = logging.getLogger(__name__)

_SPLITS_COCO_FORMAT = {}
_SPLITS_COCO_FORMAT["coco"] = {
    "coco_2017_unlabel": (
        "coco/unlabeled2017",
        "coco/annotations/image_info_unlabeled2017.json",
    ),
    "coco_2017_for_voc20": (
        "coco",
        "coco/annotations/google/instances_unlabeledtrainval20class.json",
    ),
}

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {}
seeds = [1]
percentages = [1, 2, 5, 10]
for seed in seeds:
    for perc in percentages:
        _PREDEFINED_SPLITS_COCO["coco"][f"coco_2017_train.{seed}@{perc}"] = ("coco/train2017", f"coco/annotations/instances_train2017.{seed}@{perc}.json")
        _PREDEFINED_SPLITS_COCO["coco"][f"coco_2017_train.{seed}@{perc}-unlabel"] = ("coco/train2017", f"coco/annotations/instances_train2017.{seed}@{perc}-unlabeled.json")
_PREDEFINED_SPLITS_COCO["coco"][f"coco_2017_train@30"] = ("coco/val2014", "coco/annotations/instances_val2014.json")
_PREDEFINED_SPLITS_COCO["coco"][f"coco_2017_train@30-unlabel"] = ("coco/train2017", f"coco/annotations/instances_train2017@{30}-unlabeled.json")
_PREDEFINED_SPLITS_COCO["coco"][f"coco_2017_train_sample@10000"] = ("coco/train2017", "coco/annotations/instances_train2017_sample@10000.json")
_PREDEFINED_SPLITS_COCO["coco"][f"coco_2017_train_sample@1000"] = ("coco/train2017", "coco/annotations/instances_train2017_sample@1000.json")


_PREDEFINED_SPLITS_CITYSCAPES = {}
_PREDEFINED_SPLITS_CITYSCAPES["cityscapes"] = {}
seeds = [1]
percentages = [5, 10, 20, 30, 40]
for seed in seeds:
    for perc in percentages:
        _PREDEFINED_SPLITS_CITYSCAPES["cityscapes"][f"cityscapes_fine_instance_seg_train.{seed}@{perc}"] = ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train", f"cityscapes/annotations/instancesonly_filtered_gtFine_train.{seed}@{perc}.json")
        _PREDEFINED_SPLITS_CITYSCAPES["cityscapes"][f"cityscapes_fine_instance_seg_train.{seed}@{perc}-unlabel"] = ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train", f"cityscapes/annotations/instancesonly_filtered_gtFine_train.{seed}@{perc}-unlabeled.json")
        _PREDEFINED_SPLITS_CITYSCAPES["cityscapes"][f"cityscapes_fine_instance_seg_train_{perc}sup"] = ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train", f"cityscapes/annotations/instancesonly_filtered_gtFine_train_{perc}sup.json")
        _PREDEFINED_SPLITS_CITYSCAPES["cityscapes"][f"cityscapes_fine_instance_seg_train_{perc}sup-unlabel"] = ("cityscapes/leftImg8bit/train", "cityscapes/gtFine/train", f"cityscapes/annotations/instancesonly_filtered_gtFine_train_{perc}sup-unlabeled.json")


def register_coco_unlabel(root):
    for _, splits_per_dataset in _SPLITS_COCO_FORMAT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_unlabel_instances(
                key, meta, os.path.join(root, json_file), os.path.join(root, image_root)
            )


def register_coco_unlabel_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_unlabel_json(json_file, image_root, name)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def load_coco_unlabel_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    id_map = None
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    imgs = coco_api.loadImgs(img_ids)

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs), json_file))

    dataset_dicts = []

    for img_dict in imgs:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        dataset_dicts.append(record)

    return dataset_dicts


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            meta = {}
            register_coco_instances(
                key,
                meta,
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


def get_cityscapes_files(image_dir, gt_dir, json_info):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
    image_dict = {}
    suffix = "leftImg8bit.png"
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        for basename in PathManager.ls(city_img_dir):
            image_file = os.path.join(city_img_dir, basename)

            assert basename.endswith(suffix), basename
            basename = os.path.basename(basename)[: -len(suffix)]

            instance_file = os.path.join(city_gt_dir, basename + "gtFine_instanceIds.png")
            label_file = os.path.join(city_gt_dir, basename + "gtFine_labelIds.png")
            json_file = os.path.join(city_gt_dir, basename + "gtFine_polygons.json")
            image_dict[basename] = (image_file, instance_file, label_file, json_file)

    for ann in json_info["images"]:
        image_file = image_dict.get(os.path.basename(ann["file_name"])[: -len(suffix)], None)
        assert image_dir is not None
        files.append(image_file)

    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def _cityscapes_files_to_dict(files, from_json, to_polygons):
    """
    Parse cityscapes annotation files to a instance segmentation dataset dict.

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        A dict in Detectron2 Dataset format.
    """
    from cityscapesscripts.helpers.labels import id2label, name2label

    image_file, instance_id_file, _, json_file = files

    annos = []

    if from_json:
        from shapely.geometry import MultiPolygon, Polygon

        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)
        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": jsonobj["imgHeight"],
            "width": jsonobj["imgWidth"],
        }

        # `polygons_union` contains the union of all valid polygons.
        polygons_union = Polygon()

        # CityscapesScripts draw the polygons in sequential order
        # and each polygon *overwrites* existing ones. See
        # (https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2instanceImg.py) # noqa
        # We use reverse order, and each polygon *avoids* early ones.
        # This will resolve the ploygon overlaps in the same way as CityscapesScripts.
        for obj in jsonobj["objects"][::-1]:
            if "deleted" in obj:  # cityscapes data format specific
                continue
            label_name = obj["label"]

            try:
                label = name2label[label_name]
            except KeyError:
                if label_name.endswith("group"):  # crowd area
                    label = name2label[label_name[: -len("group")]]
                else:
                    raise
            if label.id < 0:  # cityscapes data format
                continue

            # Cityscapes's raw annotations uses integer coordinates
            # Therefore +0.5 here
            poly_coord = np.asarray(obj["polygon"], dtype="f4") + 0.5
            # CityscapesScript uses PIL.ImageDraw.polygon to rasterize
            # polygons for evaluation. This function operates in integer space
            # and draws each pixel whose center falls into the polygon.
            # Therefore it draws a polygon which is 0.5 "fatter" in expectation.
            # We therefore dilate the input polygon by 0.5 as our input.
            poly = Polygon(poly_coord).buffer(0.5, resolution=4)

            if not label.hasInstances or label.ignoreInEval:
                # even if we won't store the polygon it still contributes to overlaps resolution
                polygons_union = polygons_union.union(poly)
                continue

            # Take non-overlapping part of the polygon
            poly_wo_overlaps = poly.difference(polygons_union)
            if poly_wo_overlaps.is_empty:
                continue
            polygons_union = polygons_union.union(poly)

            anno = {}
            anno["iscrowd"] = label_name.endswith("group")
            anno["category_id"] = label.id

            if isinstance(poly_wo_overlaps, Polygon):
                poly_list = [poly_wo_overlaps]
            elif isinstance(poly_wo_overlaps, MultiPolygon):
                poly_list = poly_wo_overlaps.geoms
            else:
                raise NotImplementedError("Unknown geometric structure {}".format(poly_wo_overlaps))

            poly_coord = []
            for poly_el in poly_list:
                # COCO API can work only with exterior boundaries now, hence we store only them.
                # TODO: store both exterior and interior boundaries once other parts of the
                # codebase support holes in polygons.
                poly_coord.append(list(chain(*poly_el.exterior.coords)))
            anno["segmentation"] = poly_coord
            (xmin, ymin, xmax, ymax) = poly_wo_overlaps.bounds

            anno["bbox"] = (xmin, ymin, xmax, ymax)
            anno["bbox_mode"] = BoxMode.XYXY_ABS

            annos.append(anno)
    else:
        # See also the official annotation parsing scripts at
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/instances2dict.py  # noqa
        with PathManager.open(instance_id_file, "rb") as f:
            inst_image = np.asarray(Image.open(f), order="F")
        # ids < 24 are stuff labels (filtering them first is about 5% faster)
        flattened_ids = np.unique(inst_image[inst_image >= 24])

        ret = {
            "file_name": image_file,
            "image_id": os.path.basename(image_file),
            "height": inst_image.shape[0],
            "width": inst_image.shape[1],
        }

        for instance_id in flattened_ids:
            # For non-crowd annotations, instance_id // 1000 is the label_id
            # Crowd annotations have <1000 instance ids
            label_id = instance_id // 1000 if instance_id >= 1000 else instance_id
            label = id2label[label_id]
            if not label.hasInstances or label.ignoreInEval:
                continue

            anno = {}
            anno["iscrowd"] = instance_id < 1000
            anno["category_id"] = label.id

            mask = np.asarray(inst_image == instance_id, dtype=np.uint8, order="F")

            inds = np.nonzero(mask)
            ymin, ymax = inds[0].min(), inds[0].max()
            xmin, xmax = inds[1].min(), inds[1].max()
            anno["bbox"] = (xmin, ymin, xmax, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue
            anno["bbox_mode"] = BoxMode.XYXY_ABS
            if to_polygons:
                # This conversion comes from D4809743 and D5171122,
                # when Mask-RCNN was first developed.
                contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
                    -2
                ]
                polygons = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]
                # opencv's can produce invalid polygons
                if len(polygons) == 0:
                    continue
                anno["segmentation"] = polygons
            else:
                anno["segmentation"] = mask_util.encode(mask[:, :, None])[0]
            annos.append(anno)
    ret["annotations"] = annos
    return ret


def load_cityscapes(image_dir, gt_dir, gt_json, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g.,
            "~/cityscapes/gtFine/train".
        gt_json (str): path to the json file. e.g.,
            "~/cityscapes/annotations/instancesonly_filtered_gtFine_train.json".
        meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id"
            and "stuff_dataset_id_to_contiguous_id" to map category ids to
            contiguous ids for training.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    # print('Looking for json ground truth', gt_json)
    assert os.path.exists(
        gt_json
    )
    with open(gt_json) as f:
        json_info = json.load(f)
    files = get_cityscapes_files(image_dir, gt_dir, json_info)
    logger.info("Preprocessing cityscapes annotations ...")
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))

    ret = pool.map(
        functools.partial(_cityscapes_files_to_dict, from_json=True, to_polygons=True),
        files,
    )
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))

    # Map cityscape ids to contiguous ids
    from cityscapesscripts.helpers.labels import labels

    labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
    for dict_per_image in ret:
        for anno in dict_per_image["annotations"]:
            anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]
    return ret


from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
def register_all_cityscapes(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_CITYSCAPES.items():
        for key, (image_dir, gt_dir, gt_json) in splits_per_dataset.items():
            meta = _get_builtin_metadata("cityscapes")
            image_dir = os.path.join(root, image_dir)
            gt_dir = os.path.join(root, gt_dir)
            gt_json = os.path.join(root, gt_json)

            DatasetCatalog.register(
                key,
                lambda x=image_dir, y=gt_dir, z=gt_json: load_cityscapes(x, y, z, meta),
            )
            MetadataCatalog.get(key).set(
                image_dir=image_dir, gt_dir=gt_dir, gt_json=gt_json, evaluator_type="", **meta,
            )



_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco_unlabel(_root)
register_all_coco(_root)
register_all_cityscapes(_root)