# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from math import e
import numpy as np
import operator
import json
import torch.utils.data
import torch.utils.data as torchdata
from detectron2.utils.comm import get_world_size
from detectron2.data.common import (
    DatasetFromList,
    MapDataset,
    ToIterableDataset,
    AspectRatioGroupedDataset,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.data.build import (
    trivial_batch_collator,
    worker_init_reset_seed,
)
from .common import AspectRatioGroupedDatasetTwoCrop

"""
This file contains the default logic to build a dataloader for training or testing.
"""


def build_detection_semisup_train_loader(
    dataset_label,
    dataset_unlabel,
    *,
    mapper_label,
    mapper_unlabel,
    sampler_label=None,
    sampler_unlabel=None,
    total_batch_size_label,
    total_batch_size_unlabel,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
):
    if isinstance(dataset_label, list):
        dataset_label = DatasetFromList(dataset_label, copy=False)
    if isinstance(dataset_unlabel, list):
        dataset_unlabel = DatasetFromList(dataset_unlabel, copy=False)
    if mapper_label is not None:
        dataset_label = MapDataset(dataset_label, mapper_label)
    if mapper_unlabel is not None:
        dataset_unlabel = MapDataset(dataset_unlabel, mapper_unlabel)

    if isinstance(dataset_label, torchdata.IterableDataset):
        assert sampler_label is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler_label is None:
            sampler_label = TrainingSampler(len(dataset_label))
        assert isinstance(sampler_label, torchdata.Sampler), f"Expect a Sampler but got {type(sampler_label)}"

    if isinstance(dataset_unlabel, torchdata.IterableDataset):
        assert sampler_unlabel is None, "sampler must be None if dataset is IterableDataset"
    else:    
        if sampler_unlabel is None:
            sampler_unlabel = TrainingSampler(len(dataset_unlabel))
        assert isinstance(sampler_unlabel, torchdata.Sampler), f"Expect a Sampler but got {type(sampler_unlabel)}"

    return build_semisup_batch_data_loader(
        dataset_label, dataset_unlabel,
        sampler_label, sampler_unlabel,
        total_batch_size_label, total_batch_size_unlabel,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


# batch data loader
def build_semisup_batch_data_loader(
    dataset_label,
    dataset_unlabel,
    sampler_label,
    sampler_unlabel,
    total_batch_size_label,
    total_batch_size_unlabel,
    *,
    aspect_ratio_grouping=False,
    num_workers=0,
    collate_fn=None,
):
    world_size = get_world_size()
    assert (
        total_batch_size_label > 0 and total_batch_size_label % world_size == 0
    ), "Total label batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    assert (
        total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    batch_size_label = total_batch_size_label // world_size
    batch_size_unlabel = total_batch_size_unlabel // world_size

    if isinstance(dataset_label, torchdata.IterableDataset):
        assert sampler_label is None, "sampler must be None if dataset is IterableDataset"
    else:
        dataset_label = ToIterableDataset(dataset_label, sampler_label)
    if isinstance(dataset_unlabel, torchdata.IterableDataset):
        assert sampler_unlabel is None, "sampler must be None if dataset is IterableDataset"
    else:
        dataset_unlabel = ToIterableDataset(dataset_unlabel, sampler_unlabel)

    if aspect_ratio_grouping:
        label_data_loader = torchdata.DataLoader(
            dataset_label,
            num_workers=num_workers,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        unlabel_data_loader = torchdata.DataLoader(
            dataset_unlabel,
            num_workers=num_workers,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        label_data_loader = AspectRatioGroupedDatasetTwoCrop(label_data_loader, batch_size_label)
        unlabel_data_loader = AspectRatioGroupedDatasetTwoCrop(unlabel_data_loader, batch_size_unlabel)
        if collate_fn is None:
            return label_data_loader, unlabel_data_loader
        return MapDataset(label_data_loader, collate_fn), MapDataset(unlabel_data_loader, collate_fn)
    else:
        return torchdata.DataLoader(
            dataset_label,
            batch_size=batch_size_label,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
            worker_init_fn=worker_init_reset_seed,
        ),torchdata.DataLoader(
            dataset_unlabel,
            batch_size=batch_size_unlabel,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
            worker_init_fn=worker_init_reset_seed,
        )
