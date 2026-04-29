from ast import Num
import torch.nn as nn

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from mask2former.maskformer_model import MaskFormer
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead
from mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher

dim = 256
n_class = 80
dec_layers = 9
input_shape={'res2': ShapeSpec(channels=256, height=None, width=None, stride=4), 'res3': ShapeSpec(channels=512, height=None, width=None, stride=8), 'res4': ShapeSpec(channels=1024, height=None, width=None, stride=16), 'res5': ShapeSpec(channels=2048, height=None, width=None, stride=32)}
model = L(MaskFormer)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res2", "res3", "res4", "res5"],
        freeze_at=0,
    ),
    sem_seg_head=L(MaskFormerHead)(
        input_shape=input_shape,
        num_classes=n_class,
        pixel_decoder=L(MSDeformAttnPixelDecoder)(
            input_shape=input_shape,
            transformer_dropout=0.0,
            transformer_nheads=8,
            transformer_dim_feedforward=1024,
            transformer_enc_layers=6,
            conv_dim=dim,
            mask_dim=dim,
            norm="GN",
            transformer_in_features=["res3", "res4", "res5"],
            common_stride=4,
        ),
        loss_weight=1.0,
        ignore_value=255,
        transformer_predictor=L(MultiScaleMaskedTransformerDecoder)(
            in_channels=dim,
            mask_classification=True,
            num_classes="${..num_classes}",
            hidden_dim=dim,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=dec_layers,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False,
        ),
        transformer_in_feature="multi_scale_pixel_decoder",
    ),
    criterion=L(SetCriterion)(
        num_classes="${..sem_seg_head.num_classes}",
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_mask=5.0,
            cost_dice=5.0,
            num_points=12544,
        ),
        weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0},
        eos_coef=0.1,
        losses=["labels", "masks"],
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    ),
    num_queries=100,
    object_mask_threshold=0.8,
    overlap_threshold=0.8,
    metadata=None,
    size_divisibility=32,
    sem_seg_postprocess_before_inference=True,
    # inference
    semantic_on=False,
    panoptic_on=False,
    instance_on=True,
    test_topk_per_image=100,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
)


weight_dict = model.criterion.weight_dict
aux_weight_dict = {}
for i in range(model.sem_seg_head.transformer_predictor.dec_layers - 1):
    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
weight_dict.update(aux_weight_dict)
model.criterion.weight_dict = weight_dict