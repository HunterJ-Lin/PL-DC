import torch.nn as nn
from detrex.layers import PositionEmbeddingSine
from detrex.modeling.backbone import ResNet, BasicStem

from detectron2.config import LazyCall as L

from pl_dc.modeling.meta_arch.mask2former import MaskFormer
from mask2former.modeling.meta_arch.mask_former_head import MaskFormerHead
from mask2former.modeling.pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from pl_dc.modeling.mask2former_criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, ShapeSpec, get_norm


dim=256
n_class=80
dec_layers = 10
input_shape={'res2': ShapeSpec(channels=256, height=None, width=None, stride=4), 'res3': ShapeSpec(channels=512, height=None, width=None, stride=8), 'res4': ShapeSpec(channels=1024, height=None, width=None, stride=16), 'res5': ShapeSpec(channels=2048, height=None, width=None, stride=32)}
model_student = L(MaskFormer)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res2", "res3", "res4", "res5"],
        freeze_at=5,
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
            norm = 'GN',
            transformer_in_features=['res3', 'res4', 'res5'],
            common_stride=4,
        ),
        loss_weight= 1.0,
        ignore_value= 255,
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
            mask_dim=dim,
            enforce_input_project=False,
        ),
        transformer_in_feature='multi_scale_pixel_decoder',
    ),
    criterion=L(SetCriterion)(
        num_classes="${..sem_seg_head.num_classes}",
        matcher=L(HungarianMatcher)(
            cost_class = 2.0,
            cost_mask = 5.0,
            cost_dice = 5.0,
            num_points = 12544,
        ),
        weight_dict=dict(),
        eos_coef=0.1,
        losses=['labels', 'masks'],
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    ),
    num_queries=100,
    object_mask_threshold=0.8,
    overlap_threshold=0.8,
    metadata=MetadataCatalog.get('coco_2017_train'),
    size_divisibility=32,
    sem_seg_postprocess_before_inference=True,
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
    # inference
    semantic_on=False,
    panoptic_on=False,
    instance_on=True,
    test_topk_per_image=100,
)
model_teacher = L(MaskFormer)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res2", "res3", "res4", "res5"],
        freeze_at=5,
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
            norm = 'GN',
            transformer_in_features=['res3', 'res4', 'res5'],
            common_stride=4,
        ),
        loss_weight= 1.0,
        ignore_value= 255,
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
            mask_dim=dim,
            enforce_input_project=False,
        ),
        transformer_in_feature='multi_scale_pixel_decoder',
    ),
    criterion=L(SetCriterion)(
        num_classes="${..sem_seg_head.num_classes}",
        matcher=L(HungarianMatcher)(
            cost_class = 2.0,
            cost_mask = 5.0,
            cost_dice = 5.0,
            num_points = 12544,
        ),
        weight_dict=dict(),
        eos_coef=0.1,
        losses=['labels', 'masks'],
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    ),
    num_queries=100,
    object_mask_threshold=0.8,
    overlap_threshold=0.8,
    metadata=MetadataCatalog.get('coco_2017_train'),
    size_divisibility=32,
    sem_seg_postprocess_before_inference=True,
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
    # inference
    semantic_on=False,
    panoptic_on=False,
    instance_on=True,
    test_topk_per_image=100,
)


# set aux loss weight dict
class_weight=2.0
mask_weight=5.0
dice_weight=5.0
weight_dict = {"loss_ce": class_weight}
weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})

aux_weight_dict = {}
for i in range(dec_layers):
    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
weight_dict.update(aux_weight_dict)
model_student.criterion.weight_dict=weight_dict
model_teacher.criterion.weight_dict=weight_dict