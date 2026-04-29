# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import re
import time
import logging
from sympy import rad
import torch
from torch.nn.parallel import DistributedDataParallel, DataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator, DatasetEvaluators
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures import Boxes, PolygonMasks, ROIMasks, BitMasks
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import build_detection_test_loader

from ..data.build import (
    build_detection_semisup_train_loader,
)
from ..modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ..checkpoint.detection_checkpoint import DetectionTSCheckpointer
from detectron2.utils.events import get_event_storage
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from detectron2.layers.roi_align import ROIAlign

# Supervised-only Trainer
class BaselineTrainer(SimpleTrainer):
    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        
        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler and self.amp:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler and self.amp:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        """
        If you want to do something with the losses, you can wrap the model.
        """
        with autocast(enabled=self.amp):
            record_dict = self.model(data)
            if isinstance(record_dict, torch.Tensor):
                losses = record_dict
                loss_dict = {"total_loss": record_dict}
            else:
                loss_dict = {}
                for key in record_dict.keys():
                    if key[:4] == "loss" and key[-3:] != "val":
                        loss_dict[key] = record_dict[key]
                losses = sum(loss_dict.values())

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox
        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        # losses.requires_grad_(True)
        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        self._write_metrics(metrics_dict)

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)
            if "process_pseudo_time" in all_metrics_dict[0]:
                process_pseudo_time = np.max([x.pop("process_pseudo_time")
                                    for x in all_metrics_dict])
                self.storage.put_scalar("process_pseudo_time", process_pseudo_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


class PL_DC_TeacherTrainer(SimpleTrainer):
    def __init__(
        self,
        model_student,
        model_teacher,
        dataloader_label,
        dataloader_unlabel,
        optimizer,
        burn_up_step=2000,
        teacher_update_iter=1,
        ema_keep_rate=0.9996,
        mask_quality_threshold=0.9,
        class_quality_threshold=0.7,
        mask_area_threshold=5,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
        output_dir='output',
        CLIP_model=None,
        class_name=None,
        init_class_thresholds=None,  # dict with 'class' and 'mask' per-class tensors; None = fixed threshold mode
    ):
        TrainerBase.__init__(self)
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model_student, DistributedDataParallel):
            assert not (model_student.device_ids and len(model_student.device_ids) > 1), unsupported
        assert not isinstance(model_student, DataParallel), unsupported

        self.output_dir = output_dir
        self.model = model_student
        self.model.train()
        self.model_teacher = model_teacher
        self.model_teacher.eval()
        self.CLIP_model = CLIP_model
        self.class_name = class_name
        self.dataloader_label = dataloader_label
        self.dataloader_unlabel = dataloader_unlabel
        self._data_loader_iter_obj = None
        self._data_loader_unl_iter_obj = None
        self.optimizer = optimizer
        self.burn_up_step = burn_up_step
        self.teacher_update_iter = teacher_update_iter
        self.ema_keep_rate = ema_keep_rate
        self.mask_quality_threshold = mask_quality_threshold
        self.class_quality_threshold = class_quality_threshold
        self.mask_area_threshold = mask_area_threshold

        # per-class adaptive thresholds (DF-ACAT); None means fixed-threshold mode
        if init_class_thresholds is not None:
            self.class_thresholds = init_class_thresholds['class'].cuda()
            self.mask_thresholds = init_class_thresholds['mask'].cuda()
        else:
            self.class_thresholds = None
            self.mask_thresholds = None

        if self.CLIP_model is not None:
            text_classifier = []
            self.num_arrtributes = [len(c) for c in self.class_name]
            class_name = []
            for i, c in enumerate(self.class_name):
                for cc in c:
                    # use description as-is if it's a full sentence, else wrap with template
                    if cc.endswith('.') and ' ' in cc:
                        class_name += [f"a photo of {cc}"]
                    else:
                        class_name += [f"a photo of a {cc}."]
            # this is needed to avoid oom, which may happen when num of class is large
            bs = 64
            for idx in range(0, len(class_name), bs):
                text_classifier.append(self.CLIP_model.get_text_classifier(class_name[idx:idx+bs], self.model.device).detach())
            text_classifier = torch.cat(text_classifier, dim=0)

            # average across templates and normalization.
            text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
            self.train_text_classifier = text_classifier

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        
        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler and self.amp:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler and self.amp:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])

    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self.dataloader_label)
        return self._data_loader_iter_obj

    @property
    def _data_loader_unl_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_unl_iter_obj is None:
            self._data_loader_unl_iter_obj = iter(self.dataloader_unlabel)
        return self._data_loader_unl_iter_obj

    def reset_data_loader(self, data_loader_builder):
        """
        Delete and replace the current data loader with a new one, which will be created
        by calling `data_loader_builder` (without argument).
        """
        del self.dataloader_label
        train_loader_label, train_loader_unlabel = data_loader_builder()
        self.dataloader_label = train_loader_label
        self.dataloader_unlabel = train_loader_unlabel
        self._data_loader_iter_obj = None
        self._data_loader_unl_iter_obj = None

    # # =====================================================
    # # ================== Pseduo-labeling ==================
    # # =====================================================
    @torch.no_grad()
    def threshold_pseudo_label(self, data, inst):
        pred_classes = inst['instances'].pred_classes
        mask_quality = inst['instances'].mask_quality
        class_quality = inst['instances'].class_quality

        if self.class_thresholds is not None:
            # adaptive per-class thresholds (DF-ACAT)
            cls_thr = self.class_thresholds[pred_classes]
            msk_thr = self.mask_thresholds[pred_classes]
        else:
            cls_thr = self.class_quality_threshold
            msk_thr = self.mask_quality_threshold

        valid_map = (mask_quality >= msk_thr) & \
                    (class_quality >= cls_thr) & \
                    (inst['instances'].pred_masks.sum(dim=(1,2)) > self.mask_area_threshold)

        # EMA update of per-class thresholds using scores of all predictions (not just valid)
        if self.class_thresholds is not None:
            for k in range(self.class_thresholds.shape[0]):
                cls_mask = pred_classes == k
                if cls_mask.any():
                    self.class_thresholds[k] = self.ema_keep_rate * self.class_thresholds[k] + \
                        (1 - self.ema_keep_rate) * class_quality[cls_mask].mean()
                    self.mask_thresholds[k] = self.ema_keep_rate * self.mask_thresholds[k] + \
                        (1 - self.ema_keep_rate) * mask_quality[cls_mask].mean()

        image_shape = inst['instances'].image_size
        new_proposal_inst = Instances(image_shape)

        new_bbox_loc = inst['instances'].pred_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.gt_classes = pred_classes[valid_map]
        new_proposal_inst.mask_quality = mask_quality[valid_map]
        new_proposal_inst.class_quality = class_quality[valid_map]
        new_proposal_inst.scores = inst['instances'].scores[valid_map]
        new_proposal_inst.mask_uncertainty = inst['instances'].mask_uncertainty[valid_map]
        new_proposal_inst.gt_masks = inst['instances'].pred_masks[valid_map]

        if self.CLIP_model is not None and len(new_proposal_inst.gt_masks) > 0:
            cls_logit = inst['instances'].cls_logit[valid_map]
            cls_logit = cls_logit[..., :-1]  # remove the background class
            satisfied_scores, satisfied_class = self.refine_pseudo_class(
                data,
                inst['instances'].mask_logits[valid_map],
                cls_logit,
                new_proposal_inst.class_quality.clone(),
                new_proposal_inst.gt_classes.clone()
            )
            new_proposal_inst.class_quality_old = new_proposal_inst.class_quality
            new_proposal_inst.class_quality = satisfied_scores
            new_proposal_inst.gt_classes_old = new_proposal_inst.gt_classes
            new_proposal_inst.gt_classes = satisfied_class
            new_proposal_inst.scores_old = inst['instances'].scores[valid_map]
            new_proposal_inst.scores = satisfied_scores * new_proposal_inst.mask_quality

        return new_proposal_inst

    def process_pseudo_label(
        self, unlabel_data_weak, teacher_preds
    ):
        list_instances = []
        num_inst_output = 0.0
        for i,(data, inst) in enumerate(zip(unlabel_data_weak, teacher_preds)):
            # thresholding
            r = self.threshold_pseudo_label(data, inst)
            num_inst_output += len(r)
            list_instances.append(r)        
        num_inst_output = num_inst_output / len(teacher_preds)
        return list_instances, num_inst_output
    
    @torch.no_grad()
    def refine_pseudo_class(self, data, mask_logits, cls_logit=None, satisfied_scores=None, satisfied_class=None):
        if satisfied_class.shape[0] == 0:
            return satisfied_scores, satisfied_class
        image_path = data["file_name"]
        image = data["image"].float().unsqueeze(0).to('cuda')
        clip_cls_logit = self.CLIP_model.get_classification_logits_single(image, mask_logits, self.train_text_classifier, self.num_arrtributes)

        def custom_decay_function(x, a, b):
            # 使用余弦函数生成从0.5到0的下降趋势
            y = 0.25 * (np.cos(np.pi * (x - a) / (b - a)) + 1)
            return y

        clip_cls_probs = clip_cls_logit.squeeze(0).softmax(-1)
        cls_probs = cls_logit.softmax(-1)
        # alpha, beta = 0.5, 0.5
        # cls_logits = (cls_probs ** (1 - alpha) * clip_cls_probs**alpha).log()
        # clip_cls_logits = (cls_probs ** (1 - beta) * clip_cls_probs**beta).log()
    
        # similarity = cls_logits + clip_cls_logits
        # similarity = similarity.softmax(dim=-1)
        clip_weight = custom_decay_function(self.iter, self.burn_up_step, self.max_iter)
        similarity = clip_weight * clip_cls_probs + (1 - clip_weight) * cls_probs
        classes_scores, classes_indices = torch.max(similarity, dim=-1)
        classes_scores = classes_scores.to(dtype=satisfied_scores.dtype)
        classes_indices = classes_indices.to(dtype=satisfied_class.dtype)

        return classes_scores, classes_indices

    @torch.no_grad()
    def visualize_pseudo_label(self, batched_inputs, prefix = 'label', vis_box=False):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        return
        storage = get_event_storage()
        if self.iter % 5000 != 0 :
            return
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

        def _create_text_labels(classes, scores, class_names, is_crowd=None):
            """
            Args:
                classes (list[int] or None):
                scores (list[float] or None):
                class_names (list[str] or None):
                is_crowd (list[bool] or None):
            Returns:
                list[str] or None
            """
            labels = None
            if classes is not None:
                if class_names is not None and len(class_names) > 0:
                    labels = [class_names[i] for i in classes]
                else:
                    labels = [str(i) for i in classes]
            if scores is not None:
                if labels is None:
                    labels = ["{:.0f}%".format(s * 100) for s in scores]
                else:
                    labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
            if labels is not None and is_crowd is not None:
                labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
            return labels

        for b,input in enumerate(batched_inputs):
            device_index = input["instances"].gt_boxes.tensor.device.index
            img = input["image"].cpu()
            img = img.permute(1, 2, 0)[:,:,[2,1,0]]
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(
                boxes=input["instances"].gt_boxes.tensor.cpu() if vis_box else None,
                masks=input["instances"].gt_masks.cpu() if hasattr(input["instances"],"gt_masks") else None, 
                labels=_create_text_labels(input["instances"].gt_classes.cpu().numpy(),
                                    input["instances"].scores.cpu() if hasattr(input["instances"],'scores') else None,
                                    _get_builtin_metadata('coco')['thing_classes'] if 'coco' in self.output_dir else _get_builtin_metadata('cityscapes')['thing_classes'],
                )
            )
            anno_img = v_gt.get_image()
            
            vis_img = anno_img
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = prefix+f"_i{self.iter}_g{device_index}_b{b}"
            storage.put_image(prefix+f"_b{b}", vis_img)
            plt.imsave(fname=os.path.join(self.output_dir,vis_name+'.jpg'),
                        arr=vis_img.transpose(1,2,0), 
                        format='jpg')

            if hasattr(input["instances"],'mask_uncertainty'):
                mask_uncertainty = input["instances"].mask_uncertainty
                mask_uncertainty = mask_uncertainty.mean(0)
                vis_img = torch.stack([mask_uncertainty] * 3).cpu().numpy()*255
                vis_img = vis_img.astype("uint8")
                vis_name = prefix+f"_i{self.iter}_g{device_index}_b{b}_uncertainty"
                storage.put_image(prefix+f"_b{b}_uncertainty", vis_img)
                plt.imsave(fname=os.path.join(self.output_dir,vis_name+'.jpg'),
                            arr=vis_img.transpose(1,2,0), 
                            format='jpg')
            if hasattr(input["instances"],'gt_classes_old'):
                v_gt_ori = Visualizer(img, None)
                v_gt_ori = v_gt_ori.overlay_instances(
                    masks=input["instances"].gt_masks.cpu(),
                    labels=_create_text_labels(input["instances"].gt_classes_old.cpu().numpy(),
                                    input["instances"].class_quality_old.cpu(),
                                    _get_builtin_metadata('coco')['thing_classes'] if 'coco' in self.output_dir else _get_builtin_metadata('cityscapes')['thing_classes'],
                    )
                )
                anno_img = v_gt_ori.get_image()
                v_gt = Visualizer(img, None)
                v_gt = v_gt.overlay_instances(
                    masks=input["instances"].gt_masks.cpu(),
                    labels=_create_text_labels(input["instances"].gt_classes.cpu().numpy(),
                                    input["instances"].class_quality.cpu(),
                                    _get_builtin_metadata('coco')['thing_classes'] if 'coco' in self.output_dir else _get_builtin_metadata('cityscapes')['thing_classes'],
                    )
                )
                vis_img = np.concatenate((anno_img, v_gt.get_image()), axis=1)
                vis_img = vis_img.transpose(2, 0, 1)
                vis_name = prefix+"_refine"+f"_i{self.iter}_g{device_index}_b{b}"
                storage.put_image(vis_name, vis_img)
                plt.imsave(fname=os.path.join(self.output_dir,vis_name+'.png'),
                            arr=vis_img.transpose(1,2,0),
                            format='png')
            break  # only visualize one image in a batch

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabeld_data, label):
        for unlabel_datum, lab_inst in zip(unlabeld_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabeld_data

    # # =====================================================
    # # =================== Training Flow ===================
    # # =====================================================

    def run_step(self):
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data_weak, data_strong = next(self._data_loader_iter)
        unlabel_data_weak, unlabel_data_strong = next(self._data_loader_unl_iter)
        data_time = time.perf_counter() - start
        process_pseudo_time = None
        """
        If you want to do something with the losses, you can wrap the model.
        """
        self.visualize_pseudo_label(unlabel_data_weak, 'gt_weak')
        # self.visualize_pseudo_label(unlabel_data_strong, 'gt_strong')
        # remove unlabeled data labels
        unlabel_data_weak = self.remove_label(unlabel_data_weak)
        unlabel_data_strong = self.remove_label(unlabel_data_strong)

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.burn_up_step:
            # input both strong and weak supervised data into model
            data_weak.extend(data_strong)
            with autocast(enabled=self.amp):
                record_dict = self.model(data_weak)
                if isinstance(record_dict, torch.Tensor):
                    losses = record_dict
                    loss_dict = {"total_loss": record_dict}
                else:
                    loss_dict = {}
                    for key in record_dict.keys():
                        if key[:4] == "loss" and key[-3:] != "val":
                            loss_dict[key] = record_dict[key]
                    losses = sum(loss_dict.values())
        else:
            if self.iter == self.burn_up_step:
                # update copy the the whole model
                self._update_teacher_model_ema(keep_rate=0.00)

            elif (
                self.iter - self.burn_up_step
            ) % self.teacher_update_iter == 0:
                self._update_teacher_model_ema(keep_rate=self.ema_keep_rate)

            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            with torch.no_grad():
                teacher_preds = self.model_teacher(unlabel_data_weak, return_pseudo=True)

            pseudo_start = time.perf_counter()
            pesudo_inst, _ = self.process_pseudo_label(unlabel_data_weak, teacher_preds)
            process_pseudo_time = time.perf_counter() - pseudo_start

            #  add pseudo-label to unlabeled data
            unlabel_data_strong = self.add_label(
                unlabel_data_strong, pesudo_inst
            )
            unlabel_data_weak = self.add_label(
                unlabel_data_weak, pesudo_inst
            )

            all_label_data = data_weak + data_strong
            all_unlabel_data = unlabel_data_strong

            self.visualize_pseudo_label(unlabel_data_weak, 'pgt_weak')

            with autocast(enabled=self.amp):
                record_all_label_data = self.model(all_label_data)
                record_dict.update(record_all_label_data)
                record_all_unlabel_data = self.model(all_unlabel_data, ssl=True)
                new_record_all_unlabel_data = {}
                for key in record_all_unlabel_data.keys():
                    new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[key]
                record_dict.update(new_record_all_unlabel_data)
                if isinstance(record_dict, torch.Tensor):
                    losses = record_dict
                    loss_dict = {"total_loss": record_dict}
                else:
                    # weight losses
                    loss_dict = {}
                    for key in record_dict.keys():
                        if key[:4] == "loss":
                            if key[-6:] == "pseudo":  # unsupervised loss
                                if "loss_ce" in key:
                                    loss_dict[key] = (
                                        record_dict[key] * 1
                                    )
                                else:
                                    loss_dict[key] = (
                                        record_dict[key] * 1
                                    )
                            else:  # supervised loss
                                loss_dict[key] = record_dict[key] * 1
                            record_dict[key] = loss_dict[key]
                    losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        if process_pseudo_time is not None:
            metrics_dict["process_pseudo_time"] = process_pseudo_time 
        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        # losses.requires_grad_(True)
        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        self._write_metrics(metrics_dict)

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)
            if "process_pseudo_time" in all_metrics_dict[0]:
                process_pseudo_time = np.max([x.pop("process_pseudo_time")
                                    for x in all_metrics_dict])
                self.storage.put_scalar("process_pseudo_time", process_pseudo_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model_ema(self, keep_rate=0.9996):
        # if comm.get_world_size() > 1:
        #     student_model_dict = {
        #         key[7:]: value for key, value in self.model.state_dict().items()
        #     }
        # else:
        student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

