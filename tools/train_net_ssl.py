#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import time
import numpy as np
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.events import (
    CommonMetricPrinter, 
    JSONWriter, 
    TensorboardXWriter
)
from detectron2.checkpoint import DetectionCheckpointer
# from detrex.checkpoint import DetectionCheckpointer

from detrex.utils import WandbWriter
from detrex.modeling import ema

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from pl_dc.engine.trainer import PL_DC_TeacherTrainer
from pl_dc.data import datasets
from pl_dc.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from pl_dc.checkpoint.detection_checkpoint import DetectionTSCheckpointer

class Trainer(PL_DC_TeacherTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

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
        init_class_thresholds=None,
    ):
        super().__init__(
            model_student=model_student, model_teacher=model_teacher,
            dataloader_label=dataloader_label, dataloader_unlabel=dataloader_unlabel,
            optimizer=optimizer, burn_up_step=burn_up_step,
            teacher_update_iter=teacher_update_iter, amp=amp,
            ema_keep_rate=ema_keep_rate, mask_quality_threshold=mask_quality_threshold,
            class_quality_threshold=class_quality_threshold, mask_area_threshold=mask_area_threshold,
            clip_grad_params=clip_grad_params, grad_scaler=grad_scaler, output_dir=output_dir,
            CLIP_model=CLIP_model, class_name=class_name,
            init_class_thresholds=init_class_thresholds,
        )


def do_test(cfg, model, eval_only=False, teacher=False):
    logger = logging.getLogger("detectron2")

    if eval_only:
        logger.info("Run evaluation under eval-only mode")
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            logger.info("Run evaluation with EMA.")
        else:
            logger.info("Run evaluation without EMA.")
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
            )
            print_csv_format(ret)
    else:
        logger.info("Run evaluation without EMA.")
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
            )
            print_csv_format(ret)

            if cfg.train.model_ema.enabled:
                logger.info("Run evaluation with EMA.")
                with ema.apply_model_ema_and_restore(model):
                    if "evaluator" in cfg.dataloader:
                        ema_ret = inference_on_dataset(
                            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
                        )
                        print_csv_format(ema_ret)
                        ret.update(ema_ret)
                    
    if teacher:
        _last_eval_results_teacher = {
            k: ret[k]
            for k in ret.keys()
        }
        return _last_eval_results_teacher
    else:
        _last_eval_results_student = {
            k + "_student": ret[k]
            for k in ret.keys()
        }
        return _last_eval_results_student 

def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model_student = instantiate(cfg.model_student)
    model_teacher = instantiate(cfg.model_teacher)

    logger = logging.getLogger("detectron2")
    logger.info("Model Student:\n{}".format(model_student))
    logger.info("Model Teacher:\n{}".format(model_teacher))
    model_student.to(cfg.train.device)
    model_teacher.to(cfg.train.device)
    
    # instantiate optimizer
    cfg.optimizer.params.model = model_student
    optim = instantiate(cfg.optimizer)

    # build training loader
    train_loader_label, train_loader_unlabel = instantiate(cfg.dataloader.train)
    
    # create ddp model
    model_student = create_ddp_model(model_student, **cfg.train.ddp)
    model_teacher = create_ddp_model(model_teacher, **cfg.train.ddp)

    # --- Compute per-class labeled sample counts for adaptive thresholds (Eq. 5) ---
    init_class_thresholds = None
    if getattr(cfg.ssl, 'adaptive_threshold', False):
        from detectron2.data import DatasetCatalog, MetadataCatalog
        label_dataset_name = cfg.dataloader.train.dataset_label.names
        label_dicts = DatasetCatalog.get(label_dataset_name)
        num_classes = len(MetadataCatalog.get(cfg.dataloader.test.dataset.names).thing_classes)
        class_counts = np.zeros(num_classes, dtype=np.float32)
        for d in label_dicts:
            for ann in d.get('annotations', []):
                cid = ann['category_id']
                if 0 <= cid < num_classes:
                    class_counts[cid] += 1
        total = class_counts.sum()
        t_min_c = getattr(cfg.ssl, 'T_min_c', cfg.ssl.class_quality_threshold)
        t_max_c = getattr(cfg.ssl, 'T_max_c', cfg.ssl.class_quality_threshold)
        t_min_m = getattr(cfg.ssl, 'T_min_m', cfg.ssl.mask_quality_threshold)
        t_max_m = getattr(cfg.ssl, 'T_max_m', cfg.ssl.mask_quality_threshold)
        freq = class_counts / (total + 1e-8)
        init_class_thresholds = {
            'class': torch.tensor(t_min_c + (t_max_c - t_min_c) * freq, dtype=torch.float32),
            'mask':  torch.tensor(t_min_m + (t_max_m - t_min_m) * freq, dtype=torch.float32),
        }

    CLIP_model, class_name = None, None
    if cfg.ssl.refine_label:
        from pl_dc.modeling.clip import CLIP
        from detectron2.data import MetadataCatalog
        # load clip model
        CLIP_model = CLIP(
            model_name=cfg.ssl.clip.model_name,
            pretrained_weights=cfg.ssl.clip.pretrained_weights
        )
        thing_classes = MetadataCatalog.get(cfg.dataloader.test.dataset.names).thing_classes
        try:
            from pl_dc.modeling.category_descriptions import (
                COCO_CATEGORY_DESCRIPTIONS,
                CITYSCAPES_CATEGORY_DESCRIPTIONS,
            )
            dataset_name = cfg.dataloader.test.dataset.names
            if 'coco' in dataset_name:
                desc_dict = COCO_CATEGORY_DESCRIPTIONS
            elif 'cityscapes' in dataset_name:
                desc_dict = CITYSCAPES_CATEGORY_DESCRIPTIONS
            else:
                desc_dict = {}
            class_name = [desc_dict.get(c, [c]) for c in thing_classes]
        except ImportError:
            class_name = [[c] for c in thing_classes]

    trainer = Trainer(
        model_student=model_student,
        model_teacher=model_teacher,
        dataloader_label=train_loader_label,
        dataloader_unlabel=train_loader_unlabel,
        optimizer=optim,
        burn_up_step=cfg.ssl.burn_up_step,
        teacher_update_iter=cfg.ssl.teacher_update_iter,
        ema_keep_rate=cfg.ssl.ema_keep_rate,
        mask_quality_threshold=cfg.ssl.mask_quality_threshold,
        class_quality_threshold=cfg.ssl.class_quality_threshold,
        mask_area_threshold=cfg.ssl.mask_area_threshold,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
        output_dir=cfg.train.output_dir,
        CLIP_model=CLIP_model,
        class_name=class_name,
        init_class_thresholds=init_class_thresholds,
    )
    
    # Ensemble teacher and student model is for model saving and loading
    ensem_ts_model = EnsembleTSModel(model_teacher, model_student)

    checkpointer = DetectionTSCheckpointer(
        ensem_ts_model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    if comm.is_main_process():
        # writers = default_writers(cfg.train.output_dir, cfg.train.max_iter)
        output_dir = cfg.train.output_dir
        PathManager.mkdirs(output_dir)
        writers = [
            CommonMetricPrinter(cfg.train.max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir),
        ]
        if cfg.train.wandb.enabled:
            PathManager.mkdirs(cfg.train.wandb.params.dir)
            writers.append(WandbWriter(cfg))

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            ema.EMAHook(cfg, model_student) if cfg.train.model_ema.enabled else None,
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model_student)),
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model_teacher, teacher=True)),
            hooks.PeriodicWriter(
                writers,
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)



def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    
    # Enable fast debugging by running several iterations to check for any bugs.
    if cfg.train.fast_dev_run.enabled:
        cfg.train.max_iter = 20
        cfg.train.eval_period = 10
        cfg.train.log_period = 1

    if args.eval_only:
        model_student = instantiate(cfg.model_student)
        model_teacher = instantiate(cfg.model_teacher)
        model_student.to(cfg.train.device)
        model_teacher.to(cfg.train.device)
        model_student = create_ddp_model(model_student)
        model_teacher = create_ddp_model(model_teacher)
        
        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model_student)

        DetectionTSCheckpointer(ensem_ts_model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, ensem_ts_model.modelTeacher, eval_only=True))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
