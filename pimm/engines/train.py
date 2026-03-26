"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from packaging import version
from functools import partial
from itertools import islice
import contextlib

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pimm.utils.comm as comm
from pimm.datasets import build_dataset, point_collate_fn, collate_fn, inseg_collate_fn
from pimm.models import build_model
from pimm.utils.logger import get_root_logger
from pimm.utils.optimizer import build_optimizer
from pimm.utils.scheduler import build_scheduler
from pimm.utils.events import EventStorage, ExceptionWriter, WandbSummaryWriter
from pimm.utils.registry import Registry

TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    bfloat16=torch.bfloat16,
)


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.start_iter = 0  # Added for resuming from specific step
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        if self.cfg.matmul_precision is not None:
            torch.set_float32_matmul_precision(self.cfg.matmul_precision)
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.start_iter = 0
        # When resuming, use cfg.epoch as the absolute horizon so we can
        # extend training beyond the previous end without changing schedules.
        self.max_epoch = cfg.epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.test_loader = self.build_test_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)
        self.logger.info("=> Running config modifiers ...")
        for h in self.hooks:
            h.modify_config(self.cfg)
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        

    def train(self):
        anomaly_context = torch.autograd.detect_anomaly() if self.cfg.detect_anomaly else contextlib.nullcontext()
        with EventStorage() as self.storage, ExceptionWriter(), anomaly_context:
            # => before train (loads checkpoint, sets start_iter and start_epoch)
            self.before_train()
            
            # capture start_iter for skipping (set by checkpoint hook in before_train)
            skip_batches = self.start_iter
            self.start_iter = 0
            
            # sync storage.iter with resumed position for correct wandb/tensorboard logging
            iter_per_epoch = len(self.train_loader)
            resumed_iter = self.start_epoch * iter_per_epoch + skip_batches
            if resumed_iter > 0:
                self.storage.iter = resumed_iter
                self.logger.info(f"Resuming from iteration {resumed_iter}")
            
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.comm_info["epoch"] = self.epoch
                self.comm_info["iter_per_epoch"] = iter_per_epoch
                self.model.train()
                
                # skip batches if resuming mid-epoch
                if skip_batches > 0:
                    self.logger.info(f"Skipping first {skip_batches} batches")
                    self.data_iterator = enumerate(
                        islice(self.train_loader, skip_batches, None),
                        start=skip_batches
                    )
                    skip_batches = 0  # only skip on first epoch
                else:
                    self.data_iterator = enumerate(self.train_loader)

                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        # Log average points per sample for throughput analysis
        if "offset" in input_dict:
            output_dict["avg_pts"] = input_dict["coord"].shape[0] / len(input_dict["offset"])
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters:,}")
        
        # Move model to GPU before wrapping
        model = model.cuda()
        
        # Use standard DDP
        model = create_ddp_model(
            model,
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        if self.cfg.get('use_wandb', False):
            writer = WandbSummaryWriter(
                project=self.cfg.get('wandb_project', 'pimm'),
                name=self.cfg.get('wandb_run_name', os.path.basename(self.cfg.save_path)),
                config=self.cfg
            ) if comm.is_main_process() else None
            self.logger.info(f"Weights & Biases writer initialized with project: {self.cfg.get('wandb_project', 'pimm')}")
        else:
            writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
            self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=(self.cfg.num_worker_per_gpu > 0),
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
                # in_order=self.cfg.deterministic,
            )
        return val_loader

    def build_test_loader(self):
        test_loader = None
        if self.cfg.evaluate and hasattr(self.cfg.data, 'test'):
            test_data = build_dataset(self.cfg.data.test)
            if comm.get_world_size() > 1:
                test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
            else:
                test_sampler = None
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=test_sampler,
                collate_fn=collate_fn,
                # in_order=self.cfg.deterministic,
            )
        return test_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if not self.cfg.enable_amp:
            return None
            
        # Use standard grad scaler for DDP
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler()
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader


@TRAINERS.register_module("InsegTrainer")
class InsegTrainer(Trainer):
    """Trainer for instance segmentation that handles multiple queries per sample."""
    
    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        # Use our custom collate function for instance segmentation
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(inseg_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=(self.cfg.num_worker_per_gpu > 0),
            in_order=self.cfg.deterministic,
        )
        return train_loader
    
    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                # Use inseg_collate_fn for validation as well
                collate_fn=partial(inseg_collate_fn, mix_prob=0),
                # in_order=self.cfg.deterministic,
            )
        return val_loader
