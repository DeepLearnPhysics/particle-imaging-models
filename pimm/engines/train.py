"""
Trainer implementations and lifecycle orchestration for pimm.

The default trainer builds runtime components, runs hooks around train, epoch,
and step boundaries, moves batches to the selected parallel device, and records
checkpointable resume state after each optimization step.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import contextlib
import os
import sys
import weakref
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.utils.data
from packaging import version

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter
from torchdata.stateful_dataloader import StatefulDataLoader

import pimm.utils.comm as comm
from pimm.datasets import (
    build_dataset,
    collate_fn,
    inseg_collate_fn,
    point_collate_fn,
    StatefulRandomSampler,
    set_dataloader_epoch,
)
from pimm.distributed import (
    create_parallel_context,
    move_batch_to_device,
    prepare_model,
    unwrap_model,
)
from pimm.models import build_model
from pimm.utils.events import EventStorage, ExceptionWriter, WandbSummaryWriter
from pimm.utils.logger import get_root_logger
from pimm.utils.optimizer import build_optimizer
from pimm.utils.registry import Registry
from pimm.utils.scheduler import build_scheduler

from ._train_utils import worker_init_fn
from .hooks import HookBase, build_hooks
from ._train_utils import TrainState

TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    bfloat16=torch.bfloat16,
)


class TrainerBase:
    """Base class for hook-driven trainer lifecycle execution."""

    def __init__(self) -> None:
        """Initialize shared lifecycle counters and hook-visible state."""
        self.hooks = []
        self.cfg: Any = None
        self.logger: Any = None
        self.model: nn.Module | None = None
        self.train_loader: Any = None
        self.val_loader: Any = None
        self.test_loader: Any = None
        self.optimizer: Any = None
        self.scheduler: Any = None
        self.scaler: Any = None
        self.epoch = 0
        self.start_epoch = 0
        self.start_iter = 0  # First dataloader position to consume on resume.
        self.max_epoch = 0
        self.max_iter = 0
        self.global_step = 0
        self.samples_seen = 0
        self.best_metric_value = -torch.inf
        self.train_state = TrainState()
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage | None = None
        self.writer: SummaryWriter | None = None

    def register_hooks(self, hooks) -> None:
        """Build hooks and attach this trainer through weak references."""
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
        """Run the generic train/epoch/step lifecycle."""
        with EventStorage() as self.storage:
            # Hooks bracket the whole run, each epoch, and each step.
            self.before_train()
            if self._training_already_complete():
                self._finish_completed_resume()
                return
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    self.before_step()
                    self.run_step()
                    self.after_step()
                self.after_epoch()
            self.after_train()

    def before_train(self):
        """Apply global numeric settings and call before-train hooks."""
        if self.cfg.matmul_precision is not None:
            torch.set_float32_matmul_precision(self.cfg.matmul_precision)
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        """Call hooks before the current epoch starts."""
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        """Call hooks before consuming the current batch."""
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        """Run one optimization step for the current batch."""
        raise NotImplementedError

    def after_step(self):
        """Call hooks after the current optimization step."""
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        """Call epoch-end hooks and reset per-epoch event histories."""
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        """Synchronize workers, call final hooks, and close the writer."""
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        self._close_writer()

    def _training_already_complete(self):
        """Return whether restored progress is already at the run horizon."""
        return int(getattr(self, "start_epoch", 0)) >= int(getattr(self, "max_epoch", 0))

    def _finish_completed_resume(self):
        """Exit a resumed, already-complete run without final checkpoint churn."""
        if hasattr(self, "logger"):
            self.logger.info(
                "Training already complete: "
                f"start_epoch={self.start_epoch}, max_epoch={self.max_epoch}. "
                "Exiting without running more steps."
            )
        comm.synchronize()
        self._close_writer()

    def _close_writer(self):
        """Close the event writer on the main process when one exists."""
        if comm.is_main_process():
            writer = getattr(self, "writer", None)
            if writer is not None:
                writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    """Default single-dataset trainer with resume-aware dataloading."""

    def __init__(self, cfg):
        """Build model, data loaders, optimizer, scheduler, hooks, and writer."""
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.start_iter = 0
        self.global_step = 0
        self.samples_seen = 0
        self.train_state = TrainState()
        self.parallel_context = create_parallel_context(cfg)
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
        """Run training from the configured or restored epoch/iteration."""
        anomaly_context = torch.autograd.detect_anomaly() if self.cfg.detect_anomaly else contextlib.nullcontext()
        with EventStorage() as self.storage, ExceptionWriter(), anomaly_context:
            # Checkpoint hooks can restore start_epoch, start_iter, and counters.
            self.before_train()
            
            # Keep metric writers aligned with the absolute optimization step.
            iter_per_epoch = len(self.train_loader)
            resumed_iter = self.global_step or (
                self.start_epoch * iter_per_epoch + self.start_iter
            )
            if resumed_iter > 0:
                self.storage.iter = resumed_iter
                self._align_writer_step(resumed_iter)
                self.logger.info(f"Resuming from iteration {resumed_iter}")
            if self._training_already_complete():
                self._finish_completed_resume()
                return
            
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                resume_mid_epoch = (
                    self.epoch == self.start_epoch and self.start_iter > 0
                )
                set_dataloader_epoch(
                    self.train_loader,
                    self.epoch,
                    reset_position=not resume_mid_epoch,
                )
                self.comm_info["epoch"] = self.epoch
                self.comm_info["iter_per_epoch"] = iter_per_epoch
                self.model.train()
                
                start_iter = self.start_iter if resume_mid_epoch else 0
                if start_iter > 0:
                    self.logger.info(
                        f"Resuming epoch {self.epoch} from dataloader position {start_iter}"
                    )
                self.data_iterator = enumerate(self.train_loader, start=start_iter)

                self.before_epoch()
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    self.before_step()
                    self.run_step()
                    # Capture the next resume point before checkpoint hooks run.
                    self._record_step_state()
                    self.after_step()
                self.start_iter = 0
                self.after_epoch()
            self.after_train()

    def _align_writer_step(self, global_step):
        """Align writer-internal step counters after checkpoint resume."""
        if self.writer is None:
            return
        if hasattr(self.writer, "step"):
            self.writer.step = max(int(getattr(self.writer, "step", 0)), int(global_step))
        self.cfg.log_step_offset = int(getattr(self.cfg, "log_step_offset", 0) or 0)

    def _record_step_state(self):
        """Update checkpointable counters for the next batch to consume."""
        iter_per_epoch = int(self.comm_info.get("iter_per_epoch", len(self.train_loader)))
        iter_in_epoch = int(self.comm_info.get("iter", 0)) + 1
        self.global_step = self.epoch * iter_per_epoch + iter_in_epoch

        input_dict = self.comm_info.get("input_dict", {})
        local_batch = self.cfg.batch_size_per_gpu
        # Point datasets use offset length as the actual per-rank sample count.
        if isinstance(input_dict, dict) and "offset" in input_dict:
            try:
                local_batch = len(input_dict["offset"])
            except TypeError:
                local_batch = self.cfg.batch_size_per_gpu
        self.samples_seen += int(local_batch) * comm.get_world_size()

        self.train_state = TrainState.from_trainer(self)


    def run_step(self):
        """Move one batch to device, run forward/backward, and update LR."""
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(
                torch.amp.autocast,
                device_type=self.parallel_context.device.type,
            )
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = move_batch_to_device(
            self.comm_info["input_dict"],
            self.parallel_context.device,
        )
        # Store the device-resident batch so hooks and checkpoint state agree.
        self.comm_info["input_dict"] = input_dict

        with auto_cast(
            enabled=self.cfg.enable_amp,
            dtype=AMP_DTYPE[self.cfg.amp_dtype],
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
        if self.cfg.empty_cache and self.parallel_context.device.type == "cuda":
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        """Run epoch-end hooks, clear histories, and optionally empty CUDA cache."""
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        """Construct the model and wrap it for the configured parallel strategy."""
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters:,}")
        model = prepare_model(model, self.cfg, self.parallel_context)
        self.logger.info(
            f"Parallel strategy: {self.parallel_context.strategy}, device: {self.parallel_context.device}"
        )
        return model

    def build_writer(self):
        """Create a main-rank summary writer for TensorBoard or W&B."""
        if self.cfg.get('use_wandb', False):
            wandb_kwargs = dict(
                project=self.cfg.get('wandb_project', 'pimm'),
                name=self.cfg.get('wandb_run_name', os.path.basename(self.cfg.save_path)),
                config=self.cfg,
                step_offset=self.cfg.get('log_step_offset', 0),
            )
            for cfg_key, wandb_key in (
                ('wandb_group', 'group'),
                ('wandb_job_type', 'job_type'),
                ('wandb_run_id', 'id'),
                ('wandb_resume', 'resume'),
            ):
                value = self.cfg.get(cfg_key, None)
                if value is not None:
                    wandb_kwargs[wandb_key] = value
            writer = WandbSummaryWriter(**wandb_kwargs) if comm.is_main_process() else None
            self.logger.info(f"Weights & Biases writer initialized with project: {self.cfg.get('wandb_project', 'pimm')}")
        else:
            writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
            self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        """Build the stateful training loader used for mid-epoch resume."""
        train_data = build_dataset(self.cfg.data.train)

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

        # Stateful sampler/loader snapshots let checkpoints resume mid-epoch.
        sampler = StatefulRandomSampler(
            train_data,
            shuffle=True,
            seed=self.cfg.seed if self.cfg.seed is not None else 0,
            num_replicas=comm.get_world_size(),
            rank=comm.get_rank(),
            drop_last=len(train_data) > self.cfg.batch_size,
        )
        train_loader = StatefulDataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            sampler=sampler,
            num_workers=self.cfg.num_worker_per_gpu,
            collate_fn=partial(collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=(self.cfg.num_worker_per_gpu > 0),
            snapshot_every_n_steps=self.cfg.get("dataloader_snapshot_every_n_steps", 1),
        )
        return train_loader

    def build_val_loader(self):
        """Build the optional validation loader."""
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
        """Build the optional test loader used by evaluation hooks."""
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
        """Build the optimizer from config."""
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        """Build a scheduler sized to all optimizer steps in training."""
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        """Build an AMP gradient scaler when mixed precision is enabled."""
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


@TRAINERS.register_module("GRPOTrainer")
class GRPOTrainer(Trainer):
    """Trainer-level GRPO loop with cached rollouts and K policy updates."""

    def _sync_grpo_scalar_metrics(self, output_dict):
        """Reduce scalar GRPO metrics across ranks with key-specific ops."""
        if comm.get_world_size() < 2:
            return output_dict

        synced = {}
        world_size = comm.get_world_size()
        for key, value in output_dict.items():
            if not torch.is_tensor(value) or value.numel() != 1:
                synced[key] = value
                continue

            metric = value.detach().float()
            if metric.device.type == "cpu":
                metric = metric.cuda()

            if "_min" in key:
                reduce_op = torch.distributed.ReduceOp.MIN
            elif "_max" in key or "abs_max" in key:
                reduce_op = torch.distributed.ReduceOp.MAX
            else:
                reduce_op = torch.distributed.ReduceOp.SUM

            torch.distributed.all_reduce(metric, op=reduce_op)
            if reduce_op == torch.distributed.ReduceOp.SUM:
                metric /= world_size
            synced[key] = metric
        return synced

    def _policy_updates_per_rollout(self):
        """Return the number of policy updates to run per sampled rollout."""
        train_cfg = getattr(self.cfg, "train", {})
        if hasattr(train_cfg, "get"):
            return max(1, int(train_cfg.get("policy_updates_per_rollout", 1)))
        return max(1, int(getattr(train_cfg, "policy_updates_per_rollout", 1)))

    def _trajectory_microbatch_size(self):
        """Return trajectory microbatch size, or zero to disable splitting."""
        train_cfg = getattr(self.cfg, "train", {})
        if hasattr(train_cfg, "get"):
            value = train_cfg.get("trajectory_microbatch_size", 0)
        else:
            value = getattr(train_cfg, "trajectory_microbatch_size", 0)
        return max(0, int(value or 0))

    def build_scheduler(self):
        """Build a scheduler sized to rollout count times policy updates."""
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = (
            len(self.train_loader) * self.cfg.epoch * self._policy_updates_per_rollout()
        )
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def _optimizer_update(self, loss):
        """Apply one optimizer/scheduler update for a GRPO loss."""
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)
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

    def _rollout_metric_sums(self, trajectories):
        """Aggregate scalar rollout metrics over trajectories."""
        metric_sums = {}
        for traj in trajectories:
            for key, value in traj.metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        return metric_sums

    def _slice_grpo_event(self, event, start, end):
        """Return an event view containing a trajectory slice."""
        trajectories = event["trajectories"][start:end]
        sliced = dict(event)
        sliced["trajectories"] = trajectories
        if "advantages" in sliced:
            sliced["advantages"] = sliced["advantages"][start:end]
        if "step_advantages" in sliced and sliced["step_advantages"] is not None:
            sliced["step_advantages"] = sliced["step_advantages"][start:end]
        return sliced

    def _iter_grpo_trajectory_microbatches(self, rollout_batch, microbatch_size):
        """Yield rollout-batch chunks split by trajectory count."""
        if microbatch_size <= 0:
            yield rollout_batch
            return

        for event in rollout_batch["events"]:
            trajectories = event["trajectories"]
            for start in range(0, len(trajectories), microbatch_size):
                end = min(start + microbatch_size, len(trajectories))
                sliced_event = self._slice_grpo_event(event, start, end)
                sliced_trajectories = sliced_event["trajectories"]
                chunk = dict(rollout_batch)
                chunk["events"] = [sliced_event]
                chunk["metric_count"] = len(sliced_trajectories)
                chunk["metric_sums"] = self._rollout_metric_sums(sliced_trajectories)
                if "reward_stds" in chunk:
                    chunk["reward_stds"] = [sliced_event.get("reward_std", event.get("reward_std"))]
                if "rloo_score_means" in chunk and "rloo_score_mean" in sliced_event:
                    chunk["rloo_score_means"] = [sliced_event["rloo_score_mean"]]
                if "rloo_score_stds" in chunk and "rloo_score_std" in sliced_event:
                    chunk["rloo_score_stds"] = [sliced_event["rloo_score_std"]]
                yield chunk

    def _combine_weighted_grpo_outputs(self, weighted_outputs):
        """Combine microbatch outputs using their metric-count weights."""
        if not weighted_outputs:
            raise RuntimeError("GRPOTrainer received no microbatch outputs")

        combined = {}
        first_output = weighted_outputs[0][0]
        tensor_keys = [
            key for key, value in first_output.items() if torch.is_tensor(value)
        ]
        for key in tensor_keys:
            values = [(output[key], weight) for output, weight in weighted_outputs if key in output]
            if not values:
                continue
            if "_min" in key:
                combined[key] = torch.stack([value.detach() for value, _ in values]).min()
            elif "_max" in key or "abs_max" in key:
                combined[key] = torch.stack([value.detach() for value, _ in values]).max()
            else:
                combined[key] = torch.stack(
                    [value.detach() * float(weight) for value, weight in values]
                ).sum()

        for key, value in first_output.items():
            if key not in combined and not torch.is_tensor(value):
                combined[key] = value
        return combined

    def _optimizer_update_grpo_microbatched(
        self,
        model_impl,
        rollout_batch,
        *,
        update_index,
        policy_updates,
        microbatch_size,
        auto_cast,
    ):
        """Backpropagate GRPO loss over trajectory microbatches."""
        chunks = list(
            self._iter_grpo_trajectory_microbatches(rollout_batch, microbatch_size)
        )
        total_count = sum(max(0, int(chunk.get("metric_count", 0))) for chunk in chunks)
        total_count = max(total_count, 1)
        weighted_outputs = []

        self.optimizer.zero_grad()
        for chunk in chunks:
            weight = max(0, int(chunk.get("metric_count", 0))) / total_count
            with auto_cast(
                enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
            ):
                output_dict = model_impl.grpo_loss_from_batch(
                    chunk,
                    update_index=update_index,
                    policy_updates_per_rollout=policy_updates,
                )
                loss = output_dict["loss"] * float(weight)
            detached = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in output_dict.items()
            }
            weighted_outputs.append((detached, weight))
            if self.cfg.enable_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        if self.cfg.enable_amp:
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()

        combined = self._combine_weighted_grpo_outputs(weighted_outputs)
        if hasattr(model_impl, "_rollout_metric_tensors"):
            combined.update(model_impl._rollout_metric_tensors(rollout_batch))
        return combined

    def _combine_grpo_outputs(self, outputs):
        """Summarize first, last, and mean metrics across policy updates."""
        if not outputs:
            raise RuntimeError("GRPOTrainer received no update outputs")
        first = outputs[0]
        last = outputs[-1]
        combined = {}
        for key, value in last.items():
            combined[key] = value.detach() if torch.is_tensor(value) else value
        if "loss" in first:
            combined["loss"] = torch.stack(
                [output["loss"].detach() for output in outputs]
            ).mean()

        tracked = [
            "rl_pg_loss",
            "rl_kl_loss",
            "rl_kl",
            "rl_ratio_geom",
            "rl_log_ratio",
            "rl_log_ratio_min",
            "rl_log_ratio_max",
            "rl_stop_log_ratio",
            "rl_stop_log_ratio_min",
            "rl_stop_log_ratio_max",
            "rl_class_log_ratio",
            "rl_class_log_ratio_min",
            "rl_class_log_ratio_max",
            "rl_kernel_log_ratio",
            "rl_kernel_log_ratio_min",
            "rl_kernel_log_ratio_max",
            "rl_kernel_dim_log_ratio",
            "rl_kernel_dim_log_ratio_min",
            "rl_kernel_dim_log_ratio_max",
            "rl_clip_frac",
            "rl_logprob",
            "rl_advantage_abs_max",
        ]
        for key in tracked:
            if key not in first or key not in last:
                continue
            first_value = first[key].detach() if torch.is_tensor(first[key]) else first[key]
            last_value = last[key].detach() if torch.is_tensor(last[key]) else last[key]
            combined[f"{key}_update0"] = first_value
            combined[f"{key}_last"] = last_value
            if torch.is_tensor(first[key]):
                combined[f"{key}_mean_update"] = torch.stack(
                    [output[key].detach() for output in outputs]
                ).mean()
        return combined

    def run_step(self):
        """Sample rollouts, run policy updates, and publish GRPO metrics."""
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(
                torch.amp.autocast,
                device_type=self.parallel_context.device.type,
            )
        else:
            auto_cast = torch.cuda.amp.autocast

        input_dict = move_batch_to_device(
            self.comm_info["input_dict"],
            self.parallel_context.device,
        )
        # The model rollout and loss APIs expect all tensors on one device.
        self.comm_info["input_dict"] = input_dict

        model_impl = unwrap_model(self.model)
        if not hasattr(model_impl, "sample_grpo_batch"):
            raise RuntimeError(
                "GRPOTrainer requires model.sample_grpo_batch"
            )
        if not hasattr(model_impl, "grpo_loss_from_batch"):
            raise RuntimeError(
                "GRPOTrainer requires model.grpo_loss_from_batch"
            )

        policy_updates = self._policy_updates_per_rollout()
        microbatch_size = self._trajectory_microbatch_size()
        with auto_cast(
            enabled=self.cfg.enable_amp,
            dtype=AMP_DTYPE[self.cfg.amp_dtype],
        ):
            rollout_batch = model_impl.sample_grpo_batch(input_dict)

        update_outputs = []
        for update_index in range(policy_updates):
            if microbatch_size > 0:
                output_dict = self._optimizer_update_grpo_microbatched(
                    model_impl,
                    rollout_batch,
                    update_index=update_index,
                    policy_updates=policy_updates,
                    microbatch_size=microbatch_size,
                    auto_cast=auto_cast,
                )
            else:
                with auto_cast(
                    enabled=self.cfg.enable_amp,
                    dtype=AMP_DTYPE[self.cfg.amp_dtype],
                    device_type=self.parallel_context.device.type,
                ):
                    output_dict = model_impl.grpo_loss_from_batch(
                        rollout_batch,
                        update_index=update_index,
                        policy_updates_per_rollout=policy_updates,
                    )
                    loss = output_dict["loss"]
                self._optimizer_update(loss)
            update_outputs.append(output_dict)

        if self.cfg.empty_cache and self.parallel_context.device.type == "cuda":
            torch.cuda.empty_cache()
        combined = self._combine_grpo_outputs(update_outputs)
        if "offset" in input_dict:
            avg_pts = input_dict["coord"].shape[0] / len(input_dict["offset"])
            combined["avg_pts"] = torch.as_tensor(
                avg_pts, device=input_dict["coord"].device, dtype=torch.float32
            )
        combined = self._sync_grpo_scalar_metrics(combined)
        self.comm_info["model_output_dict"] = combined


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    """Trainer variant that delegates sampling to MultiDatasetDataloader."""

    def build_train_loader(self):
        """Build a multi-dataset train loader and expose its epoch length."""
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
        """Build the stateful instance-segmentation training loader."""
        train_data = build_dataset(self.cfg.data.train)

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
        sampler = StatefulRandomSampler(
            train_data,
            shuffle=True,
            seed=self.cfg.seed if self.cfg.seed is not None else 0,
            num_replicas=comm.get_world_size(),
            rank=comm.get_rank(),
            drop_last=len(train_data) > self.cfg.batch_size,
        )
        train_loader = StatefulDataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            sampler=sampler,
            num_workers=self.cfg.num_worker_per_gpu,
            collate_fn=partial(inseg_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=(self.cfg.num_worker_per_gpu > 0),
            in_order=self.cfg.deterministic,
            snapshot_every_n_steps=self.cfg.get("dataloader_snapshot_every_n_steps", 1),
        )
        return train_loader
    
    def build_val_loader(self):
        """Build the optional instance-segmentation validation loader."""
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


@TRAINERS.register_module("ImageClassTrainer")
class ImageClassTrainer(Trainer):
    """Trainer for dense 2D image batches (e.g. rasterized ring images).

    Uses ``default_collate`` to stack per-event ``image`` -> ``(B, C, H, W)`` and
    scalar labels/momenta -> ``(B, 1)`` instead of the point-cloud collate that
    concatenates variable-length clouds along a single axis.
    """

    def build_train_loader(self):
        """Build the stateful image-classification training loader."""
        train_data = build_dataset(self.cfg.data.train)

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

        sampler = StatefulRandomSampler(
            train_data,
            shuffle=True,
            seed=self.cfg.seed if self.cfg.seed is not None else 0,
            num_replicas=comm.get_world_size(),
            rank=comm.get_rank(),
            drop_last=len(train_data) > self.cfg.batch_size,
        )
        train_loader = StatefulDataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            sampler=sampler,
            num_workers=self.cfg.num_worker_per_gpu,
            collate_fn=torch.utils.data.default_collate,
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=(self.cfg.num_worker_per_gpu > 0),
            snapshot_every_n_steps=self.cfg.get("dataloader_snapshot_every_n_steps", 1),
        )
        return train_loader

    def build_val_loader(self):
        """Build the optional image-classification validation loader."""
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
                collate_fn=torch.utils.data.default_collate,
            )
        return val_loader
