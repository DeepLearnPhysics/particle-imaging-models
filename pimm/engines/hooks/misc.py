"""
Misc Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import re
import sys
import glob
import os
import shutil
import time
import gc
import torch
import torch.nn as nn
import torch.utils.data
from collections import OrderedDict

if sys.version_info >= (3, 10):
    from collections.abc import Sequence
else:
    from collections import Sequence
from pimm.utils.timer import Timer
from pimm.utils.comm import is_main_process, synchronize
from pimm.utils.cache import shared_dict
from pimm.utils.scheduler import CosineScheduler
import pimm.utils.comm as comm

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class WandbNamer(HookBase):
    """
    Auto-generate wandb_run_name from config values.
    
    Simple hook that joins specified config values with a separator to create
    a descriptive wandb run name. No lambdas or templates - just a list of keys.
    
    Args:
        keys: Tuple of config keys to include in name.
              Supports nested keys: "data.train.max_len" or "model.type"
        sep: Join character (default: "-")
        format_numbers: Format large numbers with suffixes (1000000 -> 1M)
        extra: Extra strings to append (str or tuple of strings), e.g. "fft", "scratch"
    
    Example:
        dict(type="WandbNamer", keys=("model.type", "data.train.max_len", "seed"), extra="fft")
        # generates: "Sonata-v1m1-1M-0-fft"
    
    CLI override:
        --options wandb_run_name=my-custom-name  # overrides auto-generated name
    """
    
    def __init__(self, keys=(), sep="-", format_numbers=True, extra=None):
        self.keys = keys
        self.sep = sep
        self.format_numbers = format_numbers
        # normalize extra to tuple
        if extra is None:
            self.extra = ()
        elif isinstance(extra, str):
            self.extra = (extra,)
        else:
            self.extra = tuple(extra)
    
    def _get_nested(self, cfg, key_path):
        """Get nested config value: 'data.train.max_len' -> cfg.data['train']['max_len']"""
        parts = key_path.split('.')
        val = cfg
        for part in parts:
            if hasattr(val, part):
                val = getattr(val, part)
            elif isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return None
        return val
    
    def _format_value(self, val):
        """Format a value for the run name."""
        if self.format_numbers and isinstance(val, (int, float)):
            return self._format_number(val)
        return str(val)
    
    def _format_number(self, n):
        """Format large numbers with suffixes (K, M, B)."""
        n_val = float(n)
        if abs(n_val) >= 1_000_000_000:
            return f"{n_val / 1_000_000_000:.1f}B".rstrip('0').rstrip('.')
        elif abs(n_val) >= 1_000_000:
            return f"{n_val / 1_000_000:.1f}M".rstrip('0').rstrip('.')
        elif abs(n_val) >= 1_000:
            return f"{n_val / 1_000:.1f}K".rstrip('0').rstrip('.')
        return str(int(n_val) if n_val == int(n_val) else n_val)
    
    def modify_config(self, cfg):
        """Build wandb_run_name from specified keys."""
        # skip if wandb_run_name already set via CLI
        if 'wandb_run_name' in getattr(cfg, '_cli_options', set()):
            return
        
        parts = []
        for key in self.keys:
            val = self._get_nested(cfg, key)
            if val is not None:
                parts.append(self._format_value(val))
        
        # append extra strings
        parts.extend(self.extra)
        
        if parts:
            cfg.wandb_run_name = self.sep.join(parts)

@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        _remain_epoch = self.trainer.max_epoch - self.trainer.start_epoch
        self._remain_iter = _remain_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self):
        self.model_output_keys = []

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""

    def _get_global_step(self):
        # compute global step same way as GradientNormLogger for consistency
        current_epoch = self.trainer.comm_info["epoch"] + 1
        current_iter = self.trainer.comm_info["iter"]
        return (current_epoch - 1) * len(self.trainer.train_loader) + current_iter + 1

    def before_step(self):
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            # exclude large tensor outputs and keep only scalar-like entries
            large_tensor_keys = {'seg_logits', 'sem_logits', 'instance_embedding', 'vertex_embedding', 'sigma', 'point', 'pred_logits', 'pred_masks'}
            
            # if total_loss exists (synced loss), use it as 'loss' and skip original 'loss'
            has_total_loss = 'total_loss' in model_output_dict
            
            self.model_output_keys = [
                k
                for k in model_output_dict.keys()
                if (k not in large_tensor_keys and k != 'teacher_logits' and "match_" not in k
                    and not (has_total_loss and k == 'loss')  # skip 'loss' if total_loss exists
                    and k != 'total_loss')  # we'll handle total_loss separately
            ]
            
            # add 'loss' key if total_loss exists (to log total_loss as 'loss')
            if has_total_loss:
                self.model_output_keys.append('loss')
            
            for key in self.model_output_keys:
                # use total_loss value when logging 'loss'
                if key == 'loss' and has_total_loss:
                    val = model_output_dict['total_loss']
                else:
                    val = model_output_dict[key]
                # support torch.Tensor, Python numbers; skip others gracefully
                try:
                    if hasattr(val, "item") and callable(getattr(val, "item", None)):
                        scalar = float(val.item())
                    elif isinstance(val, (int, float)):
                        scalar = float(val)
                    else:
                        # last resort: try casting
                        scalar = float(val)
                except Exception:
                    continue
                self.trainer.storage.put_scalar(key, scalar)

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            global_step = self._get_global_step()
            self.trainer.writer.add_scalar("params/lr", lr, global_step)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    global_step,
                )

    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )

@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None, evaluator_every_n_steps=None):
        self.save_freq = save_freq  # None or int, None indicate only save model last
        self.evaluator_every_n_steps = evaluator_every_n_steps
        self.step_count = 0

    def after_step(self):
        self.step_count += 1
        if is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate and self.evaluator_every_n_steps and self.step_count % self.evaluator_every_n_steps == 0:
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

                filename = os.path.join(
                    self.trainer.cfg.save_path, "model", "model_last.pth"
                )
                self.trainer.logger.info("Saving checkpoint to: " + filename)
                torch.save(
                    {
                        "epoch": self.trainer.epoch + 1,
                        "iter": self.trainer.comm_info.get("iter", 0) + 1,
                        "state_dict": self.trainer.model.state_dict(),
                        "optimizer": self.trainer.optimizer.state_dict(),
                        "scheduler": self.trainer.scheduler.state_dict(),
                        "scaler": self.trainer.scaler.state_dict() if getattr(self.trainer, "scaler", None) is not None else None,
                        "best_metric_value": self.trainer.best_metric_value,
                    },
                    filename + ".tmp",
                )
                os.replace(filename + ".tmp", filename)
                if is_best:
                    shutil.copyfile(
                        filename,
                        os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                    )
                if self.save_freq and self.step_count % self.save_freq == 0:
                    shutil.copyfile(
                        filename,
                        os.path.join(
                            self.trainer.cfg.save_path,
                            "model",
                            f"iter_{self.step_count}.pth",
                        ),
                    )


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement)
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                # Resume iteration/step if available (backwards compatible)
                if "iter" in checkpoint:
                    self.trainer.start_iter = checkpoint["iter"]
                    self.trainer.logger.info(f"Resuming train at iteration: {self.trainer.start_iter}")
                
                self.trainer.best_metric_value = checkpoint["best_metric_value"]

                # Load optimizer state dict in non-strict mode and show any missing/unexpected keys
                optimizer_state = checkpoint.get("optimizer", None)
                if optimizer_state is not None:
                    try:
                        load_result = self.trainer.optimizer.load_state_dict(optimizer_state, strict=False)
                        missing_keys = load_result.get('missing_keys', []) if isinstance(load_result, dict) else load_result[0] if len(load_result) > 0 else []
                        unexpected_keys = load_result.get('unexpected_keys', []) if isinstance(load_result, dict) else load_result[1] if len(load_result) > 1 else []
                        if missing_keys:
                            self.trainer.logger.info(f"Optimizer missing keys: {missing_keys}")
                        if unexpected_keys:
                            self.trainer.logger.info(f"Optimizer unexpected keys: {unexpected_keys}")
                    except TypeError:
                        # fallback for PyTorch versions where strict is not allowed
                        try:
                            load_result = self.trainer.optimizer.load_state_dict(optimizer_state)
                            missing_keys = load_result.get('missing_keys', []) if isinstance(load_result, dict) else load_result[0] if len(load_result) > 0 else []
                            unexpected_keys = load_result.get('unexpected_keys', []) if isinstance(load_result, dict) else load_result[1] if len(load_result) > 1 else []
                            if missing_keys:
                                self.trainer.logger.info(f"Optimizer missing keys: {missing_keys}")
                            if unexpected_keys:
                                self.trainer.logger.info(f"Optimizer unexpected keys: {unexpected_keys}")
                        except Exception as e:
                            self.trainer.logger.warning(f"Could not load optimizer state dict: {e}")
                else:
                    self.trainer.logger.info("No optimizer state found in checkpoint.")
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp and checkpoint.get("scaler", None) is not None:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")

@HOOKS.register_module()
class CheckpointSaverIteration(HookBase):
    def __init__(self, save_freq=None, save_iter_checkpoints=False):
        self.save_freq = save_freq  # None or int, None indicate only save model last
        self.save_iter_checkpoints = save_iter_checkpoints  # if False, only save model_last/best, not iter_{step}.pth
        self.step_count = 0

    def after_step(self):
        self.step_count += 1
        if is_main_process() and self.save_freq and self.step_count % self.save_freq == 0:
            self.trainer.logger.info(f"Saving checkpoint at step {self.trainer.comm_info['iter']}, {self.trainer.comm_info['iter']} % {self.save_freq} == 0?")
            is_best = False
            if self.trainer.cfg.evaluate and "current_metric_value" in self.trainer.comm_info.keys():
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "iter": self.trainer.comm_info.get("iter", 0) + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "scaler": self.trainer.scaler.state_dict() if self.trainer.cfg.enable_amp else None,
                    "best_metric_value": self.trainer.best_metric_value,
                },
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )
            if self.save_iter_checkpoints and self.save_freq and self.step_count % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"iter_{self.step_count}.pth",
                    ),
                )



@HOOKS.register_module()
class FinalEvaluator(HookBase):
    def __init__(self, test_last=False):
        self.test_last = test_last

    def after_train(self):
        # Skip final evaluation if evaluate is disabled
        if not self.trainer.cfg.get("evaluate", True):
            self.trainer.logger.info(
                "Skipping final evaluation (evaluate=False)"
            )
            return

        # Import here to avoid circular import: pimm.engines.test -> pimm.models -> pimm.engines.hooks.misc
        from pimm.engines.test import TESTERS

        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Final Evaluation >>>>>>>>>>>>>>>>"
        )
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        tester = TESTERS.build(
            dict(type=cfg.test.type, cfg=cfg, model=self.trainer.model)
        )
        if self.test_last:
            self.trainer.logger.info("=> Testing on model_last ...")
        else:
            self.trainer.logger.info("=> Testing on model_best ...")
            best_path = os.path.join(
                self.trainer.cfg.save_path, "model", "model_best.pth"
            )
            checkpoint = torch.load(best_path)
            state_dict = checkpoint["state_dict"]
            tester.model.load_state_dict(state_dict, strict=True)
        tester.test()


@HOOKS.register_module()
class DataCacheOperator(HookBase):
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.data_list = self.get_data_list()

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split))
        else:
            raise NotImplementedError
        return data_list

    def get_cache_name(self, data_path):
        data_name = data_path.replace(os.path.dirname(self.data_root), "")
        return "pimm" + data_name.replace(os.path.sep, "-")

    def before_train(self):
        self.trainer.logger.info(
            f"=> Caching dataset: {self.data_root}, split: {self.split} ..."
        )
        if is_main_process():
            dataset = self.trainer.train_loader.dataset
            for i in range(len(dataset)):
                data_dict = dataset[i]
                name = data_dict["name"]
                shared_dict(f"pimm-{name}", data_dict)
        synchronize()


@HOOKS.register_module()
class RuntimeProfiler(HookBase):
    def __init__(
        self,
        forward=True,
        backward=True,
        interrupt=False,
        warm_up=2,
        sort_by="cuda_time_total",
        row_limit=30,
        memory=True,
    ):
        self.forward = forward
        self.backward = backward
        self.interrupt = interrupt
        self.warm_up = warm_up
        self.sort_by = sort_by
        self.row_limit = row_limit
        self.memory = memory

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import profile, record_function, ProfilerActivity
        if self.memory:
            torch.cuda.memory._record_memory_history()


        logdir = self.trainer.cfg.save_path + "/logdir/"
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        # schedule needs: wait + warmup + active steps (times repeat)
        # loop runs warm_up + 1 iterations, so we match the schedule accordingly
        num_steps = self.warm_up + 1
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                logdir, use_gzip=True
            ),
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=max(1, num_steps - 1), repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for i, input_dict in enumerate(self.trainer.train_loader):
                if i == num_steps:
                    break
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                if self.forward:
                    # with record_function("model_forward"):
                    output_dict = self.trainer.model(input_dict)
                else:
                    output_dict = self.trainer.model(input_dict)

                loss = output_dict["loss"]

                if self.backward:
                    # with record_function("model_backward"):
                    loss.backward()
                prof.step()
                
                self.trainer.logger.info(f"Profile: [{i + 1}/{num_steps}]")

        if self.forward or self.backward:
            self.trainer.logger.info(
                "Profile: \n"
                + str(
                    prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            # prof.export_chrome_trace(
            #     os.path.join(self.trainer.cfg.save_path, "trace.json")
            # )

        if self.memory:
            torch.cuda.memory._dump_snapshot(
                os.path.join(self.trainer.cfg.save_path, "memory_snapshot.pickle")
            )
        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class WeightDecayExclusion(HookBase):
    """
    Organizes optimizer parameter groups to handle both layer-wise learning rates
    and parameter-specific weight decay exclusions (bias, norm, gamma parameters).
    """
    def __init__(
        self,
        exclude_bias_from_wd=True,
        exclude_norm_from_wd=True,
        exclude_gamma_from_wd=True,
        exclude_token_from_wd=True,
        exclude_ndim_1_from_wd=True,
    ):
        self.exclude_bias_from_wd = exclude_bias_from_wd
        self.exclude_norm_from_wd = exclude_norm_from_wd
        self.exclude_gamma_from_wd = exclude_gamma_from_wd
        self.exclude_token_from_wd = exclude_token_from_wd
        self.exclude_ndim_1_from_wd = exclude_ndim_1_from_wd
    def _should_exclude_from_wd(self, name, param):
        """Check if parameter should be excluded from weight decay"""
        if self.exclude_bias_from_wd and name.endswith('.bias'):
            return True
        if self.exclude_norm_from_wd and 'norm' in name.lower():
            return True
        if self.exclude_gamma_from_wd and 'gamma' in name.lower():
            return True
        if self.exclude_token_from_wd and 'token' in name.lower():
            return True
        if self.exclude_ndim_1_from_wd and param.ndim == 1:
            return True
        return False

    def before_train(self):
        """Reorganize optimizer parameter groups to handle weight decay exclusions"""
        model = self.trainer.model
        if hasattr(model, 'module'):  # DDP case
            model = model.module

        # Get original parameter groups configuration
        original_groups = self.trainer.optimizer.param_groups.copy()
        
        # Create new parameter groups
        new_param_groups = []
        
        for group in original_groups:
            # Split this group into two: with and without weight decay
            wd_params = []
            no_wd_params = []
            
            for param in group['params']:
                # Find parameter name
                param_name = None
                for name, model_param in model.named_parameters():
                    if model_param is param:
                        param_name = name
                        break
                
                if param_name and self._should_exclude_from_wd(param_name, param):
                    no_wd_params.append(param)
                else:
                    wd_params.append(param)
            
            # Create group with weight decay if there are parameters
            if wd_params:
                wd_group = group.copy()
                wd_group['params'] = wd_params
                wd_group['apply_wd'] = True  # Mark for weight decay scheduler
                new_param_groups.append(wd_group)
            
            # Create group without weight decay if there are parameters
            if no_wd_params:
                no_wd_group = group.copy()
                no_wd_group['params'] = no_wd_params
                no_wd_group['weight_decay'] = 0.0
                no_wd_group['apply_wd'] = False  # Mark to skip weight decay scheduler
                new_param_groups.append(no_wd_group)
        
        # Update optimizer with new parameter groups
        self.trainer.optimizer.param_groups = new_param_groups
        
        self.trainer.logger.info(f"Reorganized optimizer into {len(new_param_groups)} parameter groups")
        
        # Log parameter counts for debugging
        wd_count = sum(len(g['params']) for g in new_param_groups if g.get('apply_wd', True))
        no_wd_count = sum(len(g['params']) for g in new_param_groups if not g.get('apply_wd', True))
        self.trainer.logger.info(f"Parameter groups with weight decay: {wd_count}")
        self.trainer.logger.info(f"Parameter groups without weight decay: {no_wd_count}")

@HOOKS.register_module()
class WeightDecayScheduler(HookBase):
    def __init__(
        self,
        base_value=0.04,
        final_value=0.2,
        warmup_ratio=1.0,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.warmup_ratio = warmup_ratio
        self.scheduler = None

    def before_train(self):
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        self.scheduler = CosineScheduler(
            base_value=self.base_value,
            final_value=self.final_value,
            total_iters=self.trainer.cfg.scheduler.total_steps * self.warmup_ratio,
        )
        self.scheduler.iter = curr_step

    def before_step(self):
        wd = self.scheduler.step()
        for param_group in self.trainer.optimizer.param_groups:
            # Only apply scheduled weight decay to groups marked for it
            if param_group.get('apply_wd', True):
                param_group["weight_decay"] = wd
            # Groups with apply_wd=False keep their original weight_decay (should be 0.0)
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/wd", wd, self.scheduler.iter)

@HOOKS.register_module()
class GarbageHandler(HookBase):
    def __init__(self, interval=150, disable_auto=True, empty_cache=False):
        self.interval = interval
        self.disable_auto = disable_auto
        self.empty_cache = empty_cache
        self.iter = 1

    def before_train(self):
        if self.disable_auto:
            gc.disable()
            self.trainer.logger.info("Disable automatic garbage collection")

    def before_epoch(self):
        self.iter = 1

    def after_step(self):
        if self.iter % self.interval == 0:
            gc.collect()
            if self.empty_cache:
                torch.cuda.empty_cache()
            self.trainer.logger.info("Garbage collected")
        self.iter += 1

    def after_train(self):
        gc.collect()
        torch.cuda.empty_cache()

@HOOKS.register_module()
class GradientNormLogger(HookBase):
    """
    Hook to log gradient norms to Weights & Biases (wandb).
    
    This hook computes the gradient norm of model parameters and logs it to wandb
    after each training step. It supports different norm types (L1, L2, etc.) and
    can optionally log per-layer gradient norms for detailed monitoring.
    
    Args:
        norm_type (float): Type of norm to compute (default: 2.0 for L2 norm)
        log_per_layer (bool): Whether to log gradient norms for individual layers (default: False)
        log_frequency (int): Log gradient norms every N steps (default: 1)
        prefix (str): Prefix for wandb logging keys (default: "grad_norm")
    """
    
    def __init__(self, norm_type=2.0, log_per_layer=False, log_frequency=1, prefix="grad_norm"):
        self.norm_type = norm_type
        self.log_per_layer = log_per_layer
        self.log_frequency = log_frequency
        self.prefix = prefix
        self.step_count = 0
    
    def _compute_grad_norm(self, parameters, norm_type=2.0):
        """Compute gradient norm for given parameters."""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        grads = [p.grad for p in parameters if p.grad is not None]
        if len(grads) == 0:
            return torch.tensor(0.0)
        
        if norm_type == float('inf'):
            total_norm = max(g.abs().max() for g in grads)
        else:
            total_norm = torch.norm(
                torch.stack([torch.norm(g, norm_type) for g in grads]), 
                norm_type
            )
        
        return total_norm
    
    def after_step(self):
        """Log gradient norms after each training step."""
        self.step_count += 1
        
        # Only log at specified frequency
        if self.step_count % self.log_frequency != 0:
            return
        
        # Only log if wandb writer is available
        if not hasattr(self.trainer, 'writer') or self.trainer.writer is None:
            return
        
        # Check if we're using wandb (not tensorboard)
        if not hasattr(self.trainer.writer, 'add_scalar'):
            return
        
        # Get current iteration for logging
        current_iter = self.trainer.comm_info.get("iter", 0) + 1
        current_epoch = self.trainer.epoch + 1
        global_step = (current_epoch - 1) * len(self.trainer.train_loader) + current_iter
        
        # Compute total gradient norm
        total_grad_norm = self._compute_grad_norm(
            self.trainer.model.parameters(), 
            self.norm_type
        )
        
        
        # Log total gradient norm
        self.trainer.writer.add_scalar(
            f"{self.prefix}/total", 
            total_grad_norm.item(), 
            global_step
        )
        
        # Optionally log per-layer gradient norms
        if self.log_per_layer:
            self._log_per_layer_grad_norms(global_step)
    
    def _log_per_layer_grad_norms(self, global_step):
        """Log gradient norms for individual layers."""
        # Get model without DDP wrapper if present
        model = self.trainer.model
        if hasattr(model, 'module'):
            model = model.module
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad, self.norm_type)
                # Clean up layer name for wandb logging
                clean_name = name.replace('.', '_')
                self.trainer.writer.add_scalar(
                    f"{self.prefix}/layers/{clean_name}", 
                    grad_norm.item(), 
                    global_step
                )
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(norm_type={self.norm_type}, "
                f"log_per_layer={self.log_per_layer}, "
                f"log_frequency={self.log_frequency}, "
                f"prefix='{self.prefix}')")


DTYPE_TO_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
}

@HOOKS.register_module()
class DtypeOverrider(HookBase):
    """
    Hook that forces specific layers to use a specified dtype for computation.
    
    This hook can:
    1. Override forward methods to force computation in a specific dtype
    2. Register forward/backward hooks to convert parameters/gradients
    
    This is particularly useful for forcing fp32 computation in precision-sensitive
    layers like LayerNorm, even when using mixed precision training.
    
    Args:
        patterns (list): List of regex patterns to match layer names
        class_patterns (list): List of regex patterns to match class names
        dtype (torch.dtype): Data type to use for computation (default: torch.float32)
        methods_to_override (list): List of methods to override (default: ['forward'])
        override_parameters (bool): Whether to override parameters as well (default: False)
        verbose (bool): Whether to log detailed information about overridden layers
        check_interval (int): How often to check if parameters need to be cast back (default: 10)
    
    Example usage:
        hooks = [
            dict(type='DtypeOverrider',
                 patterns=['layer_norm', 'LayerNorm', 'norm'],
                 dtype=torch.float32,
                 override_parameters=True,
                 verbose=True)
        ]
    """
    
    def __init__(
        self, 
        patterns=None, 
        class_patterns=None,
        dtype="float32", 
        methods_to_override=None,
        override_parameters=False,
        verbose=False,
        check_interval=10
    ):
        self.patterns = patterns or []
        self.class_patterns = class_patterns or []
        self.dtype = DTYPE_TO_TORCH_DTYPE[dtype]
        self.methods_to_override = methods_to_override or ["forward"]
        self.override_parameters = override_parameters
        self.verbose = verbose
        self.check_interval = check_interval
        self.overridden_layers = []
        self.overridden_params = []
        self.step_counter = 0
        self.param_original_dtypes = {}
    
    def before_train(self):
        """Apply dtype overriding before training starts."""
        self._override_layers(self.trainer.model)
        
        if self.verbose:
            self.trainer.logger.info(f"Overridden {len(self.overridden_layers)} layers to use {self.dtype}:")
            for name in self.overridden_layers:
                self.trainer.logger.info(f"  - {name}")
            
            if self.override_parameters:
                self.trainer.logger.info(f"Overridden {len(self.overridden_params)} parameters to use {self.dtype}")
    
    def _should_override(self, name, module):
        """Check if this module should be overridden based on name or class."""
        # Check if module name matches any pattern
        name_match = any(re.search(pattern, name) for pattern in self.patterns)
        
        # Check if class name matches any pattern
        class_match = any(re.search(pattern, module.__class__.__name__) for pattern in self.class_patterns)
        
        return name_match or class_match
    
    def _override_layers(self, module, prefix=''):
        """Recursively override layer methods and parameters to force dtype."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this layer should be overridden
            if self._should_override(full_name, child):
                # Override specified methods
                for method_name in self.methods_to_override:
                    if hasattr(child, method_name):
                        if self.verbose:
                            self.trainer.logger.info(f"Wrapping {full_name}.{method_name} to force {self.dtype}")
                        original_method = getattr(child, method_name)
                        
                        # Create a wrapped method that forces dtype
                        wrapped_method = self._make_dtype_wrapper(original_method, self.dtype)
                        
                        # Replace the original method
                        setattr(child, method_name, wrapped_method)
                        
                        self.overridden_layers.append(f"{full_name}.{method_name}")
                
                # Override parameters if requested
                if self.override_parameters:
                    for param_name, param in child.named_parameters(recurse=False):
                        param_full_name = f"{full_name}.{param_name}"
                        
                        if param.dtype != self.dtype:
                            if self.verbose:
                                self.trainer.logger.info(f"Overriding {param_full_name} to {self.dtype}")
                            self.param_original_dtypes[param_full_name] = param.data.dtype
                            param.data = param.data.to(self.dtype)
                            self.overridden_params.append(param_full_name)
            
            # Recursively apply to child modules
            self._override_layers(child, full_name)
    
    def _make_dtype_wrapper(self, original_method, dtype):
        """Create a wrapper function that forces computation in specified dtype."""
        def wrapped_method(*args, **kwargs):
            # Handle different argument patterns for different methods
            if args and isinstance(args[0], torch.Tensor):
                # Most common case: first arg is input tensor
                orig_dtype = args[0].dtype
                args_cast = [args[0].to(dtype)] + list(args[1:])
                output = original_method(*args_cast, **kwargs)
                
                # Convert output back to original dtype
                if isinstance(output, torch.Tensor):
                    output = output.to(orig_dtype)
                elif isinstance(output, tuple):
                    output = tuple(x.to(orig_dtype) if isinstance(x, torch.Tensor) else x for x in output)
                elif isinstance(output, list):
                    output = [x.to(orig_dtype) if isinstance(x, torch.Tensor) else x for x in output]
                
                return output
            else:
                # If no tensor input, just call original method
                return original_method(*args, **kwargs)
                
        return wrapped_method
    
    def __repr__(self):
        patterns_str = ', '.join(self.patterns)
        return (f"{self.__class__.__name__}(patterns=[{patterns_str}], "
                f"dtype={self.dtype})")


@HOOKS.register_module()
class LogitEntropyLogger(HookBase):
    """
    Hook to calculate and log entropy of teacher logits using Sonata's temperature schedule.
    
    This hook computes the entropy of teacher logits (after softmax) to help monitor
    the confidence and uncertainty of model predictions during training.
    High entropy indicates uncertain predictions, while low entropy indicates confident predictions.
    
    Args:
        logits_key (str): Key in model output dict to find the teacher logits (default: "teacher_logits")
        log_frequency (int): How often to log entropy values (default: 1)
        prefix (str): Prefix for logging keys (default: "entropy")
        reduction (str): How to reduce entropy values ('mean', 'none', etc.) (default: 'mean')
        log_per_class (bool): Whether to log per-class entropy (default: False)
        default_temperature (float): Default temperature to use if Sonata not found (default: 0.07)
    """
    
    def __init__(
        self,
        logits_key="teacher_logits",
        log_frequency=1,
        prefix="entropy",
        reduction='mean',
        log_per_class=False,
        default_temperature=0.07
    ):
        self.logits_key = logits_key
        self.log_frequency = log_frequency
        self.prefix = prefix
        self.reduction = reduction
        self.log_per_class = log_per_class
        self.default_temperature = default_temperature
        self.step_count = 0
        self.sonata_model = None
    
    def before_train(self):
        """Initialize entropy tracking and find Sonata model."""
        self.trainer.logger.info(f"Logging entropy of '{self.logits_key}' with prefix '{self.prefix}'")
        
        # Find Sonata module in the model
        if hasattr(self.trainer.model, 'module'):
            model = self.trainer.model.module  # Unwrap DDP
        else:
            model = self.trainer.model
            
        # Look for Sonata module
        if hasattr(model, 'sonata'):
            self.sonata_model = model.sonata
            self.trainer.logger.info("Found Sonata model for temperature scheduling")
        elif isinstance(model, nn.ModuleDict) and 'sonata' in model:
            self.sonata_model = model['sonata']
            self.trainer.logger.info("Found Sonata model in ModuleDict for temperature scheduling")
        elif hasattr(model, '__class__') and 'Sonata' in model.__class__.__name__:
            self.sonata_model = model
            self.trainer.logger.info("Model itself is a Sonata model")
        else:
            self.trainer.logger.warning(f"Couldn't find Sonata model, using default temperature: {self.default_temperature}")
    
    def _get_sonata_temperature(self):
        """Get the current temperature from Sonata model."""
        if self.sonata_model is not None and hasattr(self.sonata_model, 'teacher_temp'):
            return self.sonata_model.teacher_temp
        return self.default_temperature
    
    def _calculate_entropy(self, logits):
        """Calculate entropy of probability distribution from logits."""
        # Get temperature from Sonata model
        temperature = self._get_sonata_temperature()
        
        # Log temperature if writer is available
        if hasattr(self.trainer, 'writer') and self.trainer.writer is not None:
            current_iter = self.trainer.comm_info.get("iter", 0) + 1
            current_epoch = self.trainer.epoch + 1
            global_step = (current_epoch - 1) * len(self.trainer.train_loader) + current_iter
            self.trainer.writer.add_scalar(f"{self.prefix}/temperature", temperature, global_step)
        
        # Apply temperature scaling and softmax to get probabilities
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        
        # Calculate entropy: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        if self.reduction == 'mean':
            return entropy.mean()
        elif self.reduction == 'sum':
            return entropy.sum()
        else:
            return entropy
    
    def after_step(self):
        """Calculate and log entropy after each step."""
        self.step_count += 1
        
        if self.step_count % self.log_frequency != 0:
            return
            
        # Check if model output contains the logits key
        if (not hasattr(self.trainer, 'comm_info') or 
            "model_output_dict" not in self.trainer.comm_info or
            self.logits_key not in self.trainer.comm_info["model_output_dict"]):
            return
            
        # Get logits from model output
        logits = self.trainer.comm_info["model_output_dict"][self.logits_key]
        
        # Calculate entropy
        entropy = self._calculate_entropy(logits)
        
        # Add temperature to model output for other hooks/modules
        self.trainer.comm_info["model_output_dict"]["temperature"] = self._get_sonata_temperature()
        
        # Log to storage for console output
        if isinstance(entropy, torch.Tensor) and entropy.numel() == 1:
            entropy_val = entropy.item()
            self.trainer.storage.put_scalar(f"{self.prefix}", entropy_val)
            
            # Add to iteration info
            if "iter_info" in self.trainer.comm_info:
                self.trainer.comm_info["iter_info"] += f"{self.prefix}: {entropy_val:.4f} "
        
        # Log to tensorboard/wandb if available
        if hasattr(self.trainer, 'writer') and self.trainer.writer is not None:
            current_iter = self.trainer.comm_info.get("iter", 0) + 1
            current_epoch = self.trainer.epoch + 1
            global_step = (current_epoch - 1) * len(self.trainer.train_loader) + current_iter
            
            self.trainer.writer.add_scalar(
                f"train_batch/{self.prefix}", 
                entropy.item() if entropy.numel() == 1 else entropy.mean().item(),
                global_step
            )
            
            # Log per-class entropy if requested and entropy is not already reduced
            if self.log_per_class and self.reduction == 'none' and entropy.dim() > 0:
                for i, ent in enumerate(entropy.mean(dim=0)):
                    self.trainer.writer.add_scalar(
                        f"train_batch/{self.prefix}_class_{i}", 
                        ent.item(),
                        global_step
                    )
    
    def after_epoch(self):
        """Log epoch average entropy."""
        if hasattr(self.trainer, 'writer') and self.trainer.writer is not None:
            if hasattr(self.trainer.storage, 'history') and self.prefix in self.trainer.storage.history():
                avg_entropy = self.trainer.storage.history(self.prefix).avg
                self.trainer.writer.add_scalar(
                    f"train/{self.prefix}",
                    avg_entropy,
                    self.trainer.epoch + 1
                )
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(logits_key='{self.logits_key}', "
                f"prefix='{self.prefix}')")


@HOOKS.register_module()
class PrototypeUsageLogger(HookBase):
    """
    Hook to monitor prototype utilization in clustering/tokenization models like Sonata.
    
    This hook tracks:
    1. How many prototypes are actually being used (by looking at argmax assignments)
    2. What percentage of prototypes are unused
    3. How many tokens are assigned to each active prototype on average
    
    Args:
        log_frequency (int): How often to log prototype usage (default: 10)
        prefix (str): Prefix for logging keys (default: "prototypes")
    """
    
    def __init__(
        self,
        log_frequency=10,
        prefix="prototypes"
    ):
        self.log_frequency = log_frequency
        self.prefix = prefix
        self.hook_handles = []
        self._step_counters = {}
    
    def before_train(self):
        """Register hooks on Sonata heads to capture prototype usage."""
        self.trainer.logger.info(f"Monitoring prototype usage with prefix '{self.prefix}'")
        
        # Access the model (unwrap DDP if needed)
        if hasattr(self.trainer.model, 'module'):
            model = self.trainer.model.module
        else:
            model = self.trainer.model
            
        # Register hooks on the model
        self._register_hooks(model)
    
    def _register_hooks(self, model):
        """Register hooks on Sonata heads to capture prototype usage."""
        # Clear previous hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
        # Find Sonata module
        sonata_module = None
        if hasattr(model, 'sonata'):
            sonata_module = model.sonata
        elif isinstance(model, torch.nn.ModuleDict) and 'sonata' in model:
            sonata_module = model['sonata']
        elif hasattr(model, '__class__') and 'Sonata' in model.__class__.__name__:
            sonata_module = model
            
        if not sonata_module:
            self.trainer.logger.warning("Could not find Sonata module for prototype monitoring")
            return
            
        # Register hooks on teacher heads
        if hasattr(sonata_module, 'teacher') and isinstance(sonata_module.teacher, torch.nn.ModuleDict):
            for head_name, head in sonata_module.teacher.items():
                if 'head' in head_name.lower():
                    self.trainer.logger.info(f"Registering prototype monitor on {head_name}")
                    hook = head.register_forward_hook(self._prototype_stats_hook(f"teacher/{head_name}"))
                    self.hook_handles.append(hook)
                    
        # Register hooks on student heads
        if hasattr(sonata_module, 'student') and isinstance(sonata_module.student, torch.nn.ModuleDict):
            for head_name, head in sonata_module.student.items():
                if 'head' in head_name.lower():
                    self.trainer.logger.info(f"Registering prototype monitor on {head_name}")
                    hook = head.register_forward_hook(self._prototype_stats_hook(f"student/{head_name}"))
                    self.hook_handles.append(hook)
    
    def _prototype_stats_hook(self, name):
        """Create a forward hook that calculates prototype statistics."""
        def hook_fn(module, input, output):
            if not module.training:
                return
            if output is None:
                return
                
            if name not in self._step_counters:
                self._step_counters[name] = 0
                
            self._step_counters[name] += 1
            
            if self._step_counters[name] % self.log_frequency != 0:
                return
            
            if isinstance(output, tuple):
                stats = {}
                for i, o in enumerate(output):
                    stats[f"output_{i}"] = self._get_stats(o)
            else:
                stats = self._get_stats(output)

            # Log to tensorboard/wandb if available
            if hasattr(self.trainer, 'writer') and self.trainer.writer is not None:
                current_iter = self.trainer.comm_info.get("iter", 0) + 1
                current_epoch = self.trainer.epoch + 1
                global_step = (current_epoch - 1) * len(self.trainer.train_loader) + current_iter
                
                # Log metrics
                for stat_name, stat_value in stats.items():
                    if isinstance(stat_value, dict):
                        for k, v in stat_value.items():
                            self.trainer.writer.add_scalar(
                                f"{self.prefix}/{name}/{stat_name}/{k}", 
                                v,
                                global_step
                            )
                    else:
                        self.trainer.writer.add_scalar(
                            f"{self.prefix}/{name}/{stat_name}", 
                            stat_value,
                            global_step
                        )
        
        return hook_fn

    def _get_stats(self, output):
        """Calculate statistics from output with proper distributed synchronization."""
        import torch.distributed as dist
        from pimm.utils.comm import get_world_size
        
        with torch.no_grad():
            # Get assignments by taking argmax of logits
            assignments = output.argmax(dim=-1)  # (tokens,)
            
            # Total number of prototypes
            total_prototypes = output.shape[-1]
            
            # Count tokens per prototype locally
            local_counts = torch.bincount(assignments, minlength=total_prototypes).float()
            
            # Synchronize counts across all GPUs
            if get_world_size() > 1:
                dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
            
            global_counts = local_counts
            total_tokens = global_counts.sum().item()
            
            # Calculate global usage metrics
            used_mask = global_counts > 0
            used_count = used_mask.sum().item()
            unused_count = total_prototypes - used_count
            unused_percent = (unused_count / total_prototypes) * 100
            
            # Calculate tokens per prototype (global average)
            tokens_per_prototype = total_tokens / used_count if used_count > 0 else 0
            
            # Calculate entropy of assignment distribution (global)
            probs = global_counts / total_tokens if total_tokens > 0 else global_counts
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            
            # Create stats dictionary
            stats = {
                "used_count": used_count,
                "unused_percent": unused_percent,
                "tokens_per_prototype": tokens_per_prototype,
                "assignment_entropy": entropy.item()
            }
        return stats
    
    def after_train(self):
        """Clean up hooks when training is done."""
        for handle in self.hook_handles:
            handle.remove()
    
    def __repr__(self):
        return f"{self.__class__.__name__}(log_frequency={self.log_frequency}, prefix='{self.prefix}')"


@HOOKS.register_module()
class FeatureStdMonitor(HookBase):
    """
    Hook to monitor the standard deviation of feature vectors in student and teacher models.
    
    This is useful for tracking feature collapse and ensuring features remain diverse
    during training. The hook uses forward hooks to compute stats directly during
    the forward pass, avoiding storing large feature tensors in memory.
    
    Args:
        log_frequency (int): How often to log feature statistics (default: 10)
        prefix (str): Prefix for logging keys (default: "feature_std")
        monitor_student (bool): Whether to monitor student model features (default: True)
        monitor_teacher (bool): Whether to monitor teacher model features (default: True)
        track_channels (bool): Whether to track per-channel statistics (default: False)
    """
    
    def __init__(
        self,
        log_frequency=10,
        prefix="feature_std",
        monitor_student=True,
        monitor_teacher=True,
        track_channels=False
    ):
        self.log_frequency = log_frequency
        self.prefix = prefix
        self.monitor_student = monitor_student
        self.monitor_teacher = monitor_teacher
        self.track_channels = track_channels
        self.step_count = 0
        self.hook_handles = []
    
    def before_train(self):
        """Register forward hooks to capture feature statistics."""
        self.trainer.logger.info(f"Monitoring feature statistics with prefix '{self.prefix}'")
        
        # Access the model (unwrap DDP if needed)
        if hasattr(self.trainer.model, 'module'):
            model = self.trainer.model.module
        else:
            model = self.trainer.model
            
        # Find Sonata modules to monitor
        self._register_sonata_hooks(model)
    
    def _register_sonata_hooks(self, model):
        """Register hooks on student and teacher modules to capture feature stats."""
        # Clear previous hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
        # Check if model is Sonata or contains Sonata
        sonata_module = None
        if hasattr(model, 'sonata'):
            sonata_module = model.sonata
        elif isinstance(model, torch.nn.ModuleDict) and 'sonata' in model:
            sonata_module = model['sonata']
        elif hasattr(model, '__class__') and 'Sonata' in model.__class__.__name__:
            sonata_module = model
            
        if not sonata_module:
            self.trainer.logger.warning("Could not find Sonata module for feature monitoring")
            return
        
        # Register hooks on teacher backbone
        if self.monitor_teacher and hasattr(sonata_module, 'teacher') and 'backbone' in sonata_module.teacher:
            self.trainer.logger.info("Registering feature monitor on teacher backbone")
            hook = sonata_module.teacher['backbone'].register_forward_hook(
                self._feature_stats_hook('teacher')
            )
            self.hook_handles.append(hook)
        
        # Register hooks on student backbone
        if self.monitor_student and hasattr(sonata_module, 'student') and 'backbone' in sonata_module.student:
            self.trainer.logger.info("Registering feature monitor on student backbone")
            hook = sonata_module.student['backbone'].register_forward_hook(
                self._feature_stats_hook('student')
            )
            self.hook_handles.append(hook)
    
    def _feature_stats_hook(self, module_name):
        """Create a forward hook function that captures feature statistics."""
        def hook_fn(module, input, output):
            if not module.training:
                return

            if not hasattr(self, '_step_counter'):
                self._step_counter = {}
            if module_name not in self._step_counter:
                self._step_counter[module_name] = 0
            
            self._step_counter[module_name] += 1
            if self._step_counter[module_name] % self.log_frequency != 0:
                return
            
            if hasattr(output, 'feat'):
                features = output.feat
            elif isinstance(output, torch.Tensor):
                features = output
            else:
                return
                
            with torch.no_grad():
                import torch.distributed as dist
                from pimm.utils.comm import get_world_size
                
                features_flat = features.float()
                local_n = torch.tensor([features_flat.numel()], device=features.device, dtype=torch.float64)
                local_sum = features_flat.sum().to(torch.float64)
                local_sum_sq = (features_flat ** 2).sum().to(torch.float64)
                
                if get_world_size() > 1:
                    dist.all_reduce(local_n)
                    dist.all_reduce(local_sum)
                    dist.all_reduce(local_sum_sq)
                
                global_mean = local_sum / local_n
                global_var = (local_sum_sq / local_n) - (global_mean ** 2)
                global_std = torch.sqrt(global_var.clamp(min=0)).item()
                
                batch_std = torch.std(features, dim=1).mean().item()
                
                local_channel_n = torch.tensor([features.shape[0]], device=features.device, dtype=torch.float64)
                local_channel_sum = features.sum(dim=0).to(torch.float64)
                local_channel_sum_sq = (features ** 2).sum(dim=0).to(torch.float64)
                
                if get_world_size() > 1:
                    dist.all_reduce(local_channel_n)
                    dist.all_reduce(local_channel_sum)
                    dist.all_reduce(local_channel_sum_sq)
                
                channel_mean = local_channel_sum / local_channel_n
                channel_var = (local_channel_sum_sq / local_channel_n) - (channel_mean ** 2)
                channel_std = torch.sqrt(channel_var.clamp(min=0))
                
                channel_mean_std = channel_std.mean().item()
                channel_min_std = channel_std.min().item()
                channel_max_std = channel_std.max().item()
                
                stats = {
                    "global_std": global_std,
                    "batch_std": batch_std,
                    "channel_mean_std": channel_mean_std,
                    "channel_min_std": channel_min_std,
                    "channel_max_std": channel_max_std
                }
                        
            # Log to tensorboard/wandb if available
            if hasattr(self.trainer, 'writer') and self.trainer.writer is not None:
                current_iter = self.trainer.comm_info.get("iter", 0) + 1
                current_epoch = self.trainer.epoch + 1
                global_step = (current_epoch - 1) * len(self.trainer.train_loader) + current_iter
                
                # Log metrics
                for stat_name, stat_value in stats.items():
                    self.trainer.writer.add_scalar(
                        f"{self.prefix}/{module_name}/{stat_name}", 
                        stat_value,
                        global_step
                    )
                
                # Log per-channel std if requested
                if self.track_channels:
                    for i, std_val in enumerate(channel_std):
                        self.trainer.writer.add_scalar(
                            f"{self.prefix}/{module_name}/channel_{i}_std", 
                            std_val.item(),
                            global_step
                        )
        
        return hook_fn
    
    def after_train(self):
        """Clean up hooks."""
        for handle in self.hook_handles:
            handle.remove()
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"log_frequency={self.log_frequency}, "
                f"prefix='{self.prefix}')")


@HOOKS.register_module()
class ParameterCounter(HookBase):
    """
    Hook to count and log parameters in each module at the start of training.
    
    This hook provides detailed information about model architecture including:
    - Total parameters and trainable parameters
    - Parameter count breakdown by module
    - Memory footprint estimation
    
    Args:
        show_details (bool): Whether to show per-module breakdown (default: True)
        show_gradients (bool): Whether to show gradient information (default: True)
        sort_by_params (bool): Whether to sort modules by parameter count (default: True)
        min_params (int): Minimum parameters to show a module (default: 0)
    """
    
    def __init__(self, show_details=True, show_gradients=True, sort_by_params=True, min_params=0):
        self.show_details = show_details
        self.show_gradients = show_gradients
        self.sort_by_params = sort_by_params
        self.min_params = min_params
    
    def before_train(self):
        """Count and log parameter information before training starts."""
        self.trainer.logger.info("=" * 80)
        self.trainer.logger.info("MODEL PARAMETER ANALYSIS")
        self.trainer.logger.info("=" * 80)
        
        # Get the model (unwrap DDP if present)
        model = self.trainer.model
        if hasattr(model, 'module'):
            model = model.module
        
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        # Estimate memory footprint (assuming float32)
        memory_mb = (total_params * 4) / (1024 * 1024)
        
        # Log overall statistics
        self.trainer.logger.info(f"Total Parameters: {total_params:,}")
        self.trainer.logger.info(f"Trainable Parameters: {trainable_params:,}")
        self.trainer.logger.info(f"Non-trainable Parameters: {non_trainable_params:,}")
        self.trainer.logger.info(f"Estimated Memory (params only): {memory_mb:.2f} MB")
        
        if self.show_details:
            self.trainer.logger.info("-" * 80)
            self.trainer.logger.info("PARAMETER BREAKDOWN BY MODULE")
            self.trainer.logger.info("-" * 80)
            
            module_stats = []
            
            # Collect statistics for each named module
            for name, module in model.named_modules():
                if name == "":  # Skip root module
                    continue
                    
                # Count parameters in this specific module (not children)
                module_params = sum(p.numel() for p in module.parameters(recurse=False))
                module_trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
                
                if module_params >= self.min_params:
                    module_stats.append({
                        'name': name,
                        'type': module.__class__.__name__,
                        'total_params': module_params,
                        'trainable_params': module_trainable,
                        'non_trainable_params': module_params - module_trainable
                    })
            
            # Sort by parameter count if requested
            if self.sort_by_params:
                module_stats.sort(key=lambda x: x['total_params'], reverse=True)
            
            # Display module breakdown
            self.trainer.logger.info(f"{'Module Name':<50} {'Type':<20} {'Params':<12} {'Trainable':<12}")
            self.trainer.logger.info("-" * 94)
            
            for stats in module_stats:
                if stats['total_params'] > 0:
                    self.trainer.logger.info(
                        f"{stats['name']:<50} {stats['type']:<20} "
                        f"{stats['total_params']:<12,} {stats['trainable_params']:<12,}"
                    )
            
            # Show largest modules summary
            if module_stats:
                self.trainer.logger.info("-" * 80)
                self.trainer.logger.info("TOP 10 LARGEST MODULES")
                self.trainer.logger.info("-" * 80)
                
                top_modules = sorted(module_stats, key=lambda x: x['total_params'], reverse=True)[:10]
                for i, stats in enumerate(top_modules, 1):
                    if stats['total_params'] > 0:
                        percent = (stats['total_params'] / total_params) * 100
                        self.trainer.logger.info(
                            f"{i:2d}. {stats['name']:<40} {stats['total_params']:>10,} params ({percent:5.1f}%)"
                        )
        
        if self.show_gradients:
            self.trainer.logger.info("-" * 80)
            self.trainer.logger.info("GRADIENT INFORMATION")
            self.trainer.logger.info("-" * 80)
            
            grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            no_grad_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            
            if grad_params > 0:
                self.trainer.logger.info(f"Parameters requiring gradients: {grad_params:,}")
            if no_grad_params > 0:
                self.trainer.logger.info(f"Parameters NOT requiring gradients: {no_grad_params:,}")
                
                # Show which modules have frozen parameters
                frozen_modules = []
                for name, module in model.named_modules():
                    frozen_in_module = sum(p.numel() for p in module.parameters(recurse=False) if not p.requires_grad)
                    if frozen_in_module > 0:
                        frozen_modules.append((name, frozen_in_module))
                
                if frozen_modules:
                    self.trainer.logger.info("Modules with frozen parameters:")
                    for name, count in frozen_modules:
                        self.trainer.logger.info(f"  {name}: {count:,} frozen params")
        
        self.trainer.logger.info("=" * 80)
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(show_details={self.show_details}, "
                f"show_gradients={self.show_gradients})")


@HOOKS.register_module()
class AttentionMaskAnnealingHook(HookBase):
    """
    Hook to update attention mask annealing progress during training.

    For use in the Panda detector.
    
    This hook is designed for models with dynamic attention masks that gradually
    anneal during training (e.g., Mask3Former decoder). It:
    1. Updates annealing progress at each training step
    2. Logs annealing factors per layer to wandb/tensorboard
    3. Reports when annealing completes
    
    Args:
        log_frequency (int): How often to log annealing factors (default: 100)
        log_per_layer (bool): Whether to log per-layer annealing factors (default: False)
        prefix (str): Prefix for logging keys (default: "anneal")
    
    Example usage in config:
        hooks = [
            dict(type="AttentionMaskAnnealingHook", 
                 log_frequency=100,
                 log_per_layer=True),
        ]
    """
    
    def __init__(self, log_frequency=100, log_per_layer=False, prefix="anneal"):
        self.log_frequency = log_frequency
        self.log_per_layer = log_per_layer
        self.prefix = prefix
        self.step_count = 0
        self.annealing_complete = False
        self.has_annealing = False
    
    def before_train(self):
        """Check if model supports annealing and log initial state."""
        # Get model without DDP wrapper if present
        model = self.trainer.model
        if hasattr(model, 'module'):
            model = model.module
        
        # Check if model has update_anneal_step method
        if hasattr(model, 'update_anneal_step'):
            self.has_annealing = True
            self.trainer.logger.info("Attention mask annealing enabled")
            
            # Log annealing configuration if available
            if hasattr(model, 'decoder') and hasattr(model.decoder, 'attn_mask_anneal'):
                if model.decoder.attn_mask_anneal:
                    steps = model.decoder.attn_mask_anneal_steps
                    warmup = model.decoder.attn_mask_warmup_steps
                    progressive = model.decoder.attn_mask_progressive
                    delay = model.decoder.attn_mask_progressive_delay
                    
                    self.trainer.logger.info(
                        f"Annealing schedule: {steps} steps using cosine decay"
                    )
                    if warmup > 0:
                        self.trainer.logger.info(f"  Warmup: {warmup} steps before annealing starts")
                    if progressive and delay > 0:
                        self.trainer.logger.info(
                            f"  Progressive: {delay} steps delay between blocks "
                            f"(layer 0 starts at step {warmup}, layer N at step {warmup + delay * (len(model.decoder.blocks) - 1)})"
                        )
                    
                    # Log per-block warmup if available
                    if hasattr(model.decoder, 'blocks') and len(model.decoder.blocks) > 0:
                        warmup_steps = [b.attn_mask_warmup_steps for b in model.decoder.blocks]
                        if len(set(warmup_steps)) > 1:
                            self.trainer.logger.info(
                                f"  Per-block warmup: {warmup_steps}"
                            )
                else:
                    self.trainer.logger.info("Attention masks enabled (no annealing)")
                    self.has_annealing = False
        else:
            self.trainer.logger.info("Model does not support attention mask annealing")
            self.has_annealing = False
    
    def after_step(self):
        """Update annealing progress and optionally log statistics."""
        if not self.has_annealing:
            return
        
        self.step_count += 1
        
        # Get model
        model = self.trainer.model
        if hasattr(model, 'module'):
            model = model.module
        
        # Update annealing step
        if hasattr(model, 'update_anneal_step'):
            model.update_anneal_step(self.step_count)
        
        # Log annealing progress
        if self.step_count % self.log_frequency == 0:
            self._log_annealing_stats(model)
    
    def _log_annealing_stats(self, model):
        """Log annealing statistics to tensorboard/wandb."""
        # Try to access decoder blocks
        if not hasattr(model, 'decoder') or not hasattr(model.decoder, 'blocks'):
            return
        
        blocks = model.decoder.blocks
        if len(blocks) == 0:
            return
        
        # Get annealing factors from blocks
        anneal_factors = []
        for i, block in enumerate(blocks):
            if hasattr(block, 'get_anneal_factor'):
                factor = block.get_anneal_factor()
                anneal_factors.append(factor)
                
                # Log per-layer if requested
                if self.log_per_layer and self.trainer.writer is not None:
                    self.trainer.writer.add_scalar(
                        f"{self.prefix}/layer_{i}",
                        factor,
                        self.step_count
                    )
        
        if len(anneal_factors) == 0:
            return
        
        # Compute average annealing factor
        avg_factor = sum(anneal_factors) / len(anneal_factors)
        
        # Log average factor
        self.trainer.storage.put_scalar(f"{self.prefix}_factor", avg_factor)
        
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                f"{self.prefix}/average",
                avg_factor,
                self.step_count
            )
        
        # Report when annealing completes (factor drops below threshold)
        if not self.annealing_complete and avg_factor < 0.01:
            self.annealing_complete = True
            self.trainer.logger.info(
                f"Attention mask annealing completed at step {self.step_count} "
                f"(factor={avg_factor:.4f})"
            )
        
        # Log info message periodically
        if self.step_count % (self.log_frequency * 10) == 0:
            self.trainer.logger.info(
                f"Attention mask annealing factor: {avg_factor:.4f} "
                f"(step {self.step_count})"
            )
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"log_frequency={self.log_frequency}, "
            f"log_per_layer={self.log_per_layer}, "
            f"prefix='{self.prefix}')"
        )


@HOOKS.register_module()
class ResourceUtilizationLogger(HookBase):
    """
    Hook to log GPU and CPU utilization metrics over time.

    Logs GPU memory usage, GPU utilization %, CPU usage %, and system memory
    to tensorboard/wandb for monitoring resource consumption during training.

    Args:
        log_frequency (int): Log metrics every N steps (default: 10)
        prefix (str): Prefix for logged metrics (default: "resources")
        log_per_gpu (bool): Log metrics per GPU or just local GPU (default: False)
        log_cpu (bool): Log CPU utilization (default: True)
        log_system_memory (bool): Log system RAM usage (default: True)
        per_process (bool): If True, report only this process's resource usage
            instead of system-wide metrics. Recommended for shared nodes where
            other jobs may be running. (default: True)
    """

    def __init__(
        self,
        log_frequency=10,
        prefix="resources",
        log_per_gpu=False,
        log_cpu=True,
        log_system_memory=True,
        per_process=True,
    ):
        self.log_frequency = log_frequency
        self.prefix = prefix
        self.log_per_gpu = log_per_gpu
        self.log_cpu = log_cpu
        self.log_system_memory = log_system_memory
        self.per_process = per_process
        self.step_count = 0
        self._pynvml_available = False
        self._psutil_available = False
        self._nvml_initialized = False
        self._process = None

    def before_train(self):
        # try importing optional dependencies
        try:
            import psutil
            self._psutil_available = True
            if self.per_process:
                self._process = psutil.Process()
                # prime cpu_percent so the first real call returns meaningful data
                self._process.cpu_percent(interval=None)
        except ImportError:
            self.trainer.logger.warning(
                "psutil not available - CPU metrics will not be logged"
            )

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_available = True
            self._nvml_initialized = True
        except (ImportError, Exception):
            self.trainer.logger.info(
                "pynvml not available - using torch.cuda for GPU metrics"
            )

        if self.per_process:
            self.trainer.logger.info(
                "ResourceUtilizationLogger: per_process=True — reporting "
                "only this process's CPU/RAM and torch.cuda GPU memory"
            )

    def after_train(self):
        if self._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _get_gpu_metrics(self):
        """Get GPU memory and utilization metrics."""
        metrics = {}

        if not torch.cuda.is_available():
            return metrics

        local_rank = comm.get_local_rank()

        if self.per_process:
            # Per-process GPU metrics via torch.cuda (only our allocations)
            metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(local_rank) / 1e9
            metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(local_rank) / 1e9
            metrics["gpu_memory_max_allocated_gb"] = torch.cuda.max_memory_allocated(local_rank) / 1e9
            metrics["gpu_memory_max_reserved_gb"] = torch.cuda.max_memory_reserved(local_rank) / 1e9
            return metrics

        # System-wide GPU metrics (original behavior)
        if self._pynvml_available:
            try:
                import pynvml
                if self.log_per_gpu:
                    device_count = torch.cuda.device_count()
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics[f"gpu{i}_memory_used_gb"] = mem_info.used / 1e9
                        metrics[f"gpu{i}_memory_total_gb"] = mem_info.total / 1e9
                        metrics[f"gpu{i}_memory_pct"] = 100.0 * mem_info.used / mem_info.total
                        metrics[f"gpu{i}_utilization_pct"] = util.gpu
                else:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics["gpu_memory_used_gb"] = mem_info.used / 1e9
                    metrics["gpu_memory_total_gb"] = mem_info.total / 1e9
                    metrics["gpu_memory_pct"] = 100.0 * mem_info.used / mem_info.total
                    metrics["gpu_utilization_pct"] = util.gpu
            except Exception:
                pass

        # fallback or additional torch.cuda metrics
        if self.log_per_gpu:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                if f"gpu{i}_memory_used_gb" not in metrics:
                    metrics[f"gpu{i}_memory_allocated_gb"] = torch.cuda.memory_allocated(i) / 1e9
                    metrics[f"gpu{i}_memory_reserved_gb"] = torch.cuda.memory_reserved(i) / 1e9
        else:
            if "gpu_memory_used_gb" not in metrics:
                metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(local_rank) / 1e9
                metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(local_rank) / 1e9
                # max memory for peak tracking
                metrics["gpu_memory_max_allocated_gb"] = torch.cuda.max_memory_allocated(local_rank) / 1e9

        return metrics

    def _get_cpu_metrics(self):
        """Get CPU and system memory metrics."""
        metrics = {}

        if not self._psutil_available:
            return metrics

        import psutil
        import os

        if self.per_process and self._process is not None:
            # Per-process metrics
            if self.log_cpu:
                # cpu_percent returns total across all cores for this process
                # e.g. 400% means 4 cores fully used
                proc_cpu = self._process.cpu_percent(interval=None)
                metrics["process_cpu_percent"] = proc_cpu
                metrics["process_num_threads"] = self._process.num_threads()
                # Approximate number of cores used by this process
                metrics["process_cpu_cores"] = proc_cpu / 100.0

            if self.log_system_memory:
                mem_info = self._process.memory_info()
                metrics["process_rss_gb"] = mem_info.rss / 1e9
                metrics["process_vms_gb"] = mem_info.vms / 1e9

            return metrics

        # System-wide metrics (original behavior)
        if self.log_cpu:
            # overall cpu percent (averaged across all cores)
            metrics["cpu_percent"] = psutil.cpu_percent(interval=None)

            # load average - more useful for HPC nodes
            # load avg / num_cpus gives utilization ratio (>1 means oversubscribed)
            load1, load5, load15 = os.getloadavg()
            num_cpus = psutil.cpu_count(logical=True)
            metrics["load_avg_1min"] = load1
            metrics["load_avg_5min"] = load5
            metrics["load_avg_15min"] = load15
            metrics["load_per_cpu_1min"] = load1 / num_cpus  # ~1.0 = fully utilized
            metrics["num_cpus"] = num_cpus

            # per-cpu utilization stats to see distribution
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
            if per_cpu:
                metrics["cpu_max_core_pct"] = max(per_cpu)
                metrics["cpu_min_core_pct"] = min(per_cpu)
                active_cores = [c for c in per_cpu if c > 5.0]  # cores > 5% usage
                metrics["cpu_active_cores"] = len(active_cores)
                if active_cores:
                    metrics["cpu_active_avg_pct"] = sum(active_cores) / len(active_cores)

        if self.log_system_memory:
            mem = psutil.virtual_memory()
            metrics["system_memory_used_gb"] = mem.used / 1e9
            metrics["system_memory_total_gb"] = mem.total / 1e9
            metrics["system_memory_pct"] = mem.percent

        return metrics

    def after_step(self):
        self.step_count += 1

        if self.step_count % self.log_frequency != 0:
            return

        if not is_main_process():
            return

        if not hasattr(self.trainer, 'writer') or self.trainer.writer is None:
            return

        current_iter = self.trainer.comm_info.get("iter", 0) + 1
        current_epoch = self.trainer.epoch + 1
        global_step = (current_epoch - 1) * len(self.trainer.train_loader) + current_iter

        # collect metrics
        metrics = {}
        metrics.update(self._get_gpu_metrics())
        metrics.update(self._get_cpu_metrics())

        # log to writer
        for key, value in metrics.items():
            self.trainer.writer.add_scalar(
                f"{self.prefix}/{key}",
                value,
                global_step
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"log_frequency={self.log_frequency}, "
            f"prefix='{self.prefix}', "
            f"log_per_gpu={self.log_per_gpu}, "
            f"log_cpu={self.log_cpu}, "
            f"log_system_memory={self.log_system_memory}, "
            f"per_process={self.per_process})"
        )
