"""Garbage collection and resource utilization hooks."""

import gc

import torch

import pimm.utils.comm as comm
from pimm.utils.comm import is_main_process

from .builder import HOOKS
from .default import HookBase


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
                "ResourceUtilizationLogger: per_process=True - reporting "
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
        metrics = {}

        if not self._psutil_available:
            return metrics

        import os

        import psutil

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
