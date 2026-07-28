"""Garbage collection and resource utilization hooks."""

import gc

import torch

import pimm.utils.comm as comm
from pimm.observability import structured_logger as sl
from pimm.utils.comm import is_main_process

from .builder import HOOKS
from .default import HookBase


@HOOKS.register_module()
class GarbageHandler(HookBase):
    """Control Python garbage collection (and CUDA cache) on a step cadence.

    In ``before_train`` it optionally disables automatic garbage collection
    (``disable_auto``) to avoid unpredictable GC pauses during training. It
    resets a per-epoch counter in ``before_epoch`` and, in ``after_step``, runs
    ``gc.collect()`` every ``interval`` steps (also emptying the CUDA cache when
    ``empty_cache=True``). ``after_train`` performs a final collect and CUDA
    cache empty. Registered as ``GarbageHandler``.

    Args:
        interval (int): Run a manual ``gc.collect()`` every this many steps.
            Defaults to ``150``.
        disable_auto (bool): Disable Python's automatic GC at train start so
            collection happens only at the controlled interval. Defaults to
            ``True``.
        empty_cache (bool): Also call ``torch.cuda.empty_cache()`` at each
            interval collect. Defaults to ``False``.

    Example:
        Add to ``cfg.hooks``; it takes over garbage collection so it happens on a
        fixed cadence rather than unpredictably mid-step:

        .. code-block:: python

            hooks = [dict(type="GarbageHandler", interval=150)]
            # → disables automatic GC in before_train and runs gc.collect() every
            #   150 steps (also torch.cuda.empty_cache() when empty_cache=True);
            #   final collect + cache empty in after_train
    """

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
            sl.add_step_tag("garbage_collection")
            with sl.log_trace_span("garbage_collection"):
                gc.collect()
                if self.empty_cache:
                    torch.cuda.empty_cache()
            self.trainer.logger.info("Garbage collected")
        self.iter += 1

    def after_train(self):
        with sl.log_trace_span("garbage_collection"):
            gc.collect()
            torch.cuda.empty_cache()

@HOOKS.register_module()
class ResourceUtilizationLogger(HookBase):
    """Log GPU and CPU/RAM utilization to the writer over the course of training.

    In ``before_train`` it probes optional dependencies (``psutil`` for CPU/RAM,
    ``pynvml`` for system-wide GPU stats), warning/falling back to
    ``torch.cuda`` when unavailable. Runs in ``after_step`` every
    ``log_frequency`` steps on rank 0: collects GPU memory (and, system-wide,
    utilization %) plus CPU and system/process memory metrics and writes them to
    the writer under ``{prefix}/...``. ``after_train`` shuts down NVML if it was
    initialized. Registered as ``ResourceUtilizationLogger``.

    Args:
        log_frequency (int): Log metrics every this many steps. Defaults to
            ``10``.
        prefix (str): Namespace prefix for the logged keys. Defaults to
            ``"resources"``.
        log_per_gpu (bool): In system-wide mode, log metrics for every visible
            GPU instead of just the local one. Defaults to ``False``.
        log_cpu (bool): Log CPU utilization metrics. Defaults to ``True``.
        log_system_memory (bool): Log RAM usage metrics. Defaults to ``True``.
        per_process (bool): If ``True``, report only this process's CPU/RAM and
            ``torch.cuda`` GPU memory (recommended on shared nodes); if
            ``False``, report system-wide metrics (using ``pynvml`` for GPU when
            available). Defaults to ``True``.

    Note:
        Logs on rank 0 only and no-ops when ``trainer.writer`` is absent/``None``.
        ``per_process=True`` ignores ``log_per_gpu``/``pynvml`` and uses
        ``torch.cuda`` memory counters for this process.

    Example:
        Add to ``cfg.hooks``; every ``log_frequency`` steps (rank 0) it writes
        GPU/CPU/RAM utilization to the writer:

        .. code-block:: python

            hooks = [dict(type="ResourceUtilizationLogger", log_frequency=50)]
            # → every 50 steps writes "resources/gpu_memory_allocated_gb",
            #   "resources/gpu_memory_reserved_gb", "resources/process_cpu_percent",
            #   "resources/process_rss_gb", … to the writer (per-process by default)
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
