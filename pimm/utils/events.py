"""
Events Utils

Modified from Detectron2

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import datetime
import json
import logging
import os
import time
import torch
import numpy as np
import traceback
import sys
try:
    import wandb
except ImportError:
    wandb = None
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple, Dict, Any, Union
from collections import defaultdict
from contextlib import contextmanager

__all__ = [
    "get_event_storage",
    "JSONWriter",
    "TensorboardXWriter",
    "CommonMetricPrinter",
    "EventStorage",
    "ExceptionWriter",
    "WandbWriter",
    "WandbSummaryWriter",
]

_CURRENT_STORAGE_STACK = []
WANDB_STEP_METRIC = "train/global_step"


def _define_wandb_metrics(run):
    """Use optimizer progress as the x-axis without reusing W&B's history step."""
    run.define_metric(WANDB_STEP_METRIC, hidden=True)
    run.define_metric("*", step_metric=WANDB_STEP_METRIC)


def _wandb_log(run, data, train_step):
    """Append one chronological W&B row carrying its semantic training step."""
    train_step = int(train_step)
    if int(run.step) < train_step:
        # Fresh runs historically used the one-based optimizer iteration as
        # W&B's internal step. Add one metric-free anchor so new runs retain
        # that axis without coupling subsequent history writes to it.
        run.log(
            {WANDB_STEP_METRIC: train_step - 1},
            step=train_step - 1,
            commit=True,
        )
    payload = dict(data)
    payload[WANDB_STEP_METRIC] = train_step
    run.log(payload)


def get_event_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class:`EventStorage` is currently enabled.
    """
    assert len(
        _CURRENT_STORAGE_STACK
    ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]


class EventWriter:
    """
    Base class for writers that obtain events from :class:`EventStorage` and process them.
    """

    def write(self):
        """Flush newly available events to the concrete output backend."""
        raise NotImplementedError

    def close(self):
        """Release backend resources held by this writer."""
        pass


class JSONWriter(EventWriter):
    """
    Write scalars to a json file.
    It saves scalars as one json per line (instead of a big json) for easy parsing.
    Examples parsing such a json file:
    ::
        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 19,
            "loss": 1.9228371381759644,
            "loss_box_reg": 0.050025828182697296,
            "loss_classifier": 0.5316952466964722,
            "loss_mask": 0.7236229181289673,
            "loss_rpn_box": 0.0856662318110466,
            "loss_rpn_cls": 0.48198649287223816,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 39,
            "loss": 1.282649278640747,
            "loss_box_reg": 0.06222952902317047,
            "loss_classifier": 0.30682939291000366,
            "loss_mask": 0.6970193982124329,
            "loss_rpn_box": 0.038663312792778015,
            "loss_rpn_cls": 0.1471673548221588,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]
        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...
    """

    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): path to the json file. New data will be appended if the file exists.
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
        """
        self._file_handle = open(json_file, "a")
        self._window_size = window_size
        self._last_write = -1

    def write(self):
        """Append unwritten scalar metrics to the JSON-lines file."""
        storage = get_event_storage()
        to_save = defaultdict(dict)

        for k, (v, iter) in storage.latest_with_smoothing_hint(
            self._window_size
        ).items():
            # keep scalars that have not been written
            if iter <= self._last_write:
                continue
            to_save[iter][k] = v
        if len(to_save):
            all_iters = sorted(to_save.keys())
            self._last_write = max(all_iters)

        for itr, scalars_per_iter in to_save.items():
            scalars_per_iter["iteration"] = itr
            self._file_handle.write(json.dumps(scalars_per_iter, sort_keys=True) + "\n")
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def close(self):
        """Close the metrics file handle."""
        self._file_handle.close()


class TensorboardXWriter(EventWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir, **kwargs)
        self._last_write = -1

    def write(self):
        """Write scalars, images, and histograms to TensorBoard."""
        storage = get_event_storage()
        new_last_write = self._last_write
        for k, (v, iter) in storage.latest_with_smoothing_hint(
            self._window_size
        ).items():
            if iter > self._last_write:
                self._writer.add_scalar(k, v, iter)
                new_last_write = max(new_last_write, iter)
        self._last_write = new_last_write

        # storage.put_{image,histogram} is only meant to be used by
        # tensorboard writer. So we access its internal fields directly from here.
        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                self._writer.add_image(img_name, img, step_num)
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()

        if len(storage._histograms) >= 1:
            for params in storage._histograms:
                self._writer.add_histogram_raw(**params)
            storage.clear_histograms()

    def close(self):
        """Close the TensorBoard writer if it was created successfully."""
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()


class CommonMetricPrinter(EventWriter):
    """
    Print **common** metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    It also applies smoothing using a window of 20 elements.
    It's meant to print common metrics in common ways.
    To print something in more customized ways, please implement a similar printer by yourself.
    """

    def __init__(self, max_iter: Optional[int] = None, window_size: int = 20):
        """
        Args:
            max_iter: the maximum number of iterations to train.
                Used to compute ETA. If not given, ETA will not be printed.
            window_size (int): the losses will be median-smoothed by this window size
        """
        self.logger = logging.getLogger(__name__)
        self._max_iter = max_iter
        self._window_size = window_size
        self._last_write = (
            None  # (step, time) of last call to write(). Used to compute ETA
        )

    def _get_eta(self, storage) -> Optional[str]:
        """Estimate the remaining wall-clock time from stored iteration time."""
        if self._max_iter is None:
            return ""
        iteration = storage.iter
        try:
            eta_seconds = storage.history("time").median(1000) * (
                self._max_iter - iteration - 1
            )
            storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            return str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:
            # estimate eta on our own - more noisy
            eta_string = None
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                    iteration - self._last_write[0]
                )
                eta_seconds = estimate_iter_time * (self._max_iter - iteration - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())
            return eta_string

    def write(self):
        """Log a compact progress line with loss, timing, lr, and memory."""
        storage = get_event_storage()
        iteration = storage.iter
        if iteration == self._max_iter:
            # This hook only reports training progress (loss, ETA, etc) but not other data,
            # therefore do not write anything after training succeeds, even if this method
            # is called.
            return

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None
        try:
            iter_time = storage.history("time").global_avg()
        except KeyError:
            iter_time = None
        try:
            lr = "{:.5g}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        eta_string = self._get_eta(storage)

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            " {eta}iter: {iter}  {losses}  {time}{data_time}lr: {lr}  {memory}".format(
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                losses="  ".join(
                    [
                        "{}: {:.4g}".format(k, v.median(self._window_size))
                        for k, v in storage.histories().items()
                        if "loss" in k
                    ]
                ),
                time=(
                    "time: {:.4f}  ".format(iter_time) if iter_time is not None else ""
                ),
                data_time=(
                    "data_time: {:.4f}  ".format(data_time)
                    if data_time is not None
                    else ""
                ),
                lr=lr,
                memory=(
                    "max_mem: {:.0f}M".format(max_mem_mb)
                    if max_mem_mb is not None
                    else ""
                ),
            )
        )


class EventStorage:
    """
    The user-facing class that provides metric storage functionalities.
    In the future we may add support for storing / logging other types of data if needed.
    """

    def __init__(self, start_iter=0):
        """
        Args:
            start_iter (int): the iteration number to start with
        """
        self._history = defaultdict(AverageMeter)
        self._smoothing_hints = {}
        self._latest_scalars = {}
        self._iter = start_iter
        self._current_prefix = ""
        self._vis_data = []
        self._histograms = []

    # def put_image(self, img_name, img_tensor):
    #     """
    #     Add an `img_tensor` associated with `img_name`, to be shown on
    #     tensorboard.
    #     Args:
    #         img_name (str): The name of the image to put into tensorboard.
    #         img_tensor (torch.Tensor or numpy.array): An `uint8` or `float`
    #             Tensor of shape `[channel, height, width]` where `channel` is
    #             3. The image format should be RGB. The elements in img_tensor
    #             can either have values in [0, 1] (float32) or [0, 255] (uint8).
    #             The `img_tensor` will be visualized in tensorboard.
    #     """
    #     self._vis_data.append((img_name, img_tensor, self._iter))

    def put_scalar(self, name, value, n=1, smoothing_hint=False):
        """
        Add a scalar `value` to the `HistoryBuffer` associated with `name`.
        Args:
            smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.
                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal.
        """
        name = self._current_prefix + name
        history = self._history[name]
        history.update(value, n)
        self._latest_scalars[name] = (value, self._iter)

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            assert (
                existing_hint == smoothing_hint
            ), "Scalar {} was put with a different smoothing_hint!".format(name)
        else:
            self._smoothing_hints[name] = smoothing_hint

    # def put_scalars(self, *, smoothing_hint=True, **kwargs):
    #     """
    #     Put multiple scalars from keyword arguments.
    #     Examples:
    #         storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
    #     """
    #     for k, v in kwargs.items():
    #         self.put_scalar(k, v, smoothing_hint=smoothing_hint)
    #
    # def put_histogram(self, hist_name, hist_tensor, bins=1000):
    #     """
    #     Create a histogram from a tensor.
    #     Args:
    #         hist_name (str): The name of the histogram to put into tensorboard.
    #         hist_tensor (torch.Tensor): A Tensor of arbitrary shape to be converted
    #             into a histogram.
    #         bins (int): Number of histogram bins.
    #     """
    #     ht_min, ht_max = hist_tensor.min().item(), hist_tensor.max().item()
    #
    #     # Create a histogram with PyTorch
    #     hist_counts = torch.histc(hist_tensor, bins=bins)
    #     hist_edges = torch.linspace(start=ht_min, end=ht_max, steps=bins + 1, dtype=torch.float32)
    #
    #     # Parameter for the add_histogram_raw function of SummaryWriter
    #     hist_params = dict(
    #         tag=hist_name,
    #         min=ht_min,
    #         max=ht_max,
    #         num=len(hist_tensor),
    #         sum=float(hist_tensor.sum()),
    #         sum_squares=float(torch.sum(hist_tensor**2)),
    #         bucket_limits=hist_edges[1:].tolist(),
    #         bucket_counts=hist_counts.tolist(),
    #         global_step=self._iter,
    #     )
    #     self._histograms.append(hist_params)

    def history(self, name):
        """
        Returns:
            AverageMeter: the history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[str -> (float, int)]: mapping from the name of each scalar to the most
                recent value and the iteration number its added.
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(self, window_size=20):
        """
        Similar to :meth:`latest`, but the returned values
        are either the un-smoothed original latest value,
        or a median of the given window_size,
        depend on whether the smoothing_hint is True.
        This provides a default behavior that other writers can use.
        """
        result = {}
        for k, (v, itr) in self._latest_scalars.items():
            result[k] = (
                self._history[k].median(window_size) if self._smoothing_hints[k] else v,
                itr,
            )
        return result

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar
                is noisy and needs smoothing.
        """
        return self._smoothing_hints

    def step(self):
        """
        User should either: (1) Call this function to increment storage.iter when needed. Or
        (2) Set `storage.iter` to the correct iteration number before each iteration.
        The storage will then be able to associate the new data with an iteration number.
        """
        self._iter += 1

    @property
    def iter(self):
        """
        Returns:
            int: The current iteration number. When used together with a trainer,
                this is ensured to be the same as trainer.iter.
        """
        return self._iter

    @iter.setter
    def iter(self, val):
        """Set the current iteration number."""
        self._iter = int(val)

    @property
    def iteration(self):
        """Return the current iteration number for backward compatibility."""
        # for backward compatibility
        return self._iter

    def __enter__(self):
        _CURRENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_STORAGE_STACK[-1] == self
        _CURRENT_STORAGE_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
        """
        old_prefix = self._current_prefix
        self._current_prefix = name.rstrip("/") + "/"
        yield
        self._current_prefix = old_prefix

    def clear_images(self):
        """
        Delete all the stored images for visualization. This should be called
        after images are written to tensorboard.
        """
        self._vis_data = []

    def clear_histograms(self):
        """
        Delete all the stored histograms for visualization.
        This should be called after histograms are written to tensorboard.
        """
        self._histograms = []

    def reset_history(self, name):
        """Reset one named scalar history."""
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        ret.reset()

    def reset_histories(self):
        """Reset all scalar histories stored in this object."""
        for name in self._history.keys():
            self._history[name].reset()


class AverageMeter:
    """Compute and store the latest value and running average."""

    def __init__(self):
        """Initialize an empty average meter."""
        self.val = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def reset(self):
        """Clear all accumulated meter state."""
        self.val = 0
        self.avg = 0
        self.total = 0
        self.count = 0

    def update(self, val, n=1):
        """Add ``n`` observations with value ``val``."""
        self.val = val
        self.total += val * n
        self.count += n
        self.avg = self.total / self.count


class HistoryBuffer:
    """
    Track a series of scalar values and provide access to smoothed values over a
    window or the global average of the series.
    """

    def __init__(self, max_length: int = 1000000) -> None:
        """
        Args:
            max_length: maximal number of values that can be stored in the
                buffer. When the capacity of the buffer is exhausted, old
                values will be removed.
        """
        self._max_length: int = max_length
        self._data: List[Tuple[float, float]] = []  # (value, iteration) pairs
        self._count: int = 0
        self._global_avg: float = 0

    def update(self, value: float, iteration: Optional[float] = None) -> None:
        """
        Add a new scalar value produced at certain iteration. If the length
        of the buffer exceeds self._max_length, the oldest element will be
        removed from the buffer.
        """
        if iteration is None:
            iteration = self._count
        if len(self._data) == self._max_length:
            self._data.pop(0)
        self._data.append((value, iteration))

        self._count += 1
        self._global_avg += (value - self._global_avg) / self._count

    def latest(self) -> float:
        """
        Return the latest scalar value added to the buffer.
        """
        return self._data[-1][0]

    def median(self, window_size: int) -> float:
        """
        Return the median of the latest `window_size` values in the buffer.
        """
        return np.median([x[0] for x in self._data[-window_size:]])

    def avg(self, window_size: int) -> float:
        """
        Return the mean of the latest `window_size` values in the buffer.
        """
        return np.mean([x[0] for x in self._data[-window_size:]])

    def global_avg(self) -> float:
        """
        Return the mean of all the elements in the buffer. Note that this
        includes those getting removed due to limited buffer storage.
        """
        return self._global_avg

    def values(self) -> List[Tuple[float, float]]:
        """
        Returns:
            list[(number, iteration)]: content of the current buffer.
        """
        return self._data


class ExceptionWriter:
    """Context manager that logs exceptions before exiting the process."""

    def __init__(self):
        """Create an exception logger bound to this module."""
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        """Enter the exception logging context."""
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log the traceback and exit if the context raised an exception."""
        if exc_type:
            tb = traceback.format_exception(exc_type, exc_val, exc_tb)
            formatted_tb_str = "".join(tb)
            self.logger.error(formatted_tb_str)
            sys.exit(1)  # This prevents double logging the error to the console


class WandbWriter(EventWriter):
    """
    Write all scalars to Weights & Biases.
    """

    def __init__(
        self,
        window_size: int = 20,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Args:
            window_size (int): the scalars will be median-smoothed by this window size
            project (str): W&B project name
            name (str): W&B run name
            config (dict): Dictionary of configuration parameters for the run
            kwargs: other arguments passed to `wandb.init(...)`
        """
        if wandb is None:
            raise ImportError(
                "WandbWriter requires the training environment; "
                "run `uv sync --all-extras --locked`"
            )
        
        self._window_size = window_size
        self._wandb_initialized = False
        self._wandb_kwargs = {
            "project": project,
            "name": name,
            "config": config,
            **kwargs
        }
        self._last_write = -1
        self.run = None

    def _initialize_wandb(self):
        """Initialize W&B lazily when the first metrics are written."""
        if not self._wandb_initialized:
            self.run = wandb.init(**self._wandb_kwargs)
            _define_wandb_metrics(self.run)
            self._wandb_initialized = True

    def write(self):
        """Write scalars, images, and histograms to W&B."""
        storage = get_event_storage()
        
        if not self._wandb_initialized:
            self._initialize_wandb()
            
        rows = defaultdict(dict)
        for k, (v, iter) in storage.latest_with_smoothing_hint(
            self._window_size
        ).items():
            if iter > self._last_write:
                rows[iter][k] = v
        
        # Log images if present
        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                rows[step_num][img_name] = wandb.Image(img)
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()

        # Log histograms if present
        if len(storage._histograms) >= 1:
            for params in storage._histograms:
                tag = params.get("tag")
                if tag is not None:
                    data = np.zeros(len(params.get("bucket_counts", [])))
                    for i, count in enumerate(params.get("bucket_counts", [])):
                        data[i] = count
                    rows[params.get("global_step")][tag] = wandb.Histogram(
                        np_histogram=(data, params.get("bucket_limits", [])),
                    )
            storage.clear_histograms()

        for train_step in sorted(rows):
            _wandb_log(self.run, rows[train_step], train_step)
            self._last_write = max(self._last_write, train_step)

    def close(self):
        """Finish the active W&B run if this writer initialized one."""
        if self._wandb_initialized:
            wandb.finish()


def _to_serializable(obj):
    """Recursively convert a config object to a plain JSON-safe structure."""
    import types
    if hasattr(obj, '_cfg_dict'):  # Config wrapper
        return _to_serializable(obj._cfg_dict)
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):  # addict.Dict / ConfigDict
        return _to_serializable(obj.to_dict())
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_to_serializable(v) for v in obj)
    if isinstance(obj, types.GeneratorType):
        return [_to_serializable(v) for v in obj]
    return obj


class WandbSummaryWriter:
    """
    Emulates the TensorBoard SummaryWriter API but logs to Weights & Biases.
    """
    
    def __init__(
        self,
        log_dir=None,
        comment="",
        purge_step=None,
        max_queue=10,
        flush_secs=120,
        filename_suffix="",
        step_offset=0,
        **kwargs
    ):
        """
        Initialize a wandb run.
        
        Args:
            log_dir: Optional directory for logs (used for compatibility, wandb uses its own directory)
            comment: Optional comment (used for compatibility)
            purge_step: Compatibility parameter, not used
            max_queue: Compatibility parameter, not used
            flush_secs: Compatibility parameter, not used
            filename_suffix: Compatibility parameter, not used
            step_offset: Value added to every logged step. This only affects
                logging; trainer state, schedules, checkpoints, and eval cadence
                remain local to the current run.
            **kwargs: Additional arguments passed to wandb.init
        """
        if wandb is None:
            raise ImportError(
                "WandbSummaryWriter requires the training environment; "
                "run `uv sync --all-extras --locked`"
            )

        self.run = None
        self.step_offset = int(step_offset or 0)
        if log_dir:
            kwargs.setdefault('dir', log_dir)
        if comment:
            kwargs.setdefault('notes', comment)
        if 'config' in kwargs:
            kwargs['config'] = _to_serializable(kwargs['config'])
        self._wandb_kwargs = kwargs
        self.step = 0
        self._pending_step = None
        self._pending_data = {}

    def _ensure_initialized(self):
        """Initialize only after checkpoint loading has selected the resume mode."""
        if self.run is not None:
            return self.run
        if wandb.run is not None:
            if "resume_from" in self._wandb_kwargs:
                raise RuntimeError(
                    "Cannot rewind a W&B run after another run is already active. "
                    "CheckpointLoader must run before any W&B logging."
                )
            self.run = wandb.run
        else:
            self.run = wandb.init(**self._wandb_kwargs)
        _define_wandb_metrics(self.run)
        return self.run

    def resume_from_checkpoint(self, state):
        """Configure W&B to rewind to the next history row saved in a checkpoint."""
        if self.run is not None:
            raise RuntimeError(
                "Cannot configure W&B checkpoint resume after logging has started."
            )
        run_id = state.get("run_id")
        resume_step = state.get("resume_step")
        if not run_id or resume_step is None:
            raise ValueError(
                "W&B checkpoint state requires both run_id and resume_step."
            )
        for incompatible in ("id", "resume", "fork_from"):
            self._wandb_kwargs.pop(incompatible, None)
        self._wandb_kwargs["resume_from"] = (
            f"{run_id}?_step={int(resume_step)}"
        )

    def checkpoint_state(self):
        """Return the run ID and next internal W&B history step."""
        self.flush()
        run = self._ensure_initialized()
        return {
            "run_id": run.id,
            "resume_step": int(run.step),
        }

    def _train_step(self, global_step=None):
        """Return the semantic optimizer step after applying any configured offset."""
        step = self.step if global_step is None else global_step
        return int(step) + self.step_offset

    def log(self, data, global_step=None):
        """Collect W&B data into one history row per optimizer step."""
        train_step = self._train_step(global_step)
        self._ensure_initialized()
        if self._pending_step is not None and train_step != self._pending_step:
            self.flush()
        if self._pending_step is None:
            self._pending_step = train_step
        self._pending_data.update(data)
        if global_step is None:
            self.step += 1
    
    def add_scalar(
        self,
        tag,
        scalar_value,
        global_step=None,
        walltime=None,
        new_style=False,
        double_precision=False,
    ):
        """Log a scalar value."""
        self.log({tag: scalar_value}, global_step)
    
    def add_scalars(
        self,
        main_tag,
        tag_scalar_dict,
        global_step=None,
        walltime=None
    ):
        """Log multiple scalars at once."""
        log_dict = {f"{main_tag}/{tag}": value 
                   for tag, value in tag_scalar_dict.items()}
        self.log(log_dict, global_step)
    
    def add_histogram(
        self,
        tag,
        values,
        global_step=None,
        bins='tensorflow',
        walltime=None,
        max_bins=None,
    ):
        """Log a histogram."""
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()

        self.log({tag: wandb.Histogram(values)}, global_step)
    
    def add_image(
        self,
        tag,
        img_tensor,
        global_step=None,
        walltime=None,
        dataformats="CHW"
    ):
        """Log an image."""
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu().numpy()

        self.log(
            {tag: wandb.Image(img_tensor, dataformats=dataformats)},
            global_step,
        )
    
    def add_images(
        self,
        tag,
        img_tensor,
        global_step=None,
        walltime=None,
        dataformats="NCHW"
    ):
        """Log multiple images."""
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu().numpy()

        if dataformats == "NCHW":
            # Convert to a list of images
            images = [img for img in img_tensor]
            data = {tag: [wandb.Image(img, dataformats="CHW") for img in images]}
        else:
            data = {
                tag: [
                    wandb.Image(img, dataformats=dataformats)
                    for img in img_tensor
                ]
            }
        self.log(data, global_step)
    
    def add_figure(
        self,
        tag,
        figure,
        global_step=None,
        close=True,
        walltime=None,
    ):
        """Log a matplotlib figure."""
        self.log({tag: wandb.Image(figure)}, global_step)
        
        if close:
            plt.close(figure)
    
    def add_text(
        self,
        tag,
        text_string,
        global_step=None,
        walltime=None
    ):
        """Log text."""
        self.log({tag: wandb.Html(f"<pre>{text_string}</pre>")}, global_step)
    
    def add_video(
        self,
        tag,
        vid_tensor,
        global_step=None,
        fps=4,
        walltime=None
    ):
        """Log a video."""
        if isinstance(vid_tensor, torch.Tensor):
            vid_tensor = vid_tensor.detach().cpu().numpy()

        # wandb expects [time, channels, height, width]
        self.log({tag: wandb.Video(vid_tensor, fps=fps)}, global_step)
    
    def add_embedding(
        self,
        mat,
        metadata=None,
        label_img=None,
        global_step=None,
        tag="default",
        metadata_header=None,
    ):
        """Log embeddings - use wandb.plot.scatter for this functionality"""
        if isinstance(mat, torch.Tensor):
            mat = mat.detach().cpu().numpy()

        # Create a custom 2D/3D scatter plot in wandb
        if mat.shape[1] > 3:
            # If high-dimensional, log a simple message
            self.log({f"{tag}_embedding_logged": True}, global_step)
            print(f"Note: High-dimensional embeddings ({mat.shape[1]}D) aren't visualized directly in wandb.")
            print("Consider using UMAP or t-SNE to reduce to 2D/3D before logging.")
        else:
            # Basic table with embeddings
            data = [[idx] + list(row) for idx, row in enumerate(mat)]
            table = wandb.Table(columns=["id"] + [f"dim_{i}" for i in range(mat.shape[1])], data=data)
            self.log({tag: table}, global_step)

    def add_pr_curve(
        self,
        tag,
        labels,
        predictions,
        global_step=None,
        num_thresholds=127,
        weights=None,
        walltime=None,
    ):
        """Log precision-recall curve"""
        # Note: W&B has wandb.plot.pr_curve but it needs actual predictions/labels
        # This is a compatibility stub
        print("Precision-Recall curves are tracked through wandb.log({'pr': wandb.plot.pr_curve(...)})")
        
    def add_mesh(
        self,
        tag,
        vertices,
        colors=None,
        faces=None,
        config_dict=None,
        global_step=None,
        walltime=None,
    ):
        """Log 3D mesh (not directly supported in wandb API)"""
        print("3D meshes are not directly supported in W&B's Python API")
        
    def add_hparams(
        self,
        hparam_dict,
        metric_dict,
        hparam_domain_discrete=None,
        run_name=None,
        global_step=None,
    ):
        """Log hyperparameters and metrics"""
        # wandb tracks hyperparameters automatically through config
        self._ensure_initialized()
        for key, value in hparam_dict.items():
            self.run.config[key] = value

        # Log metrics
        self.log(metric_dict, global_step)
            
    def flush(self):
        """Commit the accumulated optimizer-step row to W&B."""
        if self._pending_step is None:
            return
        _wandb_log(self._ensure_initialized(), self._pending_data, self._pending_step)
        self._pending_step = None
        self._pending_data = {}

    def flush_step(self):
        """Commit the current optimizer-step row at a trainer boundary."""
        self.flush()
    
    def close(self):
        """Finish logging (optional, wandb handles this automatically)"""
        if self.run:
            self.flush()
            wandb.finish()
            self.run = None
            
    def __enter__(self):
        """Enter a SummaryWriter-compatible context."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the writer when leaving a context manager block."""
        self.close()
