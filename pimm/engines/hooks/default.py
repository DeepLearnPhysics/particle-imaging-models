"""
Default Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import pimm.utils.comm as comm
import weakref
from .builder import HOOKS


def get_writer_step(trainer):
    """Return the local step; the writer applies any configured offset."""
    info = getattr(trainer, "comm_info", {})
    if "epoch" in info and "iter" in info:
        return info.get("epoch", 0) * info.get("iter_per_epoch", 0) + info.get("iter", 0) + 1
    return getattr(trainer, "epoch", 0)


class HookBase:
    """Base class for all training hooks registered with the trainer.

    A hook plugs custom behaviour into fixed points of the training loop. The
    trainer calls each lifecycle method on every registered hook, in
    registration order, at the corresponding point in the run. Subclasses
    override only the methods they need; the defaults are no-ops. Every hook in
    ``pimm`` derives from this class.

    The lifecycle methods, in the order the trainer invokes them, are:

    - ``modify_config(cfg)``: called once after all hooks are registered but
      before the writer (W&B/TensorBoard) is built, so a hook may still mutate
      the config (e.g. set ``cfg.wandb_run_name``).
    - ``before_train()``: called once before the training loop starts.
    - ``before_epoch()``: called at the start of every epoch.
    - ``before_step()``: called before each training step (before the batch is
      consumed by the model).
    - ``after_backward()``: called after gradients are unscaled and before any
      gradient clipping or optimizer update.
    - ``after_step()``: called after each training step (after the optimizer
      update).
    - ``after_epoch()``: called at the end of every epoch.
    - ``after_train()``: called once after the training loop finishes.

    Note:
        ``self.trainer`` is a weak reference (proxy) to the owning trainer,
        assigned by the trainer at registration time. Through it a hook reaches
        shared state such as ``trainer.model``, ``trainer.optimizer``,
        ``trainer.storage``, ``trainer.writer``, ``trainer.logger``,
        ``trainer.comm_info``, and ``trainer.cfg``. In distributed runs every
        rank runs every hook, so guard rank-0-only side effects with
        ``pimm.utils.comm.is_main_process()`` and keep any collective operations
        consistent across ranks.
    """

    trainer = None  # A weak reference to the trainer object.

    def modify_config(self, cfg):
        """Called after hooks are registered but before writer is built.
        Hooks can modify the config here (e.g., wandb_run_name).
        """
        pass

    def before_train(self):
        pass

    def before_epoch(self):
        pass

    def before_step(self):
        pass

    def after_backward(self):
        pass

    def after_step(self):
        pass

    def after_epoch(self):
        pass

    def after_train(self):
        pass


@HOOKS.register_module()
class ModelHook(HookBase):
    """Bridge that forwards lifecycle calls to a model implementing ``HookBase``.

    Lets a model participate in the training loop as if it were a hook. In
    ``before_train`` it resolves the underlying model (unwrapping the DDP
    ``.module`` in distributed runs); if that model is itself a ``HookBase`` it
    is bound as a weak proxy, otherwise a no-op ``HookBase`` stand-in is used.
    Every subsequent lifecycle call (``before_epoch``, ``before_step``,
    ``after_step``, ``after_epoch``, ``after_train``) is forwarded to the model.
    Registered as ``ModelHook``.

    Note:
        ``modify_config`` is intentionally not forwarded — by the time hooks
        run, the model is already built from the config. The bound model's
        ``trainer`` attribute is set to the same trainer so it can reach shared
        training state.

    Example:
        Add to ``cfg.hooks``; at each lifecycle point the trainer forwards the
        call to the model so the model's own hook methods run inside the loop:

        .. code-block:: python

            hooks = [dict(type="ModelHook")]
            # → in before_train binds the (DDP-unwrapped) model as a HookBase and
            #   forwards before_epoch/before_step/after_step/after_epoch/after_train
            #   to model.<same method> every step (modify_config is NOT forwarded)
    """

    def before_train(self):
        if comm.get_world_size() > 1 and hasattr(self.trainer.model, 'module') and isinstance(
            self.trainer.model.module, HookBase
        ):
            self.model = weakref.proxy(self.trainer.model.module)
        elif isinstance(self.trainer.model, HookBase):
            self.model = weakref.proxy(self.trainer.model)
        else:
            self.model = HookBase()
        self.model.trainer = self.trainer
        self.model.before_train()

    def before_epoch(self):
        self.model.before_epoch()

    def before_step(self):
        self.model.before_step()

    def after_backward(self):
        self.model.after_backward()

    def after_step(self):
        self.model.after_step()

    def after_epoch(self):
        self.model.after_epoch()

    def after_train(self):
        self.model.after_train()
