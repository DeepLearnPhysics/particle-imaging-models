"""Final evaluation hook."""

import os

import torch

from pimm.engines.hooks.builder import HOOKS
from pimm.utils.checkpoints import checkpoint_model_state_dict
from pimm.engines.hooks.default import HookBase
from pimm.observability import structured_logger as sl


@HOOKS.register_module()
class FinalEvaluator(HookBase):
    """Run the configured tester once after training, usually on model_best.

    On ``after_train`` (and only when ``cfg.evaluate`` is true), builds the
    tester named by ``cfg.test.type`` from the ``TESTERS`` registry and runs its
    full ``test()`` pass. By default it loads ``model_best.pth`` (the best
    checkpoint selected during training) into the tester before testing; set
    ``test_last=True`` to instead test the in-memory ``model_last`` weights. This
    hook computes no metric of its own and does not set the checkpoint-selection
    ``comm_info`` keys — it delegates all metric computation to the tester.
    Registered as ``FinalEvaluator`` (use as ``type`` in a ``hooks=[...]``
    entry).

    Args:
        test_last (bool): Test the current ``model_last`` weights instead of
            loading ``model_best.pth``. Defaults to ``False``.

    Note:
        Runs exactly once, after training exits. When ``test_last`` is ``False``
        the best checkpoint is loaded with ``strict=True``, so the tester model
        architecture must match the saved state dict.

    Example:
        Add to ``cfg.hooks``; once in ``after_train`` it runs the configured
        tester on the best checkpoint:

        .. code-block:: python

            hooks = [dict(type="FinalEvaluator", test_last=False)]
            # → after training, builds cfg.test.type from the TESTERS registry,
            #   loads <save_path>/model/model_best.pth (strict=True) into it, and
            #   runs tester.test(); computes no metric of its own
    """

    def __init__(self, test_last=False):
        """Select whether final evaluation uses model_last instead of model_best."""
        self.test_last = test_last

    @sl.log_trace_span("evaluation.final")
    def after_train(self):
        """Build the tester and execute final evaluation after training exits."""
        if not self.trainer.cfg.get("evaluate", True):
            self.trainer.logger.info("Skipping final evaluation (evaluate=False)")
            return
        sl.add_step_tag("evaluation")

        # Import lazily to avoid pimm.engines.test importing hooks while hooks
        # are still being registered.
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
            state_dict = checkpoint_model_state_dict(checkpoint)
            tester.model.load_state_dict(state_dict, strict=True)
        tester.test()
