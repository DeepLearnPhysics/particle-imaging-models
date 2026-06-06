"""Final evaluation hook."""

import os

import torch

from pimm.engines.hooks.builder import HOOKS
from pimm.utils.checkpoints import checkpoint_model_state_dict
from pimm.engines.hooks.default import HookBase


@HOOKS.register_module()
class FinalEvaluator(HookBase):
    """Run the configured tester after training, usually on model_best."""

    def __init__(self, test_last=False):
        """Select whether final evaluation uses model_last instead of model_best."""
        self.test_last = test_last

    def after_train(self):
        """Build the tester and execute final evaluation after training exits."""
        if not self.trainer.cfg.get("evaluate", True):
            self.trainer.logger.info("Skipping final evaluation (evaluate=False)")
            return

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
