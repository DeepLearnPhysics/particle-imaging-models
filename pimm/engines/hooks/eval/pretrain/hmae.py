import torch

import pimm.utils.comm as comm
from pimm.engines.hooks.default import HookBase
from pimm.engines.hooks.builder import HOOKS

def _get_writer_step(trainer):
    """Local train step for logging; writer applies any configured offset."""
    ci = getattr(trainer, "comm_info", {})
    if "epoch" in ci and "iter" in ci:
        return ci.get("epoch", 0) * ci.get("iter_per_epoch", 0) + ci.get("iter", 0) + 1
    return getattr(trainer, "epoch", 0)

@HOOKS.register_module()
class HMAEEvaluator(HookBase):
    """
    Validation hook for HMAE that logs chamfer losses on the validation set.
    """

    def __init__(self, every_n_steps: int = 0, max_batches: int = None):
        """
        Args:
            every_n_steps: run validation every N steps. If 0, run every epoch instead.
            max_batches: limit number of batches for faster validation (None = all)
        """
        self.every_n_steps = every_n_steps
        self.max_batches = max_batches

    def after_step(self):
        """Run HMAE validation on the configured step cadence."""
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        if self.every_n_steps > 0:
            global_iter = (
                self.trainer.comm_info["iter"]
                + self.trainer.comm_info["iter_per_epoch"] * self.trainer.comm_info["epoch"]
            )
            if (global_iter + 1) % self.every_n_steps == 0:
                if comm.get_world_size() > 1:
                    if comm.get_rank() == 0:
                        self.eval()
                else:
                    self.eval()

    def after_epoch(self):
        """Run HMAE validation after epochs when step cadence is disabled."""
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        if self.every_n_steps == 0:
            if comm.get_world_size() > 1:
                if comm.get_rank() == 0:
                    self.eval()
            else:
                self.eval()

    @torch.no_grad()
    def eval(self):
        """Average HMAE validation losses over valid batches."""
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start HMAE Validation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        total_loss = 0.0
        total_coord_loss = 0.0
        total_feat_loss = 0.0
        num_batches = 0
        num_valid = 0

        for i, input_dict in enumerate(self.trainer.test_loader):
            if self.max_batches is not None and i >= self.max_batches:
                break

            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            output_dict = self.trainer.model(input_dict)

            loss_val = output_dict.get("loss", 0.0)
            coord_loss_val = output_dict.get("coord_loss", 0.0)
            feat_loss_val = output_dict.get("feat_loss", 0.0)

            # handle tensor vs scalar
            if hasattr(loss_val, "item"):
                loss_val = loss_val.item()
            if hasattr(coord_loss_val, "item"):
                coord_loss_val = coord_loss_val.item()
            if hasattr(feat_loss_val, "item"):
                feat_loss_val = feat_loss_val.item()

            # skip invalid batches (loss=0 from hmae_valid=False)
            if loss_val == 0.0:
                continue

            total_loss += loss_val
            total_coord_loss += coord_loss_val
            total_feat_loss += feat_loss_val
            num_valid += 1
            num_batches += 1

            if (i + 1) % 10 == 0:
                self.trainer.logger.info(
                    f"Val: [{i + 1}/{len(self.trainer.val_loader)}] "
                    f"Loss: {loss_val:.6f} Coord: {coord_loss_val:.6f} Feat: {feat_loss_val:.6f}"
                )

        if num_valid > 0:
            avg_loss = total_loss / num_valid
            avg_coord = total_coord_loss / num_valid
            avg_feat = total_feat_loss / num_valid
        else:
            avg_loss = avg_coord = avg_feat = 0.0

        self.trainer.logger.info(
            f"Val Result: Loss: {avg_loss:.6f} Coord: {avg_coord:.6f} Feat: {avg_feat:.6f} "
            f"({num_valid}/{num_batches} valid batches)"
        )

        # log to wandb/tensorboard
        if self.trainer.writer is not None:
            step = _get_writer_step(self.trainer)
            self.trainer.writer.add_scalar("val/loss", avg_loss, step)
            self.trainer.writer.add_scalar("val/coord_loss", avg_coord, step)
            self.trainer.writer.add_scalar("val/feat_loss", avg_feat, step)

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End HMAE Validation <<<<<<<<<<<<<<<<<")
        self.trainer.model.train()
