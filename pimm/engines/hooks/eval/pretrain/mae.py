
import numpy as np
import torch

import pimm.utils.comm as comm
import wandb
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase

def _get_writer_step(trainer):
    """Local train step for logging; writer applies any configured offset."""
    ci = getattr(trainer, "comm_info", {})
    if "epoch" in ci and "iter" in ci:
        return ci.get("epoch", 0) * ci.get("iter_per_epoch", 0) + ci.get("iter", 0) + 1
    return getattr(trainer, "epoch", 0)


@HOOKS.register_module()
class MAEEvaluator(HookBase):
    """
    Validation hook for MAE that logs reconstruction losses on the validation set.
    """

    def __init__(self, every_n_steps: int = 0, max_batches: int = None, log_pointclouds: bool = True):
        """
        Args:
            every_n_steps: run validation every N steps. If 0, run every epoch instead.
            max_batches: limit number of batches for faster validation (None = all)
            log_pointclouds: if True, log reconstruction visualizations to wandb
        """
        self.every_n_steps = every_n_steps
        self.max_batches = max_batches
        self.log_pointclouds = log_pointclouds

    def after_step(self):
        """Run MAE validation on the configured step cadence."""
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        if self.every_n_steps > 0:
            global_iter = (
                self.trainer.comm_info["iter"]
                + self.trainer.comm_info["iter_per_epoch"] * self.trainer.comm_info["epoch"]
            )
            if (global_iter + 1) % self.every_n_steps == 0:
                self.eval()

    def after_epoch(self):
        """Run MAE validation after epochs when step cadence is disabled."""
        if not self.trainer.cfg.evaluate or self.trainer.val_loader is None:
            return
        if self.every_n_steps == 0:
            self.eval()

    @torch.no_grad()
    def eval(self):
        """Average reconstruction losses and optional point-cloud visualizations."""
        # only run on rank 0
        rank = comm.get_rank()
        if rank != 0:
            if comm.get_world_size() > 1:
                comm.synchronize()
            return

        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start MAE Validation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()

        total_loss = 0.0
        total_coord_loss = 0.0
        total_feat_loss = 0.0
        total_mask_ratio = 0.0
        num_batches = 0
        num_valid = 0

        loader = self.trainer.val_loader
        for i, input_dict in enumerate(loader):
            if self.max_batches is not None and i >= self.max_batches:
                break

            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            # use AMP if enabled in config to match training conditions
            # on first batch, request visualization data
            return_viz = (i == 0 and self.log_pointclouds)
            if getattr(self.trainer.cfg, "enable_amp", False):
                amp_dtype = getattr(self.trainer.cfg, "amp_dtype", "bfloat16")
                dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    output_dict = self.trainer.model(input_dict, return_pred=return_viz)
            else:
                output_dict = self.trainer.model(input_dict, return_pred=return_viz)

            # log point cloud visualizations on first batch
            if return_viz and "viz_visible_coord" in output_dict:
                self._log_pointcloud_viz(output_dict)

            loss_val = output_dict.get("loss", 0.0)
            coord_loss_val = output_dict.get("coord_loss", 0.0)
            feat_loss_val = output_dict.get("feat_loss", 0.0)
            mask_ratio_val = output_dict.get("mask_ratio_actual", 0.0)

            # handle tensor vs scalar
            if hasattr(loss_val, "item"):
                loss_val = loss_val.item()
            if hasattr(coord_loss_val, "item"):
                coord_loss_val = coord_loss_val.item()
            if hasattr(feat_loss_val, "item"):
                feat_loss_val = feat_loss_val.item()
            if hasattr(mask_ratio_val, "item"):
                mask_ratio_val = mask_ratio_val.item()

            # skip invalid batches (loss=0 from empty masks)
            if loss_val == 0.0:
                num_batches += 1
                continue

            total_loss += loss_val
            total_coord_loss += coord_loss_val
            total_feat_loss += feat_loss_val
            total_mask_ratio += mask_ratio_val
            num_valid += 1
            num_batches += 1

            if (i + 1) % 10 == 0:
                self.trainer.logger.info(
                    f"Val: [{i + 1}/{len(loader)}] "
                    f"Loss: {loss_val:.6f} Coord: {coord_loss_val:.6f} "
                    f"Feat: {feat_loss_val:.6f} MaskRatio: {mask_ratio_val:.3f}"
                )

        if num_valid > 0:
            avg_loss = total_loss / num_valid
            avg_coord = total_coord_loss / num_valid
            avg_feat = total_feat_loss / num_valid
            avg_mask_ratio = total_mask_ratio / num_valid
        else:
            avg_loss = avg_coord = avg_feat = avg_mask_ratio = 0.0

        self.trainer.logger.info(
            f"Val Result: Loss: {avg_loss:.6f} Coord: {avg_coord:.6f} "
            f"Feat: {avg_feat:.6f} MaskRatio: {avg_mask_ratio:.3f} "
            f"({num_valid}/{num_batches} valid batches)"
        )

        # log to wandb/tensorboard
        if self.trainer.writer is not None:
            step = _get_writer_step(self.trainer)
            self.trainer.writer.add_scalar("val/loss", avg_loss, step)
            self.trainer.writer.add_scalar("val/coord_loss", avg_coord, step)
            self.trainer.writer.add_scalar("val/feat_loss", avg_feat, step)
            self.trainer.writer.add_scalar("val/mask_ratio", avg_mask_ratio, step)

        self.trainer.comm_info["current_metric_value"] = -avg_loss  # negative since lower is better
        self.trainer.comm_info["current_metric_name"] = "neg_val_loss"

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End MAE Validation <<<<<<<<<<<<<<<<<")
        self.trainer.model.train()

        # synchronize other ranks
        if comm.get_world_size() > 1:
            comm.synchronize()

    def _log_pointcloud_viz(self, output_dict):
        """Log point cloud visualizations to wandb."""
        try:
            # check if wandb is available and active
            if self.trainer.writer is None:
                return
            if not hasattr(self.trainer.writer, "run") or self.trainer.writer.run is None:
                return

            visible_coord = output_dict["viz_visible_coord"].numpy()  # (N_vis, 3)
            pred_coord = output_dict["viz_pred_coord"].numpy()  # (N_masked, K, 3)
            target_coord = output_dict["viz_target_coord"].numpy()  # (N_target, 3)
            target_counts = output_dict["viz_target_counts"].numpy()  # (N_masked,)

            # flatten predicted coordinates
            pred_coord_flat = pred_coord.reshape(-1, 3)  # (N_masked * K, 3)

            # create colored point clouds
            # visible points: blue (0, 0, 255)
            # predicted points: red (255, 0, 0)
            # target points: green (0, 255, 0)

            n_vis = visible_coord.shape[0]
            n_pred = pred_coord_flat.shape[0]
            n_target = target_coord.shape[0]

            # subsample if too many points (wandb has limits)
            max_points = 50000
            if n_vis > max_points:
                idx = np.random.choice(n_vis, max_points, replace=False)
                visible_coord = visible_coord[idx]
                n_vis = max_points
            if n_pred > max_points:
                idx = np.random.choice(n_pred, max_points, replace=False)
                pred_coord_flat = pred_coord_flat[idx]
                n_pred = max_points
            if n_target > max_points:
                idx = np.random.choice(n_target, max_points, replace=False)
                target_coord = target_coord[idx]
                n_target = max_points

            # combine visible + target (ground truth)
            gt_combined = np.vstack([visible_coord, target_coord])
            gt_colors = np.vstack([
                np.full((n_vis, 3), [0, 0, 255]),  # blue for visible
                np.full((n_target, 3), [0, 255, 0]),  # green for masked (GT)
            ])
            gt_pointcloud = np.hstack([gt_combined, gt_colors])

            # combine visible + predicted (reconstruction)
            recon_combined = np.vstack([visible_coord, pred_coord_flat])
            recon_colors = np.vstack([
                np.full((n_vis, 3), [0, 0, 255]),  # blue for visible
                np.full((n_pred, 3), [255, 0, 0]),  # red for predicted
            ])
            recon_pointcloud = np.hstack([recon_combined, recon_colors])

            # log to wandb
            self.trainer.writer.run.log({
                "val/pointcloud_gt": wandb.Object3D(gt_pointcloud),
                "val/pointcloud_recon": wandb.Object3D(recon_pointcloud),
            })

            self.trainer.logger.info("Logged point cloud visualizations to wandb")

        except Exception as e:
            self.trainer.logger.warning(f"Failed to log point cloud viz: {e}")
