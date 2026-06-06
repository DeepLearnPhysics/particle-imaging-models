"""Optimizer mutation and scheduling hooks."""

from pimm.utils.scheduler import CosineScheduler

from .default import HookBase
from .builder import HOOKS

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
    """Apply a cosine schedule to optimizer parameter-group weight decay."""

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
        curr_step = getattr(self.trainer, "global_step", 0) or (
            self.trainer.start_epoch * len(self.trainer.train_loader)
        )
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
