"""Gradient, dtype, entropy, prototype, and parameter diagnostics."""

import re
import torch
import torch.nn as nn

from .default import HookBase
from .builder import HOOKS

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
        """Configure gradient norm type, cadence, and writer namespace."""
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
        """Return a compact configuration summary for logs."""
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
        """Configure name/class matching and target compute dtype."""
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
        """Return a compact configuration summary for logs."""
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
        """Configure logits source, logging cadence, and entropy reduction."""
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
        """Return a compact configuration summary for logs."""
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
        """Configure prototype usage logging cadence and namespace."""
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
        """Return a compact configuration summary for logs."""
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
        """Configure which teacher/student features are monitored."""
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
        
        # Check if model has student/teacher structure (Sonata, JEPA, etc.)
        sonata_module = None
        if hasattr(model, 'sonata'):
            sonata_module = model.sonata
        elif isinstance(model, torch.nn.ModuleDict) and 'sonata' in model:
            sonata_module = model['sonata']
        elif hasattr(model, 'student') and hasattr(model, 'teacher'):
            sonata_module = model

        if not sonata_module:
            self.trainer.logger.warning("Could not find student/teacher module for feature monitoring")
            return
        
        # Register hooks on teacher backbone
        if self.monitor_teacher and hasattr(sonata_module, 'teacher') and 'backbone' in sonata_module.teacher:
            self.trainer.logger.info("Registering feature monitor on teacher backbone")
            hook = sonata_module.teacher['backbone'].register_forward_hook(
                self._feature_stats_hook('teacher')
            )
            self.hook_handles.append(hook)
            if 'representation_fusion' in sonata_module.teacher:
                self.trainer.logger.info("Registering feature monitor on teacher representation_fusion")
                hook = sonata_module.teacher['representation_fusion'].register_forward_hook(
                    self._feature_stats_hook('teacher/representation_fusion')
                )
                self.hook_handles.append(hook)
        
        # Register hooks on student backbone
        if self.monitor_student and hasattr(sonata_module, 'student') and 'backbone' in sonata_module.student:
            self.trainer.logger.info("Registering feature monitor on student backbone")
            hook = sonata_module.student['backbone'].register_forward_hook(
                self._feature_stats_hook('student')
            )
            self.hook_handles.append(hook)
            if 'representation_fusion' in sonata_module.student:
                self.trainer.logger.info("Registering feature monitor on student representation_fusion")
                hook = sonata_module.student['representation_fusion'].register_forward_hook(
                    self._feature_stats_hook('student/representation_fusion')
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
        """Return a compact configuration summary for logs."""
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
        """Configure parameter detail level and filtering for startup logs."""
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
        """Return a compact configuration summary for logs."""
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
        """Configure annealing update logging cadence and namespace."""
        self.log_frequency = log_frequency
        self.log_per_layer = log_per_layer
        self.prefix = prefix
        self.step_count = 0
        self.annealing_complete = False
        self.has_annealing = False
    
    def before_train(self):
        """Check if model supports annealing and log initial state."""
        self.step_count = int(getattr(self.trainer, "global_step", 0) or (
            self.trainer.start_epoch * len(self.trainer.train_loader)
            + self.trainer.start_iter
        ))

        # Get model without DDP wrapper if present
        model = self.trainer.model
        if hasattr(model, 'module'):
            model = model.module
        
        # Check if model has update_anneal_step method
        if hasattr(model, 'update_anneal_step'):
            self.has_annealing = True
            self.trainer.logger.info("Attention mask annealing enabled")
            model.update_anneal_step(self.step_count)
            
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
        """Return a compact configuration summary for logs."""
        return (
            f"{self.__class__.__name__}("
            f"log_frequency={self.log_frequency}, "
            f"log_per_layer={self.log_per_layer}, "
            f"prefix='{self.prefix}')"
        )
