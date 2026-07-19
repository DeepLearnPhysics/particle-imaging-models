"""Gradient, dtype, entropy, prototype, and parameter diagnostics."""

import os
import re
import torch
import torch.nn as nn

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class SimulateCrash(HookBase):
    """Hard-exit the process after a fixed number of steps to simulate a crash.

    Fires only when ``PIMM_SIMULATE_CRASH_STEP`` is set, so a run can die
    deterministically after writing a checkpoint while a later resume of the
    same config runs to completion. Used to test crash recovery.
    """

    def after_step(self):
        target = os.environ.get("PIMM_SIMULATE_CRASH_STEP")
        if target is None:
            return
        self.step_count = getattr(self, "step_count", 0) + 1
        if self.step_count >= int(target):
            os._exit(137)

@HOOKS.register_module()
class GradientNormLogger(HookBase):
    """Log model gradient norms to the writer after each training step.

    Runs in ``after_step`` every ``log_frequency`` steps: computes the
    aggregate gradient norm over all model parameters (for the configured
    ``norm_type``) and writes it under ``{prefix}/total``. With
    ``log_per_layer=True`` it additionally writes each named parameter's
    gradient norm under ``{prefix}/layers/<name>``. Registered as
    ``GradientNormLogger``.

    Args:
        norm_type (float): Order of the norm to compute, e.g. ``2.0`` for L2 or
            ``float("inf")`` for max-abs. Defaults to ``2.0``.
        log_per_layer (bool): If ``True``, also log per-parameter gradient
            norms (verbose). Defaults to ``False``.
        log_frequency (int): Compute and log every this many steps. Defaults to
            ``1``.
        prefix (str): Namespace prefix for the writer keys. Defaults to
            ``"grad_norm"``.

    Note:
        No-ops when ``trainer.writer`` is absent/``None`` or does not expose
        ``add_scalar``. Reads gradients after backward, so it reflects the
        un-clipped gradients of that step.

    Example:
        Add to ``cfg.hooks``; after every ``log_frequency`` steps it writes the
        total gradient norm to the experiment writer (W&B/TensorBoard):

        .. code-block:: python

            hooks = [dict(type="GradientNormLogger", log_frequency=50)]
            # → logs scalar  "grad_norm/total"  every 50 optimizer steps
            #   (with log_per_layer=True, also "grad_norm/layers/<name>" per param)

        The norm helper is pure and can be exercised standalone:

        .. code-block:: python

            >>> import torch
            >>> from pimm.engines.hooks.diagnostics import GradientNormLogger
            >>> p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
            >>> (0.5 * (p ** 2).sum()).backward()   # grad = [3., 4.]
            >>> float(GradientNormLogger()._compute_grad_norm([p]))
            5.0
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
    """Force matched layers to compute (and optionally store params) in a dtype.

    Runs once in ``before_train``: walks the model and, for every submodule
    whose dotted name matches ``patterns`` or whose class name matches
    ``class_patterns``, wraps each method in ``methods_to_override`` so its
    tensor input is cast to ``dtype`` and its output cast back to the original
    dtype. With ``override_parameters=True`` the matched module's own
    parameters are also cast to ``dtype`` in place. Useful for keeping
    precision-sensitive layers (e.g. LayerNorm) in fp32 under mixed-precision
    training. Registered as ``DtypeOverrider``.

    Args:
        patterns (list, optional): Regex patterns matched against dotted module
            names. Defaults to ``None`` (treated as ``[]``).
        class_patterns (list, optional): Regex patterns matched against module
            class names. Defaults to ``None`` (treated as ``[]``).
        dtype (str): Target dtype name (key of the module's dtype table, e.g.
            ``"float32"``, ``"bfloat16"``), resolved to a ``torch.dtype``.
            Defaults to ``"float32"``.
        methods_to_override (list, optional): Method names on matched modules to
            wrap with the cast. Defaults to ``None`` (treated as
            ``["forward"]``).
        override_parameters (bool): If ``True``, also cast matched modules'
            parameters to ``dtype`` in place. Defaults to ``False``.
        verbose (bool): If ``True``, log each wrapped method/parameter and a
            summary. Defaults to ``False``.
        check_interval (int): Reserved cadence for periodic parameter-dtype
            re-checks. Defaults to ``10``.

    Note:
        Only the first positional tensor argument of a wrapped method is cast;
        methods whose first argument is not a tensor are called unchanged. The
        DDP wrapper is handled by walking the underlying model.

    Example:
        Add to ``cfg.hooks``; once in ``before_train`` it wraps matched modules
        so they compute in ``dtype`` regardless of AMP autocast:

        .. code-block:: python

            hooks = [
                dict(type="DtypeOverrider",
                     patterns=["layer_norm", "LayerNorm", "norm"],
                     dtype="float32",
                     override_parameters=True,
                     verbose=True),
            ]
            # → each matched module's forward() now casts its input to float32 and
            #   casts the output back to the original dtype; with
            #   override_parameters=True the module's own params are converted to
            #   float32 in place (each wrap/conversion logged via verbose)
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
    """Log the entropy of teacher logits using Sonata's temperature schedule.

    Monitors prediction confidence by computing the Shannon entropy of the
    temperature-scaled softmax over ``logits_key`` in the model output dict. In
    ``before_train`` it locates the Sonata submodule (unwrapping DDP) to read
    its current ``teacher_temp``; if none is found it falls back to
    ``default_temperature``. Runs in ``after_step`` every ``log_frequency``
    steps: logs the entropy (and the temperature) to the writer under
    ``train_batch/{prefix}`` and to ``trainer.storage``/``iter_info``, and
    stashes the temperature back into the model output dict for other consumers.
    ``after_epoch`` logs the epoch-average entropy under ``train/{prefix}``.
    Registered as ``LogitEntropyLogger``.

    Args:
        logits_key (str): Key in the model output dict holding the logits.
            Defaults to ``"teacher_logits"``.
        log_frequency (int): Compute and log every this many steps. Defaults to
            ``1``.
        prefix (str): Namespace prefix for the logged keys. Defaults to
            ``"entropy"``.
        reduction (str): How to reduce the per-token entropy: ``"mean"``,
            ``"sum"``, or ``"none"``. Defaults to ``"mean"``.
        log_per_class (bool): If ``True`` (and ``reduction="none"``), also log
            per-class mean entropy. Defaults to ``False``.
        default_temperature (float): Temperature used when no Sonata model is
            found. Defaults to ``0.07``.

    Note:
        Higher entropy indicates more uncertain predictions; lower entropy more
        confident ones. No-ops on a given step if the logits key is absent from
        the model output dict.

    Example:
        Add to ``cfg.hooks`` for Sonata-style SSL; every ``log_frequency`` steps
        it logs the entropy of the teacher logits:

        .. code-block:: python

            hooks = [dict(type="LogitEntropyLogger", log_frequency=50)]
            # → writes "train_batch/entropy" and "entropy/temperature" to the writer
            #   (and the epoch average "train/entropy"), puts "entropy" in
            #   trainer.storage, and appends "entropy: <val>" to the iter log line
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
    """Monitor prototype utilization in Sonata-style clustering heads.

    In ``before_train`` (after unwrapping DDP) it finds the Sonata submodule and
    registers a forward hook on each teacher/student head whose name contains
    ``"head"``. Each forward hook fires every ``log_frequency`` calls (training
    mode only) and, from the head's output logits, computes prototype-usage
    statistics via argmax assignment counts (synchronized across ranks):
    number of used prototypes, percent unused, average tokens per active
    prototype, and the assignment-distribution entropy. These are written to the
    writer under ``{prefix}/<head>/...``. ``after_train`` removes all registered
    hooks. Registered as ``PrototypeUsageLogger``.

    When ``log_pooling_levels`` is set it ALSO hooks each teacher/student
    ``backbone`` and, by walking the ``pooling_parent`` chain of the returned
    ``Point`` (read-only, so the model's later up-cast is untouched), logs the
    cross-rank average number of points per event at each hierarchy stage under
    ``{prefix}/<backbone>/points_per_event/level{k}`` (``level0`` = finest input
    resolution, increasing with pooling depth). This surfaces redundant pooling
    stages -- e.g. a deepest level that collapses to ~1 point per event after
    many pooling operations.

    Args:
        log_frequency (int): Log statistics every this many head/backbone forward
            passes. Defaults to ``10``.
        prefix (str): Namespace prefix for the logged keys. Defaults to
            ``"prototypes"``.
        log_pooling_levels (bool): Also log average points-per-event at each
            backbone pooling stage. Defaults to ``True``.

    Note:
        Specific to models exposing a Sonata-like ``teacher``/``student``
        ``ModuleDict`` of heads; warns and does nothing if none is found.
        Counts are all-reduced so the reported usage is global across GPUs.

    Example:
        Add to ``cfg.hooks`` for Sonata-style SSL; it watches each clustering
        head's prototype assignments:

        .. code-block:: python

            hooks = [dict(type="PrototypeUsageLogger", log_frequency=20)]
            # → every 20 head forward passes writes, per head, the cross-rank
            #   "prototypes/<teacher|student>/<head>/{used_count,unused_percent,
            #   tokens_per_prototype,assignment_entropy}" to the writer, plus
            #   "prototypes/<teacher|student>/backbone/points_per_event/level{k}"
            #   for each pooling stage
    """

    def __init__(
        self,
        log_frequency=10,
        prefix="prototypes",
        log_pooling_levels=True,
    ):
        """Configure prototype usage logging cadence and namespace."""
        self.log_frequency = log_frequency
        self.prefix = prefix
        self.log_pooling_levels = log_pooling_levels
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

        # Register hooks on the backbones to log per-stage point counts.
        if self.log_pooling_levels:
            for role in ("teacher", "student"):
                sub = getattr(sonata_module, role, None)
                if isinstance(sub, torch.nn.ModuleDict) and "backbone" in sub:
                    self.trainer.logger.info(
                        f"Registering pooling-level monitor on {role}/backbone"
                    )
                    hook = sub["backbone"].register_forward_hook(
                        self._pooling_levels_hook(f"{role}/backbone")
                    )
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

    def _pooling_levels_hook(self, name):
        """Create a forward hook that logs avg points/event at each pooling stage."""
        def hook_fn(module, input, output):
            # Only log during training steps; eval/probe passes also call the
            # backbone but are not what we want to characterize here.
            if not getattr(self.trainer.model, "training", False):
                return
            point = output[0] if isinstance(output, tuple) else output
            if not hasattr(point, "keys") or "feat" not in point:
                return

            key = f"pooling::{name}"
            self._step_counters[key] = self._step_counters.get(key, 0) + 1
            if self._step_counters[key] % self.log_frequency != 0:
                return

            stats = self._get_pooling_stats(point)
            if not stats:
                return

            if hasattr(self.trainer, "writer") and self.trainer.writer is not None:
                current_iter = self.trainer.comm_info.get("iter", 0) + 1
                current_epoch = self.trainer.epoch + 1
                global_step = (current_epoch - 1) * len(self.trainer.train_loader) + current_iter
                for level, value in stats.items():
                    self.trainer.writer.add_scalar(
                        f"{self.prefix}/{name}/points_per_event/level{level}",
                        value,
                        global_step,
                    )

        return hook_fn

    def _get_pooling_stats(self, point):
        """Average points per event at each pooling stage (cross-rank).

        Walks the ``pooling_parent`` chain WITHOUT consuming it (the model still
        needs it for up-casting). ``level0`` is the finest input resolution and
        the level index grows with pooling depth, so the largest level index is
        the most aggressively pooled stage.
        """
        import torch.distributed as dist
        from pimm.utils.comm import get_world_size

        with torch.no_grad():
            # Walk from the returned (coarsest) point to the finest parent.
            chain = [point]
            cur = point
            while "pooling_parent" in cur.keys():
                cur = cur["pooling_parent"]
                chain.append(cur)
            levels = list(reversed(chain))  # finest first

            world_size = get_world_size()
            stats = {}
            for k, lvl in enumerate(levels):
                n_points = float(lvl["feat"].shape[0])
                if "batch" in lvl.keys():
                    n_events = float(int(lvl["batch"].max().item()) + 1)
                elif "offset" in lvl.keys():
                    n_events = float(lvl["offset"].shape[0])
                else:
                    n_events = 1.0
                totals = torch.tensor(
                    [n_points, n_events], device=lvl["feat"].device, dtype=torch.float64
                )
                if world_size > 1:
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                total_points, total_events = totals[0].item(), totals[1].item()
                stats[k] = total_points / total_events if total_events > 0 else 0.0
        return stats

    def after_train(self):
        """Clean up hooks when training is done."""
        for handle in self.hook_handles:
            handle.remove()
    
    def __repr__(self):
        """Return a compact configuration summary for logs."""
        return (
            f"{self.__class__.__name__}(log_frequency={self.log_frequency}, "
            f"prefix='{self.prefix}', log_pooling_levels={self.log_pooling_levels})"
        )

@HOOKS.register_module()
class FeatureStdMonitor(HookBase):
    """Monitor feature standard deviation to detect representation collapse.

    In ``before_train`` (after unwrapping DDP) it locates the student/teacher
    module (Sonata, JEPA, etc.) and registers forward hooks on the selected
    backbones (and their ``representation_fusion`` if present). Each forward
    hook fires every ``log_frequency`` calls (training mode only) and computes,
    with cross-rank synchronization, the global feature std, per-sample batch
    std, and per-channel std summary (mean/min/max), logging them to the writer
    under ``{prefix}/<module>/...``; with ``track_channels=True`` it also logs
    each channel's std. ``after_train`` removes all hooks. Useful for catching
    feature collapse (std trending to zero). Registered as ``FeatureStdMonitor``.

    Args:
        log_frequency (int): Log statistics every this many forward passes.
            Defaults to ``10``.
        prefix (str): Namespace prefix for the logged keys. Defaults to
            ``"feature_std"``.
        monitor_student (bool): Register hooks on the student branch. Defaults
            to ``True``.
        monitor_teacher (bool): Register hooks on the teacher branch. Defaults
            to ``True``.
        track_channels (bool): If ``True``, additionally log per-channel std.
            Defaults to ``False``.

    Note:
        Statistics are computed inside the forward pass to avoid retaining large
        feature tensors. Warns and does nothing if no student/teacher module is
        found.

    Example:
        Add to ``cfg.hooks`` to catch representation collapse during SSL; it
        watches the std of teacher/student backbone features:

        .. code-block:: python

            hooks = [dict(type="FeatureStdMonitor", log_frequency=20)]
            # → every 20 forward passes writes "feature_std/<student|teacher>/
            #   {global_std,batch_std,channel_mean_std,channel_min_std,
            #   channel_max_std}" to the writer (std trending to 0 signals collapse)
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
    """Log a parameter-count breakdown of the model at the start of training.

    Runs once in ``before_train``: logs total / trainable / non-trainable
    parameter counts and an estimated (fp32) memory footprint. With
    ``show_details=True`` it logs a per-module breakdown (optionally sorted by
    size, filtered by ``min_params``) and the ten largest modules; with
    ``show_gradients=True`` it lists how many parameters require gradients and
    which modules contain frozen parameters. Output goes to the trainer logger
    only. Registered as ``ParameterCounter``.

    Args:
        show_details (bool): Show the per-module parameter breakdown. Defaults
            to ``True``.
        show_gradients (bool): Show gradient-requirement and frozen-module info.
            Defaults to ``True``.
        sort_by_params (bool): Sort the module breakdown by parameter count
            (descending). Defaults to ``True``.
        min_params (int): Omit modules with fewer than this many parameters from
            the breakdown. Defaults to ``0``.

    Note:
        Purely diagnostic — logs to the console/logger, not the writer. The DDP
        wrapper is unwrapped before counting.

    Example:
        Add to ``cfg.hooks``; once in ``before_train`` it logs a parameter
        breakdown to the trainer logger (not the writer):

        .. code-block:: python

            hooks = [dict(type="ParameterCounter", min_params=1000)]
            # → logs total/trainable/non-trainable param counts, an fp32 memory
            #   estimate, a per-module table (modules with >=1000 params), the top
            #   10 largest modules, and which modules hold frozen params
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
    """Drive and log attention-mask annealing for the Panda detector decoder.

    For models that gradually anneal their decoder attention masks (e.g. a
    Mask3Former-style decoder). In ``before_train`` it seeds the step counter
    from resumed progress, checks whether the model exposes
    ``update_anneal_step`` (disabling itself otherwise), and logs the annealing
    schedule. In ``after_step`` it advances the model's annealing step every
    step and, every ``log_frequency`` steps, reads each decoder block's anneal
    factor to log the average (and per-block factors when
    ``log_per_layer=True``) to ``trainer.storage`` and the writer, announcing
    once when annealing completes. Registered as ``AttentionMaskAnnealingHook``.

    Args:
        log_frequency (int): Log annealing factors every this many steps.
            Defaults to ``100``.
        log_per_layer (bool): If ``True``, also log each decoder block's anneal
            factor. Defaults to ``False``.
        prefix (str): Namespace prefix for the logged keys. Defaults to
            ``"anneal"``.

    Note:
        This hook *updates* the model's annealing state, so it is functionally
        required (not merely diagnostic) for models that anneal. It silently
        disables itself if the model lacks ``update_anneal_step`` or has
        annealing turned off.

    Example:
        Add to ``cfg.hooks`` for a Mask3Former-style Panda decoder; it both
        drives and logs attention-mask annealing:

        .. code-block:: python

            hooks = [
                dict(type="AttentionMaskAnnealingHook",
                     log_frequency=100, log_per_layer=True),
            ]
            # → calls model.update_anneal_step(step) every step; every 100 steps
            #   puts "anneal_factor" in trainer.storage and writes "anneal/average"
            #   (and "anneal/layer_<i>" per block) to the writer; logs once when the
            #   factor drops below 0.01 (annealing complete)
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
