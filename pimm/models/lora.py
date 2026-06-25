"""LoRA adapter wrapper for parameter-efficient fine-tuning.

`LoRAAdapter` wraps an already-built model (a segmentor, classifier, detector,
... anything `build_model` produces), freezes all of its weights, and injects
low-rank adapters into the attention projection layers of every attention block.
Only the LoRA parameters (and any explicitly whitelisted modules, e.g. the task
head) remain trainable.

Usage (config) -- LoRA the backbone attention of a semseg model, train the head:

    model = dict(
        type="LoRAAdapter",
        rank=8,
        alpha=16,
        dropout=0.0,
        target_modules=("attn.qkv", "attn.proj"),  # PT-v3 attention projections
        trainable_keywords=("seg_head",),          # keep the readout trainable
        model=dict(
            type="DefaultSegmentorV2",
            num_classes=5,
            backbone_out_channels=64,
            backbone=dict(type="PT-v3m2", ...),
            criteria=[...],
        ),
    )

Note: the wrapped model now lives under ``.model``, so a warm-start
``CheckpointLoader`` must target ``module.model.backbone`` (not ``module.backbone``).
"""

import math

import torch
import torch.nn as nn

from pimm.utils.logger import get_root_logger

from .builder import MODELS, build_model


class LoRALinear(nn.Linear):
    """A frozen ``nn.Linear`` with an additive low-rank update.

    y = W0 x + b0 + (alpha / rank) * (B @ A) x

    Subclasses ``nn.Linear`` and REUSES the original ``weight`` / ``bias``
    Parameters, so the base weights keep their original dotted names
    (``...attn.qkv.weight``, not ``...attn.qkv.base.weight``). That matters: a
    pretrained checkpoint loads straight into the adapted layers, and only the
    extra ``lora_A`` / ``lora_B`` keys are new.

    ``A`` is kaiming-initialised and ``B`` is zero, so at initialisation the
    adapter is an exact no-op and the wrapped model reproduces its pretrained
    outputs. Only ``lora_A`` / ``lora_B`` are trainable; ``weight`` / ``bias``
    are frozen.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        # reuse (not copy) the pretrained Parameters and freeze them
        self.weight = base.weight
        self.weight.requires_grad_(False)
        if base.bias is not None:
            self.bias = base.bias
            self.bias.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        delta = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + delta * self.scaling

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, lora_rank={self.rank}, scaling={self.scaling:.3g}"


@MODELS.register_module()
class LoRAAdapter(nn.Module):
    """Freeze a built model and LoRA-tune its attention blocks.

    Args:
        model: config of the inner model to build and adapt.
        rank: LoRA rank for every injected adapter.
        alpha: LoRA scaling numerator (effective scale = alpha / rank).
        dropout: dropout applied to the LoRA branch input.
        target_modules: a Linear module is adapted if its dotted name equals or
            ends with ".<entry>" (anchored on a path-component boundary). Default
            targets PT-v3 attention projections ("...block.attn.qkv/proj"); the
            anchoring means it will NOT match a decoder's "self_attn.proj".
        trainable_keywords: parameters whose dotted name contains any of these
            substrings stay trainable (e.g. the task head). Everything else in the
            base model is frozen. Empty => only LoRA params train.
    """

    def __init__(
        self,
        model,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules=("attn.qkv", "attn.proj"),
        trainable_keywords=(),
    ):
        super().__init__()
        self.model = build_model(model)
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = tuple(target_modules)
        self.trainable_keywords = tuple(trainable_keywords)

        # 1) freeze the entire base model
        for p in self.model.parameters():
            p.requires_grad_(False)

        # 2) inject LoRA into every targeted attention projection
        self._adapted = self._inject_lora()

        # 3) re-enable any explicitly whitelisted modules (e.g. the readout head)
        if self.trainable_keywords:
            for name, p in self.model.named_parameters():
                if any(k in name for k in self.trainable_keywords):
                    p.requires_grad_(True)

        self._log_summary()

    def _inject_lora(self):
        """Replace targeted ``nn.Linear`` modules in-place with ``LoRALinear``."""
        targets = []
        for name, module in self.model.named_modules():
            # anchored suffix match on dotted-path components, so "attn.proj"
            # matches "...block0.attn.proj" but NOT "...decoder.self_attn.proj".
            if isinstance(module, nn.Linear) and any(
                name == t or name.endswith("." + t) for t in self.target_modules
            ):
                targets.append(name)

        if not targets:
            raise ValueError(
                f"LoRAAdapter found no nn.Linear matching target_modules="
                f"{self.target_modules}. Check the module names of the built model."
            )

        for name in targets:
            parent_name, _, child_name = name.rpartition(".")
            parent = self.model.get_submodule(parent_name) if parent_name else self.model
            base = getattr(parent, child_name)
            setattr(
                parent,
                child_name,
                LoRALinear(base, rank=self.rank, alpha=self.alpha, dropout=self.dropout),
            )
        return targets

    def _log_summary(self):
        logger = get_root_logger()
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"LoRAAdapter: rank={self.rank} alpha={self.alpha} -> adapted "
            f"{len(self._adapted)} attention projections; trainable "
            f"{n_train:,}/{n_total:,} params ({100.0 * n_train / max(n_total, 1):.2f}%)"
        )

    def forward(self, *args, **kwargs):
        # Transparent pass-through to the base model
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        # Make the wrapper transparent: attributes/methods not defined on the
        # adapter (e.g. detector ``postprocess``, ``criteria``, ``label_specs``)
        # resolve on the wrapped model, so evaluators/testers that call
        # ``unwrap_model(model).postprocess(...)`` keep working.
        try:
            return super().__getattr__(name)  # nn.Module: params/buffers/submodules
        except AttributeError:
            if name == "model":
                raise
            return getattr(self.model, name)
