"""
General checkpoint loading utilities for PointModule models.

Supports loading pretrained weights and fine-tuned checkpoints with flexible
key matching and prefix remapping for partial loading scenarios.
"""
import os
import logging
from typing import Dict, Any, Optional, Union, Callable

import torch

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

_logger = logging.getLogger(__name__)


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove DDP and torch.compile prefixes from state dict keys.
    
    This ensures compatibility between single-GPU and multi-GPU checkpoints.
    
    Args:
        state_dict: State dictionary with potentially prefixed keys.
    
    Returns:
        Cleaned state dictionary.
    """
    cleaned_state_dict = {}
    to_remove = ('module.', '_orig_mod.')
    for k, v in state_dict.items():
        for prefix in to_remove:
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned_state_dict[k] = v
    return cleaned_state_dict


def load_checkpoint_metadata(
    checkpoint_path: str,
    device: Union[str, torch.device] = 'cpu',
    weights_only: bool = False,
) -> Dict[str, Any]:
    """Load checkpoint metadata (everything except state_dict).
    
    Returns metadata dict with keys like 'epoch', 'config', 'args', etc.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
    
    if checkpoint_path.endswith(".safetensors"):
        # Safetensors doesn't support metadata in the same way
        return {}
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if not isinstance(checkpoint, dict):
        return {}
    
    # Extract metadata (everything except state dicts)
    metadata = {}
    state_dict_keys = ('state_dict', 'state_dict_ema', 'model_ema', 'model', 'optimizer', 'scheduler', 'scaler')
    for k, v in checkpoint.items():
        if k not in state_dict_keys:
            metadata[k] = v
    
    return metadata


def load_state_dict_from_checkpoint(
    checkpoint_path: str,
    device: Union[str, torch.device] = 'cpu',
    weights_only: bool = False,
) -> Dict[str, Any]:
    """Load state dictionary from checkpoint file.
    
    Automatically detects checkpoint format and extracts the state dict.
    Supports both pretrained weights (just state dict) and training checkpoints
    (dict with metadata).
    
    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load checkpoint to.
        weights_only: Whether to load only weights (torch.load parameter).
    
    Returns:
        State dictionary extracted from checkpoint.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
    
    # Load checkpoint file
    if checkpoint_path.endswith(".safetensors"):
        if not _has_safetensors:
            raise ImportError("safetensors package required for .safetensors files. Install with: pip install safetensors")
        checkpoint = safetensors.torch.load_file(checkpoint_path, device=device)
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict from checkpoint
    state_dict = None
    if isinstance(checkpoint, dict):
        # Try different possible keys for state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # Checkpoint might be the state dict itself (pretrained weights)
            state_dict = checkpoint
    else:
        # Checkpoint is directly the state dict
        state_dict = checkpoint
    
    # Clean state dict keys (remove DDP/compile prefixes)
    state_dict = clean_state_dict(state_dict)
    
    return state_dict


def filter_state_dict_by_prefix(
    state_dict: Dict[str, Any],
    prefix: str,
    remove_prefix: bool = True,
) -> Dict[str, Any]:
    """Filter state dict to only include keys with given prefix.
    
    Args:
        state_dict: Source state dictionary.
        prefix: Prefix to match (e.g., "backbone." or "student.backbone.").
        remove_prefix: If True, remove the prefix from keys in output.
    
    Returns:
        Filtered state dictionary.
    """
    filtered = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):] if remove_prefix else k
            filtered[new_key] = v
    return filtered


def remap_state_dict_keys(
    state_dict: Dict[str, Any],
    key_mapping: Dict[str, str],
) -> Dict[str, Any]:
    """Remap state dict keys according to mapping.
    
    Args:
        state_dict: Source state dictionary.
        key_mapping: Dictionary mapping old keys to new keys.
    
    Returns:
        Remapped state dictionary.
    """
    remapped = {}
    for k, v in state_dict.items():
        new_key = key_mapping.get(k, k)
        remapped[new_key] = v
    return remapped


def load_pretrained(
    model: torch.nn.Module,
    checkpoint_path: str,
    prefix: Optional[str] = None,
    remove_prefix: bool = True,
    key_mapping: Optional[Dict[str, str]] = None,
    strict: bool = True,
    device: Union[str, torch.device] = 'cpu',
    filter_fn: Optional[Callable[[Dict[str, Any], torch.nn.Module], Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Load pretrained weights into model with flexible key matching.
    
    This function supports various loading scenarios:
    - Full checkpoint loading (all keys match)
    - Partial loading with prefix filtering (e.g., load only "backbone." weights)
    - Key remapping (e.g., map "student.backbone." to "backbone.")
    - Custom filtering via filter_fn
    
    Examples:
        # Load full checkpoint
        load_pretrained(model, 'checkpoint.pth')
        
        # Load only backbone weights from Sonata checkpoint
        load_pretrained(model, 'sonata_checkpoint.pth', prefix='student.backbone.')
        
        # Load backbone weights and remap keys
        load_pretrained(
            model, 
            'sonata_checkpoint.pth',
            prefix='student.backbone.',
            key_mapping={'backbone.': ''}  # Remove 'backbone.' prefix
        )
    
    Args:
        model: Model to load weights into.
        checkpoint_path: Path to checkpoint file.
        prefix: Optional prefix to filter keys (e.g., "backbone.", "student.backbone.").
            If provided, only keys starting with this prefix will be loaded.
        remove_prefix: If True and prefix is provided, remove prefix from keys.
        key_mapping: Optional dictionary to remap keys (old_key -> new_key).
        strict: Whether to strictly enforce all keys match.
        device: Device to load checkpoint to.
        filter_fn: Optional function(state_dict, model) -> filtered_state_dict
            for custom filtering logic.
    
    Returns:
        Incompatible keys (missing_keys, unexpected_keys) if strict=False, None otherwise.
    """
    # Load state dict from checkpoint
    state_dict = load_state_dict_from_checkpoint(
        checkpoint_path,
        device=device,
    )
    
    # Apply prefix filtering if specified
    if prefix is not None:
        state_dict = filter_state_dict_by_prefix(
            state_dict,
            prefix=prefix,
            remove_prefix=remove_prefix,
        )
        if not state_dict:
            _logger.warning(f"No keys found with prefix '{prefix}' in checkpoint")
            return None
    
    # Apply key remapping if specified
    if key_mapping is not None:
        state_dict = remap_state_dict_keys(state_dict, key_mapping)
    
    # Apply custom filter function if specified
    if filter_fn is not None:
        state_dict = filter_fn(state_dict, model)
    
    # Load into model
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    
    if not strict and incompatible_keys:
        missing_keys = incompatible_keys.missing_keys
        unexpected_keys = incompatible_keys.unexpected_keys
        if missing_keys:
            _logger.warning(f"Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            _logger.warning(f"Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    
    return incompatible_keys if not strict else None

