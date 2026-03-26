import sys
from typing import Optional, Callable, Dict, Any
import torch.nn as nn
import spconv.pytorch as spconv
import torch

try:
    import ocnn
except ImportError:
    ocnn = None

from collections import OrderedDict
from pimm.models.utils.structure import Point
from pimm.engines.hooks import HookBase


def is_ocnn_module(module):
    ocnn_modules = (
        ocnn.nn.OctreeConv,
        ocnn.nn.OctreeDeconv,
        ocnn.nn.OctreeGroupConv,
        ocnn.nn.OctreeDWConv,
    )
    return isinstance(module, ocnn_modules)


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Optional[str] = None,
        exp_dir: Optional[str] = None,
        sparsify: Optional[str] = None,
        prefix: Optional[str] = None,
        remove_prefix: bool = True,
        key_mapping: Optional[Dict[str, str]] = None,
        strict: bool = True,
        device: str = 'cpu',
        filter_fn: Optional[Callable[[Dict[str, Any], nn.Module], Dict[str, Any]]] = None,
        config: Optional[Any] = None,
        config_path: Optional[str] = None,
        **model_kwargs,
    ):
        """Create model instance and load pretrained weights.
        
        This is a general method that works for loading both pretrained weights
        and fine-tuned checkpoints. It supports partial loading with prefix
        filtering and key remapping.
        
        Config handling:
        - If `config` is provided, it takes precedence
        - If `config_path` is provided, config is loaded from that file
        - If checkpoint contains 'config' or 'args', it's used as default
        - `model_kwargs` override any config values
        - If no config is available, model_kwargs must provide all required args
        
        Examples:
            # Load with explicit config
            from pimm.utils.config import Config
            cfg = Config.fromfile('config.py')
            model = PointTransformerV3.from_pretrained(
                'checkpoint.pth',
                config=cfg.model,
            )
            
            # Load config from file
            model = PointTransformerV3.from_pretrained(
                'checkpoint.pth',
                config_path='exp/scannet/debug/config.py',
            )
            
            # Load with model_kwargs (config inferred from checkpoint if available)
            model = PointTransformerV3.from_pretrained(
                'checkpoint.pth',
                in_channels=6,
                enc_channels=(32, 64, 128, 256, 512),
            )
            
            # Load backbone from Sonata checkpoint
            model = PointTransformerV3.from_pretrained(
                'sonata_checkpoint.pth',
                prefix='student.backbone.',
                config=cfg.model.backbone,  # Backbone config
            )
        
        Args:
            checkpoint_path: Path to checkpoint file.
            prefix: Optional prefix to filter keys (e.g., "backbone.", "student.backbone.").
                If provided, only keys starting with this prefix will be loaded.
            remove_prefix: If True and prefix is provided, remove prefix from keys.
            key_mapping: Optional dictionary to remap keys (old_key -> new_key).
            strict: Whether to strictly enforce all keys match.
            device: Device to load checkpoint to.
            filter_fn: Optional function(state_dict, model) -> filtered_state_dict
                for custom filtering logic.
            config: Optional config object/dict to use for model construction.
                If provided, model_kwargs override config values.
            config_path: Optional path to config file. Loaded if config is not provided.
            **model_kwargs: Additional arguments passed to model constructor.
                These override config values if both are provided.
        
        Returns:
            Model instance with loaded weights.
        """
        from pimm.models.utils.checkpoint_loader import load_pretrained, load_checkpoint_metadata
        import os
        
        # Handle config loading
        final_config = {}
        
        # 1. Try to load config from checkpoint metadata
        try:
            metadata = load_checkpoint_metadata(checkpoint_path, device=device)
            if 'config' in metadata:
                checkpoint_config = metadata['config']
                if isinstance(checkpoint_config, dict):
                    final_config.update(checkpoint_config)
            elif 'args' in metadata:
                # timm-style args, convert to dict
                args = metadata['args']
                if hasattr(args, '__dict__'):
                    final_config.update(vars(args))
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Could not load config from checkpoint: {e}")
        
        # 2. Load config from file if provided
        if config is None and config_path is not None:
            if os.path.isfile(config_path):
                file_config = None
                
                # Try OmegaConf for JSON/YAML files
                if config_path.endswith(('.json', '.yaml', '.yml')):
                    import json
                    import yaml
                    with open(config_path, 'r') as f:
                        if config_path.endswith('.json'):
                            file_config = json.load(f)
                        else:
                            file_config = yaml.safe_load(f)
                
                # Fall back to Config system for .py files
                if file_config is None:
                    from pimm.utils.config import Config
                    file_config = Config.fromfile(config_path)
                
                # Extract model config if it's a full config
                if isinstance(file_config, dict):
                    if 'model' in file_config:
                        file_config = file_config['model']
                elif hasattr(file_config, 'model'):
                    file_config = file_config.model
                
                # Convert to dict if needed
                if hasattr(file_config, '__dict__'):
                    final_config.update(vars(file_config))
                elif isinstance(file_config, dict):
                    final_config.update(file_config)
        
        # 3. Use provided config object
        if config is not None:
            if hasattr(config, '__dict__'):
                final_config.update(vars(config))
            elif isinstance(config, dict):
                final_config.update(config)
            else:
                # Try to convert to dict
                final_config.update(dict(config))
        
        # 4. model_kwargs override everything
        final_config.update(model_kwargs)
        
        # Create model instance
        final_config.pop("type", None)
        model = cls(**final_config)
        
        # Load pretrained weights
        load_pretrained(
            model,
            checkpoint_path,
            prefix=prefix,
            remove_prefix=remove_prefix,
            key_mapping=key_mapping,
            strict=strict,
            device=device,
            filter_fn=filter_fn,
        )
        
        return model


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            torch.cuda.synchronize()
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            elif is_ocnn_module(module):
                if isinstance(input, Point):
                    input.octree.features[-1] = module(
                        input.feat[input.octree_order], input.octree, input.octree.depth
                    )
                    input.feat = input.octree.features[-1][input.octree_inverse]
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)

            torch.cuda.synchronize()
        return input


class PointModel(PointModule, HookBase):
    r"""PointModel
    placeholder, PointModel can be customized as a pimm hook.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
