"""Native training entrypoint for pimm configs.

This module is executed under ``torchrun`` for local and Slurm jobs. The public
``pimm launch`` and ``pimm submit`` commands render the torchrun invocation
directly; each process parses the same config, initializes distributed state
from the environment when available, builds the configured trainer, and runs
training.

Modified from the original Pointcept ``tools/train.py``.
"""

import sys
import os
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pimm.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pimm.engines.train import TRAINERS
from pimm.utils import comm


def main_worker(cfg):
    """Build and run the trainer after config normalization."""
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()

def main():
    """Parse CLI args, initialize distributed state, and start training."""
    logging.basicConfig(level=logging.INFO)
    
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    try:
        comm.setup_distributed()
        main_worker(cfg)
    finally:
        comm.cleanup_distributed()

if __name__ == "__main__":
    main()
