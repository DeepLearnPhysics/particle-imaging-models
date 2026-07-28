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

# Drop the script dir (the `pimm/` package dir) from sys.path so pimm submodules
# don't shadow installed distributions (e.g. `datasets` -> HuggingFace, not
# `pimm.datasets`); make the repo root importable instead.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _script_dir]
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from pimm.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pimm.engines.train import TRAINERS
from pimm.observability import structured_logger as sl
from pimm.utils import comm


def main_worker(cfg):
    """Build and run the trainer after config normalization."""
    cfg = default_setup(cfg)
    with sl.log_trace_span("trainer_build"):
        trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    with sl.log_trace_span("training"):
        trainer.train()


def main():
    """Parse CLI args, initialize distributed state, and start training."""
    logging.basicConfig(level=logging.INFO)

    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    sl.init_structured_logger(
        source="training",
        output_dir=cfg.save_path,
        enable=cfg.structured_logging.enabled,
        max_file_size_mb=cfg.structured_logging.max_file_size_mb,
        backup_count=cfg.structured_logging.backup_count,
    )
    sl.log_trace_instant("process_start")

    try:
        with sl.log_trace_span("distributed_setup"):
            comm.setup_distributed()
        with sl.log_trace_span("trainer_lifecycle"):
            main_worker(cfg)
    finally:
        try:
            with sl.log_trace_span("distributed_cleanup"):
                comm.cleanup_distributed()
        finally:
            try:
                sl.log_trace_instant("process_end")
            finally:
                sl.shutdown_structured_logger()


if __name__ == "__main__":
    main()
