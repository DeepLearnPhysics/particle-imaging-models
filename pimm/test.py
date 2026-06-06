"""Native testing entrypoint for pimm configs.

This mirrors :mod:`pimm.train`: every launched process parses the same config,
initializes distributed state from the environment, builds the configured
tester, and runs evaluation.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
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
from pimm.engines.test import TESTERS
from pimm.utils import comm


def main_worker(cfg):
    """Build and run the configured tester after config normalization."""
    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()


def main():
    """Parse CLI args, initialize distributed state, and start testing."""
    logging.basicConfig(level=logging.INFO)
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options, save_artifacts=False)

    comm.setup_distributed()
    main_worker(cfg)


if __name__ == "__main__":
    main()
