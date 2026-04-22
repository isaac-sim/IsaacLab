# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
from importlib.metadata import version
from importlib.util import find_spec

__version__ = version("isaaclab")

# Extend the package search path so subpackages (app/, envs/, etc.) in the
# nested source tree are importable as isaaclab.app, isaaclab.envs, etc.
__path__.append(os.path.join(os.path.dirname(__file__), "source", "isaaclab", "isaaclab"))

# TODO(myurasov-nv): bootstrap_kernel() is ported from the internal GitLab wheel builder
# for backwards compatibility. It is not called currently, but may be needed if Isaac Sim
# requires explicit kernel bootstrapping before use. Remove once confirmed unnecessary.
def bootstrap_kernel():
    # Isaac Lab path
    isaaclab_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

    # bootstrap kernel via Isaac Sim
    if find_spec("isaacsim") is not None:
        import isaacsim

        # log info
        if find_spec("carb") is not None:
            import carb
            carb.log_info(f"Isaac Lab path: {isaaclab_path}")

def main():
    """Entry point for the ``isaaclab`` console script (python -m isaaclab)."""
    from isaaclab.__main__ import main as _main

    sys.exit(_main())


if __name__ == "__main__":
    bootstrap_kernel()
    main()
