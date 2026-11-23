# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated wrapper for :mod:`isaaclab_contrib.rl.ray.submit_job`."""

import sys
from pathlib import Path

if __package__:
    from ._compat import forward
else:
    from _compat import forward

if __name__ == "__main__":
    # Keep adjacent task configs and Python modules in the uploaded directory.
    # An explicit --working_dir supplied by the caller takes precedence.
    sys.argv[1:1] = ["--working_dir", str(Path(__file__).resolve().parent)]

forward("isaaclab_contrib.rl.ray.submit_job", globals())
