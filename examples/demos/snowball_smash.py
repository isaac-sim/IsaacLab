# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated entry point; use ``examples/snowball_smash.py`` instead."""

import runpy
import warnings
from pathlib import Path

if __name__ == "__main__":
    warnings.warn(
        "examples/demos/snowball_smash.py is deprecated; use examples/snowball_smash.py instead.",
        FutureWarning,
        stacklevel=1,
    )
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "snowball_smash.py"), run_name="__main__")
