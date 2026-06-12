# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Make the perf_regression_gate package (and tools/) importable in tests."""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _TESTS_DIR.parent
_TOOLS_DIR = _MODULE_DIR.parent

for path in (_TESTS_DIR, _MODULE_DIR, _TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
