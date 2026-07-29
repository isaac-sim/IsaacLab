# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest configuration for OdinV2's offline test suite.

Puts the repository root on ``sys.path`` so ``tools.odin.*`` resolves when
pytest is invoked with ``--confcutdir`` pointed at this directory, which
bypasses the repository-level ``tools/conftest.py`` CI orchestrator.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
