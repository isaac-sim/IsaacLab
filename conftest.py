# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared pytest configuration for repository tests.

Wires the Testmon subprocess-coverage hooks (see :mod:`tools.testmon_subprocess_coverage`)
into the session so that code executed in child Python processes is attributed to the
active test. The hooks are no-ops unless pytest-testmon is collecting coverage.
"""

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent / "tools"
if _TOOLS_DIR.is_dir() and str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from testmon_subprocess_coverage import (  # noqa: E402, F401
    pytest_runtest_makereport,
    pytest_runtest_setup,
    pytest_runtest_teardown,
    pytest_sessionfinish,
    pytest_sessionstart,
)
