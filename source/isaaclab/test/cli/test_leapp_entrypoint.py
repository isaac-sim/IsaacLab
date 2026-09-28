# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the installed LEAPP commands."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_deploy_module_does_not_import_pxr_before_app_launch():
    """Importing deploy must not load pxr before SimulationApp starts."""
    env = os.environ.copy()
    env.update({"ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "Y"})
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import isaaclab.cli.commands.deploy; assert 'pxr' not in sys.modules"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
