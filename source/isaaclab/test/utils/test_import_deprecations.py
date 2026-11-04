# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public import deprecation behavior for backend forwarding paths."""

import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_forwarded_import_warns_once_at_caller_with_removal_version():
    script = """
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", DeprecationWarning)
    from isaaclab.sim import NewtonRigidBodyPropertiesCfg
    import isaaclab.sim as sim
    assert sim.NewtonRigidBodyPropertiesCfg is NewtonRigidBodyPropertiesCfg

warnings_for_import = [
    warning for warning in caught
    if warning.category is DeprecationWarning
    and "isaaclab.sim.NewtonRigidBodyPropertiesCfg is deprecated" in str(warning.message)
]
assert len(warnings_for_import) == 1, warnings_for_import
assert warnings_for_import[0].filename == "<string>"
assert "Isaac Lab 4.0" in str(warnings_for_import[0].message)
assert "isaaclab_newton.sim.schemas" in str(warnings_for_import[0].message)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
