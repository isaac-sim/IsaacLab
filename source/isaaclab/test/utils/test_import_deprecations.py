# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public import deprecation behavior across module and symbol forwarding paths."""

import subprocess
import sys

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_deprecated_imports_warn_once_at_caller_with_removal_version():
    script = """
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", DeprecationWarning)
    from isaaclab.utils import modifiers
    from isaaclab.utils.warp import kernels
    old_wrench = kernels.compose_wrench_to_body_frame
    assert kernels.compose_wrench_to_body_frame is old_wrench
    import isaaclab.sim as sim
    old_schema = sim.NewtonRigidBodyPropertiesCfg
    assert sim.NewtonRigidBodyPropertiesCfg is old_schema

warnings_by_name = {
    name: [warning for warning in caught if warning.category is DeprecationWarning and name in str(warning.message)]
    for name in (
        "isaaclab.utils.modifiers is deprecated",
        "isaaclab.utils.warp.kernels.compose_wrench_to_body_frame is deprecated",
        "isaaclab.sim.NewtonRigidBodyPropertiesCfg is deprecated",
    )
}
assert all(len(warnings) == 1 for warnings in warnings_by_name.values()), warnings_by_name
assert all(warnings[0].filename == "<string>" for warnings in warnings_by_name.values())
messages = [str(warnings[0].message) for warnings in warnings_by_name.values()]
assert "future release" in messages[0]
assert "Isaac Lab 3.2" in messages[1]
assert "Isaac Lab 4.0" in messages[2]
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
