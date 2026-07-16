# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for kitless construction of shared environment configurations."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

_FRESH_PROCESS_SCRIPT = """
import sys

from isaaclab.test.env_cfgs import (
    make_empty_direct_marl_env_cfg,
    make_empty_manager_based_env_cfg,
    make_empty_manager_based_rl_env_cfg,
)

factories = (
    make_empty_manager_based_env_cfg,
    make_empty_manager_based_rl_env_cfg,
    make_empty_direct_marl_env_cfg,
)
for factory in factories:
    cfg = factory(device="cpu", num_envs=2, env_spacing=1.5)
    cfg.validate()
    assert cfg.sim.device == "cpu"
    assert cfg.scene.num_envs == 2
    assert cfg.scene.env_spacing == 1.5

forbidden_roots = {
    "carb",
    "isaaclab_assets",
    "isaaclab_tasks",
    "isaaclab_tasks_experimental",
    "isaacsim",
    "omni",
    "pxr",
    "scipy",
}
forbidden_prefixes = (
    "isaaclab.actuators.actuator_base_cfg",
    "isaaclab.actuators.actuator_pd_cfg",
    "isaaclab.assets",
    "isaaclab.sim.spawners.from_files",
    "isaaclab.sim.spawners.shapes",
    "isaaclab.utils.assets",
    "isaaclab_physx.sim.schemas",
)
forbidden_modules = sorted(
    module_name
    for module_name in sys.modules
    if module_name.partition(".")[0] in forbidden_roots
    or any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in forbidden_prefixes
    )
)
assert not forbidden_modules, f"Forbidden modules loaded: {forbidden_modules}"
"""


def test_factories_are_kitless_in_fresh_process():
    """Every shared factory should validate without loading integration or runtime modules."""
    result = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    assert result.returncode == 0, output
