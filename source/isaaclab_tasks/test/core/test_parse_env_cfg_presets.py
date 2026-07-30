# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that :func:`parse_env_cfg` honors a requested preset selection.

A selection has to be supplied while the ``PresetCfg`` wrappers are being resolved: the returned
config no longer carries the alternatives, so applying a preset afterwards silently keeps the
default. These cases use tasks whose default backend is *not* the requested one, so a dropped
selection fails instead of coincidentally matching.
"""

import pytest
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

pytestmark = pytest.mark.unit

# Defaults to PhysX and offers a newton_mjwarp preset, so the selection has to do real work.
_PHYSX_DEFAULT_TASK = "Isaac-Velocity-Flat-G1"


def test_parse_env_cfg_defaults_to_the_preset_default() -> None:
    """Without a selection, a wrapper still resolves to its default."""
    env_cfg = parse_env_cfg(_PHYSX_DEFAULT_TASK)

    assert isinstance(env_cfg.sim.physics, PhysxCfg)


def test_parse_env_cfg_applies_requested_physics_preset() -> None:
    """A requested preset must win over the task's default backend."""
    env_cfg = parse_env_cfg(_PHYSX_DEFAULT_TASK, presets=("newton_mjwarp",))

    assert isinstance(env_cfg.sim.physics, NewtonCfg)


def test_parse_env_cfg_applies_preset_alongside_other_overrides() -> None:
    """Selecting a preset must not discard the device and num_envs overrides."""
    env_cfg = parse_env_cfg(_PHYSX_DEFAULT_TASK, device="cuda:0", num_envs=3, presets=("newton_mjwarp",))

    assert isinstance(env_cfg.sim.physics, NewtonCfg)
    assert env_cfg.sim.device == "cuda:0"
    assert env_cfg.scene.num_envs == 3
