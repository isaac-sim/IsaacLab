# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the custom coupling environment configuration."""

from isaaclab_contrib.custom_coupling.franka_soft_env_cfg import FrankaSoftCustomCouplingEnvCfg

from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg
from isaaclab_tasks.utils.hydra import resolve_presets


def test_default_preset_resolves_to_multiple_worlds() -> None:
    """The manual coupler keeps reset state per world, unlike the proxy default."""
    env_cfg = resolve_presets(FrankaSoftCustomCouplingEnvCfg(), selected=())

    assert env_cfg.scene.num_envs == 128
    assert env_cfg.scene.replicate_physics
    assert env_cfg.sim.physics.class_type == (
        "isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager:NewtonCoupledMJWarpVBDManager"
    )


def test_core_task_default_preset_stays_single_world() -> None:
    """Overriding the scene preset in the example must not leak into the core task."""
    env_cfg = resolve_presets(FrankaSoftEnvCfg(), selected=())

    assert env_cfg.scene.num_envs == 1
