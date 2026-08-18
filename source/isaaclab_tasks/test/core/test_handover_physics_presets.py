# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Shadow Hand handover physics presets."""

from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.core.handover.handover_env_cfg import HandoverEnvCfg, ObjectCfg
from isaaclab_tasks.utils.hydra import resolve_presets

from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG


def test_handover_isaacsim_physx_resolves_physx_assets() -> None:
    """The concrete PhysX selector must resolve the scene, hands, and object together."""
    env_cfg = resolve_presets(HandoverEnvCfg(), selected=("isaacsim_physx",))

    assert isinstance(env_cfg.sim.physics, PhysxCfg)
    assert env_cfg.right_robot_cfg.spawn.usd_path == SHADOW_HAND_CFG.spawn.usd_path
    assert env_cfg.left_robot_cfg.spawn.usd_path == SHADOW_HAND_CFG.spawn.usd_path
    assert env_cfg.object_cfg == ObjectCfg().physx
