# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Shadow Hand handover environment configuration."""

import pytest

from isaaclab_tasks.core.handover.handover_env_cfg import HandoverEnvCfg
from isaaclab_tasks.core.handover.handover_manager_env_cfg import HandoverManagerEnvCfg
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg


def test_hands_use_standard_shadow_assets():
    """Verify both hands reuse the single-agent Shadow Hand assets per backend."""
    cfg = HandoverEnvCfg()
    single_agent_robot_cfg = ShadowHandEnvCfg().robot_cfg

    for hand in (cfg.right_robot_cfg, cfg.left_robot_cfg):
        assert hand.physx.spawn.usd_path == single_agent_robot_cfg.physx.spawn.usd_path
        assert hand.newton_mjwarp.spawn.usd_path == single_agent_robot_cfg.newton_mjwarp.spawn.usd_path
        assert hand.ovphysx.spawn.usd_path == single_agent_robot_cfg.ovphysx.spawn.usd_path
        assert hand.newton_mjwarp.spawn.fixed_tendons_props.damping == 0.1


def test_hands_place_each_backend_with_the_same_pose():
    """Verify each hand uses one init pose across backends and normalized rotations."""
    cfg = HandoverEnvCfg()

    for hand in (cfg.right_robot_cfg, cfg.left_robot_cfg):
        assert hand.newton_mjwarp.init_state.pos == hand.physx.init_state.pos == hand.ovphysx.init_state.pos
        assert hand.physx.init_state.rot == hand.ovphysx.init_state.rot
        assert sum(value**2 for value in hand.newton_mjwarp.init_state.rot) == pytest.approx(1.0)

    assert cfg.right_robot_cfg.physx.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert cfg.left_robot_cfg.physx.init_state.rot == (0.0, 0.0, 1.0, 0.0)

    manager_cfg = HandoverManagerEnvCfg()
    manager_scene = manager_cfg.scene.newton_mjwarp
    assert manager_scene.right_hand.newton_mjwarp.init_state.rot == cfg.right_robot_cfg.newton_mjwarp.init_state.rot
    assert manager_scene.left_hand.newton_mjwarp.init_state.rot == cfg.left_robot_cfg.newton_mjwarp.init_state.rot


def test_newton_hands_override_passive_distal_joints():
    """Verify Newton hands keep the near-passive distal-joint override."""
    cfg = HandoverEnvCfg()

    for hand in (cfg.right_robot_cfg, cfg.left_robot_cfg):
        assert set(hand.newton_mjwarp.actuators) == {"fingers", "distal_passive"}
        assert hand.newton_mjwarp.actuators["distal_passive"].stiffness == 10.0
