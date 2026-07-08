# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared Shadow Hand environment configuration."""

import pytest

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_env_cfg import (
    NewtonEventCfg,
    PhysxEventCfg,
    ShadowHandEnvCfg,
)

from isaaclab_assets import SHADOW_HAND_CFG


def test_events_randomize_fixed_tendons_on_both_backends():
    """Verify PhysX and Newton events preserve fixed-tendon randomization."""
    for event_cfg in (PhysxEventCfg(), NewtonEventCfg()):
        event = event_cfg.robot_tendon_properties

        assert event.func.__name__ == "randomize_fixed_tendon_parameters"
        assert event.params["asset_cfg"].fixed_tendon_names == ".*"


def test_shadow_variants_use_supported_asset_and_backend_physics_layers():
    """Verify each backend preset uses the production Shadow Hand asset it supports."""
    cfg = ShadowHandEnvCfg()
    legacy_path = f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowRobot/ShadowHand/shadow_hand_instanceable.usd"

    assert SHADOW_HAND_CFG.spawn.usd_path == legacy_path
    assert SHADOW_HAND_CFG.spawn.fixed_tendons_props is not None
    assert cfg.robot_cfg.physx.spawn.usd_path == legacy_path
    assert cfg.robot_cfg.ovphysx.spawn.usd_path == legacy_path
    assert cfg.robot_cfg.newton_mjwarp.spawn.usd_path == (
        f"{ISAAC_NUCLEUS_DIR}/Robots/ShadowRobot/ShadowHandNewton/shadow_hand_instanceable.usda"
    )
    assert cfg.robot_cfg.physx.spawn.variants is None
    assert cfg.robot_cfg.ovphysx.spawn.variants is None
    assert cfg.robot_cfg.newton_mjwarp.spawn.variants is None
    assert cfg.robot_cfg.ovphysx.spawn.fixed_tendons_props is None
    assert cfg.robot_cfg.physx.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert cfg.robot_cfg.ovphysx.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert cfg.robot_cfg.newton_mjwarp.spawn.fixed_tendons_props.damping == 0.1
    assert cfg.robot_cfg.newton_mjwarp.init_state.rot == (0.0, 0.0, -0.70710678118, 0.70710678118)
    assert sum(value**2 for value in cfg.robot_cfg.newton_mjwarp.init_state.rot) == pytest.approx(1.0)
    assert cfg.sim.physics.newton_mjwarp.num_substeps == 2


def test_shadow_joint_mapping_preserves_twenty_action_contract():
    """Verify the actuated-joint list preserves the legacy 20-action order."""
    cfg = ShadowHandEnvCfg()

    assert cfg.action_space == len(cfg.actuated_joint_names) == 20
    assert cfg.actuated_joint_names == [
        "robot0_WRJ1",
        "robot0_WRJ0",
        "robot0_FFJ3",
        "robot0_FFJ2",
        "robot0_FFJ1",
        "robot0_MFJ3",
        "robot0_MFJ2",
        "robot0_MFJ1",
        "robot0_RFJ3",
        "robot0_RFJ2",
        "robot0_RFJ1",
        "robot0_LFJ4",
        "robot0_LFJ3",
        "robot0_LFJ2",
        "robot0_LFJ1",
        "robot0_THJ4",
        "robot0_THJ3",
        "robot0_THJ2",
        "robot0_THJ1",
        "robot0_THJ0",
    ]
