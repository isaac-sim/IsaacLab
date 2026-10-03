# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the RealHand P7 robot configurations."""

import pytest

from isaaclab_assets import P7_L6_CFG, P7_L20_CFG, P7_O6_CFG


@pytest.mark.parametrize(
    ("cfg", "asset_name", "actuator_names"),
    [
        (P7_L6_CFG, "P7_l6_bimanual/P7_l6_bimanual.usda", {"arms", "hands"}),
        (P7_O6_CFG, "P7_o6_bimanual/P7_o6_bimanual.usda", {"arms", "hands", "hand_mimics"}),
        (P7_L20_CFG, "P7_L20_bimanual/P7_L20_bimanual.usda", {"arms", "hands", "hand_mimics"}),
    ],
)
def test_realhand_asset_configuration(cfg, asset_name, actuator_names):
    """Each model points to its public USD and declares the expected actuator groups."""
    assert cfg.spawn.usd_path.startswith("https://huggingface.co/realhandinc/realhand-teleop/resolve/")
    revision = cfg.spawn.usd_path.split("/resolve/", maxsplit=1)[1].split("/", maxsplit=1)[0]
    assert len(revision) == 40
    assert all(character in "0123456789abcdef" for character in revision)
    assert cfg.spawn.usd_path.endswith(asset_name)
    assert cfg.spawn.fix_root_link is None
    assert cfg.spawn.variants == {"Physics": "physx"}
    assert cfg.init_state.pos == (0.0, 0.0, 1.25)
    assert set(cfg.actuators) == actuator_names


def test_realhand_hand_specific_settings():
    """Model-specific mimic, armature, and velocity settings remain isolated."""
    assert P7_L6_CFG.actuators["hands"].armature == 0.0
    assert P7_O6_CFG.actuators["hands"].armature == 0.001
    assert P7_L20_CFG.actuators["hands"].armature == 0.001
    assert P7_L20_CFG.actuators["hands"].joint_velocity_limit == 1.0
    assert P7_L20_CFG.actuators["hand_mimics"].joint_effort_limit == 0.0


def test_realhand_configs_are_independent():
    """Mutating a copied model does not leak into another exported configuration."""
    copied = P7_L6_CFG.copy()
    copied.actuators["arms"].stiffness = 1.0
    assert P7_L6_CFG.actuators["arms"].stiffness == 1100.0
    assert P7_O6_CFG.actuators["arms"].stiffness == 1100.0
