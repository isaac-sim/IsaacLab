# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the assembled and standalone RealHand asset configurations."""

import re

import pytest

from isaaclab_assets import (
    L6_LEFT_CFG,
    L6_RIGHT_CFG,
    L20_LEFT_CFG,
    L20_RIGHT_CFG,
    O6_LEFT_CFG,
    O6_RIGHT_CFG,
    P7_CFG,
    P7_L6_CFG,
    P7_L20_CFG,
    P7_O6_CFG,
)

pytestmark = pytest.mark.kitless


@pytest.mark.parametrize(
    ("cfg", "asset_name", "actuator_names"),
    [
        (P7_L6_CFG, "P7_l6_bimanual/P7_l6_bimanual.usda", {"arms", "hands"}),
        (P7_O6_CFG, "P7_o6_bimanual/P7_o6_bimanual.usda", {"arms", "hands", "hand_mimics"}),
        (P7_L20_CFG, "P7_L20_bimanual/P7_L20_bimanual.usda", {"arms", "hands", "hand_mimics"}),
    ],
)
def test_realhand_asset_configuration(cfg, asset_name, actuator_names):
    """Each model pins a public USD and starts with zero-angle arms and open hands."""
    assert cfg.spawn.usd_path.startswith("https://huggingface.co/realhandinc/realhand-teleop/resolve/")
    revision = cfg.spawn.usd_path.split("/resolve/", maxsplit=1)[1].split("/", maxsplit=1)[0]
    assert len(revision) == 40
    assert all(character in "0123456789abcdef" for character in revision)
    assert cfg.spawn.usd_path.endswith(asset_name)
    assert cfg.spawn.fix_root_link is None
    assert cfg.spawn.variants == {"Physics": "physx"}
    assert cfg.init_state.pos == (0.0, 0.0, 1.25)
    assert set(cfg.actuators) == actuator_names
    for side in ("left", "right"):
        for joint in range(1, 8):
            assert cfg.init_state.joint_pos[f"{side}_joint{joint}"] == 0.0
    assert cfg.init_state.joint_pos[".*_hand_.*"] == 0.0


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


def test_realhand_mimic_configs_are_independent():
    """An in-place edit of one preset's mimic patterns cannot alter the other."""
    o6_patterns = P7_O6_CFG.actuators["hand_mimics"].joint_names_expr
    l20_patterns = P7_L20_CFG.actuators["hand_mimics"].joint_names_expr
    original = list(l20_patterns)
    try:
        o6_patterns.append("test_only_mimic_joint")
        assert l20_patterns == original
    finally:
        o6_patterns.pop()


def test_standalone_p7_configuration():
    """The bare arm preset has no hand actuators or hand initial-state expressions."""
    assert re.fullmatch(
        r"https://huggingface.co/realhandinc/realhand-teleop/resolve/[0-9a-f]{40}/assets/isaaclab/robots/P7/P7.usda",
        P7_CFG.spawn.usd_path,
    )
    assert set(P7_CFG.actuators) == {"arms"}
    assert P7_CFG.init_state.pos == (0.0, 0.0, 1.25)
    assert len(P7_CFG.init_state.joint_pos) == 14
    assert all("_hand_" not in name and value == 0.0 for name, value in P7_CFG.init_state.joint_pos.items())


@pytest.mark.parametrize(
    ("cfg", "model", "side", "has_mimics"),
    [
        (L6_LEFT_CFG, "L6", "left", False),
        (L6_RIGHT_CFG, "L6", "right", False),
        (O6_LEFT_CFG, "O6", "left", True),
        (O6_RIGHT_CFG, "O6", "right", True),
        (L20_LEFT_CFG, "L20", "left", True),
        (L20_RIGHT_CFG, "L20", "right", True),
    ],
)
def test_standalone_realhand_configuration(cfg, model, side, has_mimics):
    """Each public hand preset addresses only its own side, in the native palm frame."""
    assert re.fullmatch(
        r"https://huggingface.co/realhandinc/realhand-teleop/resolve/[0-9a-f]{40}/assets/isaaclab/robots/.*",
        cfg.spawn.usd_path,
    )
    assert cfg.spawn.usd_path.endswith(f"/{model}_{side}/{model}_{side}.usda")
    assert set(cfg.actuators) == ({"hands", "hand_mimics"} if has_mimics else {"hands"})
    assert cfg.init_state.pos == (0.0, 0.0, 0.5)
    assert cfg.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert cfg.init_state.joint_pos == {f"{side}_hand_.*": 0.0}
    for joint in ("thumb_cmc_pitch", "index_mcp_pitch"):
        assert any(
            re.fullmatch(expression, f"{side}_hand_{joint}") for expression in cfg.actuators["hands"].joint_names_expr
        )
    other_side = "right" if side == "left" else "left"
    for actuator in cfg.actuators.values():
        for expression in actuator.joint_names_expr:
            assert not re.fullmatch(expression, f"{other_side}_hand_thumb_cmc_pitch")
            assert not re.fullmatch(expression, f"{side}_joint1")
    if has_mimics:
        assert cfg.actuators["hand_mimics"].joint_effort_limit == 0.0
        assert cfg.actuators["hand_mimics"].stiffness == 0.0
        assert cfg.actuators["hand_mimics"].damping == 0.0


def test_standalone_realhand_configs_are_independent():
    """Editing a standalone preset cannot change its sibling or assembled preset."""
    original = list(L20_LEFT_CFG.actuators["hand_mimics"].joint_names_expr)
    try:
        L20_LEFT_CFG.actuators["hand_mimics"].joint_names_expr.append("test_only_mimic_joint")
        assert "test_only_mimic_joint" not in L20_RIGHT_CFG.actuators["hand_mimics"].joint_names_expr
        assert "test_only_mimic_joint" not in P7_L20_CFG.actuators["hand_mimics"].joint_names_expr
    finally:
        L20_LEFT_CFG.actuators["hand_mimics"].joint_names_expr[:] = original
