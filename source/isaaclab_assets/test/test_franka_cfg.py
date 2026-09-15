# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Franka asset configuration contracts."""

from isaaclab_assets import FRANKA_PANDA_MENAGERIE_CFG


def test_franka_menagerie_uses_safe_default_variants() -> None:
    """The generic asset configuration must load PhysX with complete primitive colliders."""
    assert FRANKA_PANDA_MENAGERIE_CFG.spawn.variants == {"Physics": "physx", "Colliders": "primitives"}


def test_franka_menagerie_actuators_define_backend_invariant_properties() -> None:
    """Solver payloads must not change the controller's damping, effort, or mimic-drive contract."""
    arm = FRANKA_PANDA_MENAGERIE_CFG.actuators["panda_arm"]
    hand = FRANKA_PANDA_MENAGERIE_CFG.actuators["panda_hand"]
    follower = FRANKA_PANDA_MENAGERIE_CFG.actuators["panda_finger2_passive"]

    assert arm.joint_effort_limit == {"panda_joint[1-4]": 100.0, "panda_joint[5-7]": 12.0}
    assert arm.viscous_friction == 0.0
    assert hand.joint_names_expr == ["panda_finger_joint1"]
    assert hand.joint_effort_limit == 200.0
    assert hand.stiffness is None
    assert hand.damping is None
    assert hand.viscous_friction == 0.0
    assert follower.joint_names_expr == ["panda_finger_joint2"]
    assert follower.joint_effort_limit == 200.0
    assert follower.stiffness == 0.0
    assert follower.damping == 0.0
    assert follower.viscous_friction == 0.0
