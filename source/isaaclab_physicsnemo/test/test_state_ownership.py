# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scenes whose state and actuation a learned joint-state solver can or cannot represent."""

import pytest

pytest.importorskip("physicsnemo_robotics")

import newton
import warp as wp
from isaaclab_physicsnemo.physics import check_velocity_targets, validate_effort_control, validate_state_ownership
from physicsnemo_robotics.dynamics import newton_state_codec


def _two_world_robot(*, root_anchor: wp.transform | None = None, hinge_enabled: bool = True) -> newton.Model:
    """Build two worlds of a free-floating base with one hinged link."""
    inertia = wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)
    scene = newton.ModelBuilder()
    for _ in range(2):
        robot = newton.ModelBuilder()
        base = robot.add_link(mass=1.0, inertia=inertia)
        root = robot.add_joint_free(base, child_xform=root_anchor)
        link = robot.add_link(mass=1.0, inertia=inertia)
        hinge = robot.add_joint_revolute(base, link, enabled=hinge_enabled)
        robot.add_articulation([root, hinge])
        scene.begin_world()
        scene.add_builder(robot)
        scene.end_world()
    return scene.finalize(device="cpu")


def test_body_state_cannot_drive_joint_backed_assets():
    """Isaac Lab reads articulations through joint state, which a body-state model would leave stale."""
    model = _two_world_robot()
    validate_state_ownership(model, newton_state_codec(model, "joint"))
    with pytest.raises(ValueError, match="joint-backed"):
        validate_state_ownership(model, newton_state_codec(model, "body"))


@pytest.mark.parametrize(
    ("model_kwargs", "reason"),
    [
        ({"root_anchor": wp.transform((0.0, 0.0, 0.1), wp.quat_identity())}, "identity anchors"),
        ({"hinge_enabled": False}, "Disabled joint"),
    ],
    ids=["offset-free-root", "disabled-joint"],
)
def test_forward_kinematics_must_rebuild_body_state(model_kwargs, reason):
    """Body state follows learned joint state only for trees that forward kinematics reproduces."""
    model = _two_world_robot(**model_kwargs)
    with pytest.raises(ValueError, match=reason):
        validate_state_ownership(model, newton_state_codec(model, "joint"))


def test_implicit_position_drives_are_not_effort_control():
    """Damping is a function of the state, but position-drive targets never reach ``joint_f``."""
    model = _two_world_robot()
    wp.to_torch(model.joint_target_kd).fill_(5.0)
    validate_effort_control(model)

    wp.to_torch(model.joint_target_ke)[6] = 40.0
    with pytest.raises(ValueError, match=r"Joint DOFs \[6\] use implicit position drives"):
        validate_effort_control(model)


def test_velocity_targets_of_damped_joints_are_rejected():
    """A velocity target on a damped joint drives it without appearing in the recorded or modeled control."""
    model = _two_world_robot()
    wp.to_torch(model.joint_target_kd)[6] = 5.0
    control = model.control()
    check_velocity_targets(model, control)

    wp.to_torch(control.joint_target_qd)[0] = 1.0  # an undamped joint
    check_velocity_targets(model, control)
    wp.to_torch(control.joint_target_qd)[6] = 1.0
    with pytest.raises(RuntimeError, match="velocity targets"):
        check_velocity_targets(model, control)
