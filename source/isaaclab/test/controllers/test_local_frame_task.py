# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test cases for LocalFrameTask class."""

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest
from pink.tasks.frame_task import FrameTask as PinkFrameTask

from isaaclab.controllers.pink_ik.pink_kinematics_configuration import PinkKinematicsConfiguration
from isaaclab.controllers.pink_ik.pink_task_cfg import LocalFrameTaskCfg
from isaaclab.controllers.pink_ik.pink_tasks import LocalFrameTask

pytestmark = pytest.mark.unit

_URDF_PATH = Path(__file__).parent / "urdfs" / "test_urdf_two_link_robot.urdf"


def _se3(translation=(0.0, 0.0, 0.0), rotation_vector=(0.0, 0.0, 0.0)) -> pin.SE3:
    transform = pin.SE3.Identity()
    transform.translation = np.asarray(translation, dtype=float)
    transform.rotation = pin.exp3(np.asarray(rotation_vector, dtype=float))
    return transform


@pytest.fixture
def pink_config():
    return PinkKinematicsConfiguration(urdf_path=str(_URDF_PATH), controlled_joint_names=["joint_1", "joint_2"])


@pytest.fixture
def task():
    return LocalFrameTask(frame="link_2", base_link_frame_name="base_link", position_cost=1.0, orientation_cost=1.0)


@pytest.mark.parametrize(
    "make_task",
    [
        lambda: LocalFrameTask(
            "link_1",
            base_link_frame_name="base_link",
            position_cost=[1.0, 2.0, 3.0],
            orientation_cost=[0.5, 1.0, 1.5],
            lm_damping=0.1,
            gain=2.0,
        ),
        lambda: LocalFrameTask(
            LocalFrameTaskCfg(
                frame="link_1",
                base_link_frame_name="base_link",
                position_cost=[1.0, 2.0, 3.0],
                orientation_cost=[0.5, 1.0, 1.5],
                lm_damping=0.1,
                gain=2.0,
            )
        ),
    ],
    ids=["arguments", "cfg"],
)
def test_initialization(make_task):
    """Both construction paths store the frame, base frame, per-axis costs, and gains; no target is set."""
    task = make_task()
    assert isinstance(task, PinkFrameTask)
    assert task.frame == "link_1"
    assert task.base_link_frame_name == "base_link"
    np.testing.assert_allclose(task.cost, [1.0, 2.0, 3.0, 0.5, 1.0, 1.5])
    assert task.lm_damping == 0.1
    assert task.gain == 2.0
    assert task.transform_target_to_base is None


def test_missing_arguments_are_rejected():
    with pytest.raises(ValueError, match="base_link_frame_name"):
        LocalFrameTask("link_1", position_cost=1.0, orientation_cost=1.0)
    with pytest.raises(ValueError, match="position_cost and orientation_cost"):
        LocalFrameTask("link_1", base_link_frame_name="base_link")


def test_set_target_copies_transform(task):
    target = _se3([0.1, 0.2, 0.3], [0.1, 0.0, 0.0])
    task.set_target(target)
    assert task.transform_target_to_base is not target
    target.translation = np.array([0.5, 0.5, 0.5])
    np.testing.assert_allclose(task.transform_target_to_base.translation, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(task.transform_target_to_base.rotation, pin.exp3(np.array([0.1, 0.0, 0.0])))


def test_set_target_from_configuration_makes_error_zero(task, pink_config):
    """The target read from the configuration is the current relative pose, so the error vanishes."""
    task.set_target_from_configuration(pink_config)
    assert isinstance(task.transform_target_to_base, pin.SE3)
    np.testing.assert_allclose(task.compute_error(pink_config), np.zeros(6), atol=1e-10)


def test_error_and_jacobian_require_target_and_configuration_type(task, pink_config):
    with pytest.raises(ValueError, match="configuration must be a PinkKinematicsConfiguration"):
        task.set_target_from_configuration("not_a_configuration")
    with pytest.raises(ValueError, match="no target set for frame 'link_2'"):
        task.compute_error(pink_config)
    with pytest.raises(Exception, match="no target set for frame 'link_2'"):
        task.compute_jacobian(pink_config)
    task.set_target(_se3())
    with pytest.raises(ValueError, match="configuration must be a PinkKinematicsConfiguration"):
        task.compute_error("not_a_configuration")


@pytest.mark.parametrize(
    ("frame", "base_link_frame_name"), [("link_2", "base_link"), ("link_2", "link_1"), ("base_link", "base_link")]
)
def test_error_and_jacobian_follow_target_and_configuration(pink_config, frame, base_link_frame_name):
    """Errors and Jacobians have the local-frame shape and respond to both the target and the joint state."""
    task = LocalFrameTask(frame, base_link_frame_name, position_cost=1.0, orientation_cost=1.0)
    task.set_target(_se3([0.1, 0.2, 0.3], [0.2, 0.0, 0.0]))

    error = task.compute_error(pink_config)
    jacobian = task.compute_jacobian(pink_config)
    assert error.shape == (6,) and np.all(np.isfinite(error))
    assert jacobian.shape == (6, 2)

    task.set_target(_se3([0.0, 0.1, 0.0]))
    assert not np.allclose(task.compute_error(pink_config), error)

    new_q = pink_config.full_q.copy()
    new_q[1] = 0.5
    pink_config.update(new_q)
    if frame != base_link_frame_name:
        assert not np.allclose(task.compute_error(pink_config), error)
        assert not np.allclose(task.compute_jacobian(pink_config), jacobian)


def test_jacobian_keeps_full_rank_across_configurations(task, pink_config):
    task.set_target(_se3([0.0, 0.0, 0.45], [np.pi / 2, 0.0, 0.0]))
    for angle in np.linspace(0.0, 0.4, 5):
        new_q = pink_config.full_q.copy()
        new_q[1] = angle
        pink_config.update(new_q)
        assert np.linalg.matrix_rank(task.compute_jacobian(pink_config)) == 2
