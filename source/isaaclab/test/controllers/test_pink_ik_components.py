# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test cases for PinkKinematicsConfiguration and the Pink IK action helpers."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pinocchio as pin
import pytest
import torch
from pink.configuration import Configuration
from pink.exceptions import FrameNotFound

from isaaclab.controllers.pink_ik.pink_kinematics_configuration import PinkKinematicsConfiguration

pytestmark = pytest.mark.unit

_URDF_PATH = str(Path(__file__).parent / "urdfs" / "test_urdf_two_link_robot.urdf")
_ALL_JOINTS = ["joint_1", "joint_2"]


def _configuration(controlled_joint_names: list[str]) -> PinkKinematicsConfiguration:
    return PinkKinematicsConfiguration(urdf_path=_URDF_PATH, controlled_joint_names=controlled_joint_names)


@pytest.fixture
def pink_config():
    return _configuration(_ALL_JOINTS)


@pytest.mark.parametrize(
    ("controlled_joint_names", "expected_controlled"),
    [(_ALL_JOINTS, _ALL_JOINTS), (["joint_1"], ["joint_1"]), ([], []), (["nonexistent_joint"], [])],
)
def test_controlled_model_follows_controlled_joints(controlled_joint_names, expected_controlled):
    """The reduced model only keeps the requested joints while the full model keeps all of them."""
    config = _configuration(controlled_joint_names)
    assert isinstance(config, Configuration)
    assert config.all_joint_names_pinocchio_order == _ALL_JOINTS
    assert config.controlled_joint_names_pinocchio_order == expected_controlled
    assert config.controlled_model.nq == len(expected_controlled)
    assert config.full_model.nq == len(_ALL_JOINTS)
    np.testing.assert_allclose(config.controlled_q, config.full_q[config._controlled_joint_indices])


def test_invalid_urdf_is_rejected():
    with pytest.raises(Exception):
        PinkKinematicsConfiguration(urdf_path="nonexistent.urdf", controlled_joint_names=[])


def test_update_configuration(pink_config):
    """Updating moves the full configuration, ``None`` keeps it, and wrong lengths are rejected."""
    initial_q = pink_config.full_q.copy()
    transform_initial = pink_config.get_transform_frame_to_world("link_2")
    jacobian_initial = pink_config.get_frame_jacobian("link_2")

    new_q = initial_q.copy()
    new_q[1] = 0.5
    pink_config.update(new_q)
    np.testing.assert_allclose(pink_config.full_q, new_q)
    assert not np.allclose(
        pink_config.get_transform_frame_to_world("link_2").homogeneous, transform_initial.homogeneous
    )
    assert not np.allclose(pink_config.get_frame_jacobian("link_2"), jacobian_initial)

    pink_config.update(None)
    np.testing.assert_allclose(pink_config.full_q, new_q)

    with pytest.raises(ValueError, match="q must have the same length as the number of joints"):
        pink_config.update(np.array([0.1, 0.2, 0.3]))


def test_frame_queries(pink_config):
    jacobian = pink_config.get_frame_jacobian("link_1")
    assert jacobian.shape == (6, len(_ALL_JOINTS))
    assert not np.allclose(jacobian, 0.0)
    transform = pink_config.get_transform_frame_to_world("link_1")
    assert isinstance(transform, pin.SE3)
    assert not np.allclose(transform.homogeneous, np.eye(4))
    with pytest.raises(FrameNotFound):
        pink_config.get_frame_jacobian("nonexistent_frame")
    with pytest.raises(FrameNotFound):
        pink_config.get_transform_frame_to_world("nonexistent_frame")


@pytest.mark.parametrize("fixed_base", [False, True])
@pytest.mark.parametrize("disable_gravity", [False, True])
@pytest.mark.parametrize("robot_name", ["GR1T2_HIGH_PD_CFG", "G1_INSPIRE_FTP_CFG"])
def test_action_gravity_compensation_with_migrated_robot_configs(fixed_base, disable_gravity, robot_name):
    """The Pink robot configs retain direct gravity access and the action's effort targets."""
    from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction

    import isaaclab_assets

    robot_cfg = getattr(isaaclab_assets, robot_name).copy()
    robot_cfg.spawn.rigid_props.disable_gravity = disable_gravity
    num_base_dofs = 0 if fixed_base else 6
    forces = torch.arange(2 * (3 + num_base_dofs), dtype=torch.float32).reshape(2, -1)
    asset = SimpleNamespace(
        cfg=robot_cfg,
        data=SimpleNamespace(gravity_compensation_forces=SimpleNamespace(torch=forces)),
        num_base_dofs=num_base_dofs,
        is_fixed_base=fixed_base,
        set_joint_effort_target_index=Mock(),
    )
    action = SimpleNamespace(
        _asset=asset, _controlled_joint_ids=[0, 2], _controlled_joint_ids_tensor=torch.tensor([0, 2])
    )
    PinkInverseKinematicsAction._apply_gravity_compensation(action)

    if disable_gravity:
        asset.set_joint_effort_target_index.assert_not_called()
    else:
        asset.set_joint_effort_target_index.assert_called_once()
        kwargs = asset.set_joint_effort_target_index.call_args.kwargs
        assert kwargs["joint_ids"] == [0, 2]
        expected = torch.zeros(2, 2) if fixed_base else forces[:, [6, 8]]
        torch.testing.assert_close(kwargs["target"], expected)
