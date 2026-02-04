# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for mock asset interfaces."""

import pytest
import torch

from isaaclab.test.mock_interfaces.assets import (
    MockArticulation,
    MockRigidObject,
    MockRigidObjectCollection,
    create_mock_articulation,
    create_mock_humanoid,
    create_mock_quadruped,
    create_mock_rigid_object,
    create_mock_rigid_object_collection,
)
from isaaclab.test.mock_interfaces.utils import MockArticulationBuilder

# ==============================================================================
# MockArticulation Tests
# ==============================================================================


class TestMockArticulation:
    """Tests for MockArticulation and MockArticulationData."""

    @pytest.fixture
    def robot(self):
        """Create a mock articulation fixture."""
        return MockArticulation(
            num_instances=4,
            num_joints=12,
            num_bodies=13,
            device="cpu",
        )

    def test_initialization(self, robot):
        """Test that MockArticulation initializes correctly."""
        assert robot.num_instances == 4
        assert robot.num_joints == 12
        assert robot.num_bodies == 13
        assert robot.device == "cpu"
        assert robot.root_view is None
        assert robot.data is not None

    def test_joint_state_shapes(self, robot):
        """Test joint state tensor shapes."""
        assert robot.data.joint_pos.shape == (4, 12)
        assert robot.data.joint_vel.shape == (4, 12)
        assert robot.data.joint_acc.shape == (4, 12)

    def test_root_state_shapes(self, robot):
        """Test root state tensor shapes."""
        # Link frame
        assert robot.data.root_link_pose_w.shape == (4, 7)
        assert robot.data.root_link_vel_w.shape == (4, 6)
        assert robot.data.root_link_state_w.shape == (4, 13)

        # Sliced properties
        assert robot.data.root_link_pos_w.shape == (4, 3)
        assert robot.data.root_link_quat_w.shape == (4, 4)
        assert robot.data.root_link_lin_vel_w.shape == (4, 3)
        assert robot.data.root_link_ang_vel_w.shape == (4, 3)

    def test_body_state_shapes(self, robot):
        """Test body state tensor shapes."""
        assert robot.data.body_link_pose_w.shape == (4, 13, 7)
        assert robot.data.body_link_vel_w.shape == (4, 13, 6)
        assert robot.data.body_link_state_w.shape == (4, 13, 13)

    def test_default_state_shapes(self, robot):
        """Test default state tensor shapes."""
        assert robot.data.default_root_pose.shape == (4, 7)
        assert robot.data.default_root_vel.shape == (4, 6)
        assert robot.data.default_root_state.shape == (4, 13)
        assert robot.data.default_joint_pos.shape == (4, 12)
        assert robot.data.default_joint_vel.shape == (4, 12)

    def test_identity_quaternion_default(self, robot):
        """Test that default quaternions are identity quaternions."""
        quat = robot.data.root_link_quat_w
        # w=1, x=y=z=0
        assert torch.all(quat[:, 0] == 1)
        assert torch.all(quat[:, 1:] == 0)

    def test_set_joint_pos(self, robot):
        """Test setting joint positions."""
        joint_pos = torch.randn(4, 12)
        robot.data.set_joint_pos(joint_pos)
        assert torch.allclose(robot.data.joint_pos, joint_pos)

    def test_set_mock_data_bulk(self, robot):
        """Test bulk data setter."""
        joint_pos = torch.randn(4, 12)
        joint_vel = torch.randn(4, 12)

        robot.data.set_mock_data(joint_pos=joint_pos, joint_vel=joint_vel)

        assert torch.allclose(robot.data.joint_pos, joint_pos)
        assert torch.allclose(robot.data.joint_vel, joint_vel)

    def test_find_joints(self):
        """Test joint finding by regex."""
        joint_names = ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf"]
        robot = MockArticulation(
            num_instances=1,
            num_joints=6,
            num_bodies=7,
            joint_names=joint_names,
            device="cpu",
        )

        # Find all hip joints
        indices, names = robot.find_joints(".*_hip")
        assert len(indices) == 2
        assert "FL_hip" in names
        assert "FR_hip" in names

        # Find FL leg joints
        indices, names = robot.find_joints("FL_.*")
        assert len(indices) == 3

    def test_find_bodies(self):
        """Test body finding by regex."""
        body_names = ["base", "FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf"]
        robot = MockArticulation(
            num_instances=1,
            num_joints=6,
            num_bodies=7,
            body_names=body_names,
            device="cpu",
        )

        # Find base
        indices, names = robot.find_bodies("base")
        assert indices == [0]

        # Find all thigh bodies
        indices, names = robot.find_bodies(".*_thigh")
        assert len(indices) == 2

    def test_set_joint_position_target(self, robot):
        """Test setting joint position targets."""
        target = torch.randn(4, 12)
        robot.set_joint_position_target(target)
        assert torch.allclose(robot.data.joint_pos_target, target)

    def test_joint_limits(self, robot):
        """Test joint limits."""
        limits = robot.data.joint_pos_limits
        assert limits.shape == (4, 12, 2)
        # Default limits should be -inf to inf
        assert torch.all(limits[..., 0] == float("-inf"))
        assert torch.all(limits[..., 1] == float("inf"))


# ==============================================================================
# MockRigidObject Tests
# ==============================================================================


class TestMockRigidObject:
    """Tests for MockRigidObject and MockRigidObjectData."""

    @pytest.fixture
    def obj(self):
        """Create a mock rigid object fixture."""
        return MockRigidObject(num_instances=4, device="cpu")

    def test_initialization(self, obj):
        """Test that MockRigidObject initializes correctly."""
        assert obj.num_instances == 4
        assert obj.num_bodies == 1  # Always 1 for rigid object
        assert obj.root_view is None

    def test_root_state_shapes(self, obj):
        """Test root state tensor shapes."""
        assert obj.data.root_link_pose_w.shape == (4, 7)
        assert obj.data.root_link_vel_w.shape == (4, 6)
        assert obj.data.root_link_state_w.shape == (4, 13)

    def test_body_state_shapes(self, obj):
        """Test body state tensor shapes (single body)."""
        assert obj.data.body_link_pose_w.shape == (4, 1, 7)
        assert obj.data.body_link_vel_w.shape == (4, 1, 6)

    def test_body_properties(self, obj):
        """Test body property shapes."""
        assert obj.data.body_mass.shape == (4, 1, 1)
        assert obj.data.body_inertia.shape == (4, 1, 9)


# ==============================================================================
# MockRigidObjectCollection Tests
# ==============================================================================


class TestMockRigidObjectCollection:
    """Tests for MockRigidObjectCollection and MockRigidObjectCollectionData."""

    @pytest.fixture
    def collection(self):
        """Create a mock rigid object collection fixture."""
        return MockRigidObjectCollection(
            num_instances=4,
            num_bodies=5,
            device="cpu",
        )

    def test_initialization(self, collection):
        """Test that MockRigidObjectCollection initializes correctly."""
        assert collection.num_instances == 4
        assert collection.num_bodies == 5

    def test_body_state_shapes(self, collection):
        """Test body state tensor shapes."""
        assert collection.data.body_link_pose_w.shape == (4, 5, 7)
        assert collection.data.body_link_vel_w.shape == (4, 5, 6)
        assert collection.data.body_link_state_w.shape == (4, 5, 13)

    def test_find_bodies_returns_mask(self, collection):
        """Test that find_bodies returns a mask tensor."""
        body_mask, names, indices = collection.find_bodies("body_0")
        assert isinstance(body_mask, torch.Tensor)
        assert body_mask.shape == (5,)
        assert body_mask[0]


# ==============================================================================
# Factory Function Tests
# ==============================================================================


class TestAssetFactories:
    """Tests for asset factory functions."""

    def test_create_mock_quadruped(self):
        """Test quadruped factory function."""
        robot = create_mock_quadruped(num_instances=4)
        assert robot.num_instances == 4
        assert robot.num_joints == 12
        assert robot.num_bodies == 13
        assert "FL_hip" in robot.joint_names
        assert "base" in robot.body_names

    def test_create_mock_humanoid(self):
        """Test humanoid factory function."""
        robot = create_mock_humanoid(num_instances=2)
        assert robot.num_instances == 2
        assert robot.num_joints == 21

    def test_create_mock_articulation(self):
        """Test generic articulation factory function."""
        robot = create_mock_articulation(
            num_instances=2,
            num_joints=6,
            num_bodies=7,
            is_fixed_base=True,
        )
        assert robot.num_instances == 2
        assert robot.num_joints == 6
        assert robot.is_fixed_base

    def test_create_mock_rigid_object(self):
        """Test rigid object factory function."""
        obj = create_mock_rigid_object(num_instances=3)
        assert obj.num_instances == 3
        assert obj.num_bodies == 1
        assert obj.data.root_link_pose_w.shape == (3, 7)

    def test_create_mock_rigid_object_collection(self):
        """Test rigid object collection factory function."""
        collection = create_mock_rigid_object_collection(
            num_instances=4,
            num_bodies=6,
            body_names=["obj_0", "obj_1", "obj_2", "obj_3", "obj_4", "obj_5"],
        )
        assert collection.num_instances == 4
        assert collection.num_bodies == 6
        assert collection.body_names == ["obj_0", "obj_1", "obj_2", "obj_3", "obj_4", "obj_5"]
        assert collection.data.body_link_pose_w.shape == (4, 6, 7)


# ==============================================================================
# MockArticulationBuilder Tests
# ==============================================================================


class TestMockArticulationBuilder:
    """Tests for MockArticulationBuilder."""

    def test_basic_build(self):
        """Test building a basic articulation."""
        robot = (
            MockArticulationBuilder()
            .with_num_instances(4)
            .with_joints(["joint_0", "joint_1", "joint_2"])
            .with_bodies(["base", "link_1", "link_2", "link_3"])
            .build()
        )

        assert robot.num_instances == 4
        assert robot.num_joints == 3
        assert robot.num_bodies == 4

    def test_with_default_positions(self):
        """Test setting default joint positions."""
        default_pos = [0.0, 0.5, -0.5]
        robot = (
            MockArticulationBuilder()
            .with_num_instances(2)
            .with_joints(["j0", "j1", "j2"], default_pos=default_pos)
            .build()
        )

        expected = torch.tensor([default_pos, default_pos])
        assert torch.allclose(robot.data.joint_pos, expected)

    def test_with_joint_limits(self):
        """Test setting joint limits."""
        robot = (
            MockArticulationBuilder()
            .with_num_instances(1)
            .with_joints(["j0", "j1"])
            .with_joint_limits(-1.0, 1.0)
            .build()
        )

        limits = robot.data.joint_pos_limits
        assert torch.all(limits[..., 0] == -1.0)
        assert torch.all(limits[..., 1] == 1.0)
