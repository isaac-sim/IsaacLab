# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for mock asset interfaces."""

import warnings
from types import SimpleNamespace
from typing import get_args, get_type_hints

import pytest
import torch
import warp as wp

from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg, RemotizedPDActuator, RemotizedPDActuatorCfg
from isaaclab.assets.articulation import BaseArticulation
from isaaclab.envs.mdp.actions import BinaryJointPositionAction, BinaryJointPositionActionCfg
from isaaclab.managers import SceneEntityCfg
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
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.unit


def _make_finder_case(case: str):
    """Create an asset and finder invocation for a finder return-mode test case."""
    if case.startswith("articulation_"):
        asset = MockArticulation(
            num_instances=1,
            num_joints=3,
            num_bodies=3,
            joint_names=["item_0", "item_1", "other"],
            body_names=["item_0", "item_1", "other"],
            num_fixed_tendons=3,
            num_spatial_tendons=3,
            fixed_tendon_names=["item_0", "item_1", "other"],
            spatial_tendon_names=["item_0", "item_1", "other"],
            device="cpu",
        )
        finder_name = case.removeprefix("articulation_")
    elif case == "rigid_object_bodies":
        asset = MockRigidObject(num_instances=1, body_names=["item_0"], device="cpu")
        finder_name = "bodies"
    elif case == "collection_bodies":
        asset = MockRigidObjectCollection(
            num_instances=1,
            num_bodies=3,
            body_names=["item_0", "item_1", "other"],
            device="cpu",
        )
        finder_name = "bodies"
    else:
        raise ValueError(f"Unknown finder case: {case}")

    return asset, getattr(asset, f"find_{finder_name}"), "item_.*"


_LIST_FINDER_CASES = [
    "articulation_bodies",
    "articulation_joints",
    "articulation_fixed_tendons",
    "articulation_spatial_tendons",
    "rigid_object_bodies",
]

_ALL_FINDER_CASES = _LIST_FINDER_CASES + ["collection_bodies"]


@pytest.mark.parametrize("case", _ALL_FINDER_CASES)
def test_finders_support_transitional_return_modes(case):
    """Test implicit legacy, explicit legacy, and proxy finder return modes."""
    asset, finder, name_keys = _make_finder_case(case)

    with pytest.warns(DeprecationWarning, match="as_proxy"):
        implicit_indices, implicit_names = finder(name_keys)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        explicit_indices, explicit_names = finder(name_keys, as_proxy=False)
        proxy_indices, proxy_names = finder(name_keys, as_proxy=True)
        empty_indices, empty_names = finder([], as_proxy=True)
        repeated_empty_indices, repeated_empty_names = finder([], as_proxy=True)

    expected_indices = [0] if case == "rigid_object_bodies" else [0, 1]
    expected_names = ["item_0"] if case == "rigid_object_bodies" else ["item_0", "item_1"]
    if case == "collection_bodies":
        assert isinstance(implicit_indices, torch.Tensor)
        assert isinstance(explicit_indices, torch.Tensor)
        assert implicit_indices.dtype == torch.int32
        assert explicit_indices.dtype == torch.int32
        assert implicit_indices.device.type == asset.device
        assert explicit_indices.device.type == asset.device
        assert implicit_indices.tolist() == expected_indices
        assert explicit_indices.tolist() == expected_indices
    else:
        assert implicit_indices == expected_indices
        assert explicit_indices == expected_indices

    assert implicit_names == expected_names
    assert explicit_names == implicit_names
    assert proxy_names == implicit_names
    assert isinstance(proxy_indices, ProxyArray)
    assert proxy_indices.warp.dtype == wp.int32
    assert str(proxy_indices.warp.device) == asset.device
    assert proxy_indices.torch.dtype == torch.int32
    assert proxy_indices.torch.device.type == asset.device
    assert proxy_indices.torch.tolist() == expected_indices
    assert proxy_indices.warp.ptr == proxy_indices.torch.data_ptr()
    assert empty_names == repeated_empty_names == []
    assert empty_indices.torch.tolist() == []
    assert empty_indices is repeated_empty_indices
    assert empty_indices.warp is repeated_empty_indices.warp
    assert empty_indices.torch is repeated_empty_indices.torch


@pytest.mark.parametrize("case", _ALL_FINDER_CASES)
def test_finders_reuse_proxy_but_return_fresh_names(case):
    """Test equivalent finder results share only cached selector storage."""
    _, finder, _ = _make_finder_case(case)

    first_indices, first_names = finder(["item_0", "item_1"], preserve_order=True, as_proxy=True)
    second_indices, second_names = finder("item_.*", as_proxy=True)

    expected_names = ["item_0"] if case == "rigid_object_bodies" else ["item_0", "item_1"]
    assert first_indices is second_indices
    assert first_indices.warp.ptr == second_indices.warp.ptr
    assert first_names == second_names == expected_names
    assert first_names is not second_names
    first_names.append("mutated")
    assert second_names == expected_names


@pytest.mark.parametrize(
    ("finder_name", "subset_arg"),
    [
        ("find_joints", "joint_subset"),
        ("find_fixed_tendons", "tendon_subsets"),
        ("find_spatial_tendons", "tendon_subsets"),
    ],
)
def test_articulation_finders_cache_after_subset_remapping(finder_name, subset_arg):
    """Test subset-local matches reuse proxies by final articulation indices."""
    robot = MockArticulation(
        num_instances=1,
        num_joints=3,
        num_bodies=3,
        joint_names=["item_0", "item_1", "item_2"],
        num_fixed_tendons=3,
        num_spatial_tendons=3,
        fixed_tendon_names=["item_0", "item_1", "item_2"],
        spatial_tendon_names=["item_0", "item_1", "item_2"],
        device="cpu",
    )
    finder = getattr(robot, finder_name)

    with pytest.warns(DeprecationWarning):
        implicit_indices, implicit_names = finder(".*", **{subset_arg: ["item_2", "item_0"]}, preserve_order=True)
    explicit_indices, explicit_names = finder(
        ".*", **{subset_arg: ["item_2", "item_0"]}, preserve_order=True, as_proxy=False
    )
    subset_indices, subset_names = finder(
        ".*", **{subset_arg: ["item_2", "item_0"]}, preserve_order=True, as_proxy=True
    )
    direct_indices, direct_names = finder(["item_2", "item_0"], preserve_order=True, as_proxy=True)

    assert implicit_indices == explicit_indices == [2, 0]
    assert implicit_names == explicit_names == ["item_2", "item_0"]
    assert subset_indices is direct_indices
    assert subset_indices.torch.tolist() == [2, 0]
    assert subset_names == direct_names == ["item_2", "item_0"]


def test_articulation_finder_domains_do_not_alias():
    """Test equal selector values in distinct articulation domains do not alias."""
    robot, _, _ = _make_finder_case("articulation_bodies")

    selectors = [
        robot.find_bodies("item_.*", as_proxy=True)[0],
        robot.find_joints("item_.*", as_proxy=True)[0],
        robot.find_fixed_tendons("item_.*", as_proxy=True)[0],
        robot.find_spatial_tendons("item_.*", as_proxy=True)[0],
    ]

    assert len({id(selector) for selector in selectors}) == len(selectors)
    assert len({selector.warp.ptr for selector in selectors}) == len(selectors)


def test_collection_find_objects_forwards_return_mode():
    """Test deprecated collection object finder forwards all return modes."""
    collection, _, name_keys = _make_finder_case("collection_bodies")

    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always", DeprecationWarning)
        implicit_indices, implicit_names = collection.find_objects(name_keys)
    assert len(warning_records) == 2
    assert all(issubclass(record.category, DeprecationWarning) for record in warning_records)
    assert any("find_objects" in str(record.message) for record in warning_records)
    assert any("as_proxy" in str(record.message) for record in warning_records)

    for as_proxy in (False, True):
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always", DeprecationWarning)
            indices, names = collection.find_objects(name_keys, as_proxy=as_proxy)
        assert len(warning_records) == 1
        assert issubclass(warning_records[0].category, DeprecationWarning)
        assert "find_objects" in str(warning_records[0].message)
        assert "as_proxy" not in str(warning_records[0].message)
        assert names == implicit_names
        if as_proxy:
            assert isinstance(indices, ProxyArray)
            assert indices is collection.find_bodies(name_keys, as_proxy=True)[0]
        else:
            assert isinstance(indices, torch.Tensor)
            assert indices.dtype == torch.int32
            assert torch.equal(indices, implicit_indices)


def test_binary_joint_action_reuses_cached_proxy_without_torch_materialization():
    """Test repeated action writes retain the cached finder proxy through the writer boundary."""
    robot = MockArticulation(
        num_instances=2,
        num_joints=3,
        num_bodies=1,
        joint_names=["joint_0", "joint_1", "joint_2"],
        device="cpu",
    )
    expected_selector = robot.find_joints(["joint_0", "joint_2"], preserve_order=True, as_proxy=True)[0]
    received_selectors = []

    def record_target(*, target, joint_ids=None, env_ids=None):
        received_selectors.append(joint_ids)

    robot.set_joint_position_target_index = record_target
    env = SimpleNamespace(scene={"robot": robot}, num_envs=2, device="cpu")
    cfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_0", "joint_2"],
        open_command_expr={"joint_.*": 1.0},
        close_command_expr={"joint_.*": -1.0},
    )

    action = BinaryJointPositionAction(cfg, env)
    action.process_actions(torch.ones(2, 1))
    action.apply_actions()
    action.apply_actions()

    assert action._joint_ids is expected_selector
    assert len(received_selectors) == 2
    assert all(selector is expected_selector for selector in received_selectors)
    assert expected_selector._torch_cache is None


def test_actuator_selector_keeps_partial_proxy_and_optimizes_full_order():
    """Test the shared actuator branch keeps partial proxies and full ordered selections as a slice."""
    assert getattr(BaseArticulation._process_actuators_cfg, "__isabstractmethod__", False)
    assert not getattr(BaseArticulation._select_actuator_joint_ids, "__isabstractmethod__", False)
    robot = MockArticulation(
        num_instances=1,
        num_joints=3,
        num_bodies=1,
        joint_names=["joint_0", "joint_1", "joint_2"],
        device="cpu",
    )
    partial_ids, partial_names = robot.find_joints(["joint_2", "joint_0"], preserve_order=True, as_proxy=True)
    full_ids, full_names = robot.find_joints(".*", as_proxy=True)

    reordered_ids, reordered_names = robot.find_joints(
        list(reversed(robot.joint_names)), preserve_order=True, as_proxy=True
    )
    resolved_partial = BaseArticulation._select_actuator_joint_ids(robot, partial_ids, partial_names)
    resolved_full = BaseArticulation._select_actuator_joint_ids(robot, full_ids, full_names)

    resolved_reordered = BaseArticulation._select_actuator_joint_ids(robot, reordered_ids, reordered_names)
    assert resolved_partial is partial_ids
    assert resolved_full == slice(None)
    assert partial_ids._torch_cache is None
    assert resolved_reordered is reordered_ids
    assert full_ids._torch_cache is None

    assert reordered_ids._torch_cache is None


@pytest.mark.parametrize("constructor", [ActuatorBase.__init__, RemotizedPDActuator.__init__])
def test_actuator_constructor_accepts_exact_proxy_selector_annotation(constructor):
    """Test actuator signatures advertise exactly the supported selector representations."""

    type_globals = dict(constructor.__globals__)
    type_globals.update(
        ActuatorBaseCfg=ActuatorBaseCfg,
        RemotizedPDActuatorCfg=RemotizedPDActuatorCfg,
    )
    joint_ids_type = get_type_hints(constructor, globalns=type_globals)["joint_ids"]
    joint_ids_args = set(get_args(joint_ids_type))

    assert joint_ids_args == {slice, torch.Tensor, ProxyArray}
    assert object not in joint_ids_args


def test_scene_entity_cfg_requests_legacy_list_for_joint_bookkeeping():
    """Test SceneEntityCfg keeps list equality and slice-optimization semantics explicit."""
    robot = MockArticulation(
        num_instances=1,
        num_joints=3,
        num_bodies=1,
        joint_names=["joint_0", "joint_1", "joint_2"],
        device="cpu",
    )
    original_find_joints = robot.find_joints
    requested_modes = []

    def record_find_joints(*args, **kwargs):
        requested_modes.append(kwargs.get("as_proxy"))
        return original_find_joints(*args, **kwargs)

    robot.find_joints = record_find_joints
    entity_cfg = SceneEntityCfg("robot", joint_names=["joint_2", "joint_0"], preserve_order=True)

    entity_cfg.resolve({"robot": robot})

    assert requested_modes == [False]
    assert entity_cfg.joint_ids == [2, 0]
    assert entity_cfg.joint_names == ["joint_2", "joint_0"]


def test_scene_entity_cfg_does_not_pass_asset_mode_to_sensor_finder():
    """Test the sensor-specific SceneEntityCfg finder branch remains out of the asset migration."""

    class FakeSensor:
        body_names = ["foot_0", "foot_1"]
        num_sensors = 2

        def find_sensors(self, name_keys, preserve_order=False):
            assert name_keys == ["foot_1"]
            return [1], ["foot_1"]

    entity_cfg = SceneEntityCfg("feet", body_names=["foot_1"])

    entity_cfg.resolve({"feet": FakeSensor()})

    assert entity_cfg.body_ids == [1]
    assert entity_cfg.body_names == ["foot_1"]


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
        # Link frame - pose is wp.transformf so shape is (4,) but converts to (4, 7)
        assert robot.data.root_link_pose_w.torch.shape == (4, 7)
        # vel is wp.spatial_vectorf so shape is (4,) but converts to (4, 6)
        assert robot.data.root_link_vel_w.torch.shape == (4, 6)
        assert robot.data.root_link_state_w.torch.shape == (4, 13)

        # Sliced properties
        assert robot.data.root_link_pos_w.torch.shape == (4, 3)
        assert robot.data.root_link_quat_w.torch.shape == (4, 4)
        assert robot.data.root_link_lin_vel_w.torch.shape == (4, 3)
        assert robot.data.root_link_ang_vel_w.torch.shape == (4, 3)

    def test_body_state_shapes(self, robot):
        """Test body state tensor shapes."""
        # body_link_pose_w is wp.transformf so shape is (4, 13) but converts to (4, 13, 7)
        assert robot.data.body_link_pose_w.torch.shape == (4, 13, 7)
        # body_link_vel_w is wp.spatial_vectorf so shape is (4, 13) but converts to (4, 13, 6)
        assert robot.data.body_link_vel_w.torch.shape == (4, 13, 6)
        assert robot.data.body_link_state_w.torch.shape == (4, 13, 13)

    def test_default_state_shapes(self, robot):
        """Test default state tensor shapes."""
        # default_root_pose is wp.transformf so shape is (4,) but converts to (4, 7)
        assert robot.data.default_root_pose.torch.shape == (4, 7)
        # default_root_vel is wp.spatial_vectorf so shape is (4,) but converts to (4, 6)
        assert robot.data.default_root_vel.torch.shape == (4, 6)
        assert robot.data.default_root_state.torch.shape == (4, 13)
        assert robot.data.default_joint_pos.shape == (4, 12)
        assert robot.data.default_joint_vel.shape == (4, 12)

    def test_identity_quaternion_default(self, robot):
        """Test that default quaternions are identity quaternions."""
        quat = robot.data.root_link_quat_w.torch
        # XYZW format: x=y=z=0, w=1
        expected = torch.zeros_like(quat)
        expected[:, 3] = 1.0  # Set w=1
        assert torch.allclose(quat, expected, atol=1e-5)

    def test_set_joint_pos(self, robot):
        """Test setting joint positions."""
        joint_pos = torch.randn(4, 12)
        robot.data.set_joint_pos(joint_pos)
        assert torch.allclose(robot.data.joint_pos.torch, joint_pos)

    def test_set_mock_data_bulk(self, robot):
        """Test bulk data setter."""
        joint_pos = torch.randn(4, 12)
        joint_vel = torch.randn(4, 12)

        robot.data.set_mock_data(joint_pos=joint_pos, joint_vel=joint_vel)

        assert torch.allclose(robot.data.joint_pos.torch, joint_pos)
        assert torch.allclose(robot.data.joint_vel.torch, joint_vel)

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
        indices, names = robot.find_joints(".*_hip", as_proxy=False)
        assert len(indices) == 2
        assert "FL_hip" in names
        assert "FR_hip" in names

        # Find FL leg joints
        indices, names = robot.find_joints("FL_.*", as_proxy=False)
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
        indices, names = robot.find_bodies("base", as_proxy=False)
        assert indices == [0]

        # Find all thigh bodies
        indices, names = robot.find_bodies(".*_thigh", as_proxy=False)
        assert len(indices) == 2

    def test_set_joint_position_target(self, robot):
        """Test setting joint position targets."""
        target = torch.randn(4, 12)
        robot.set_joint_position_target(target)
        assert torch.allclose(robot.data.joint_pos_target.torch, target)

    def test_joint_limits(self, robot):
        """Test joint limits."""
        limits = robot.data.joint_pos_limits.torch
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
        # root_link_pose_w is wp.transformf so shape is (4,) but converts to (4, 7)
        assert obj.data.root_link_pose_w.torch.shape == (4, 7)
        # root_link_vel_w is wp.spatial_vectorf so shape is (4,) but converts to (4, 6)
        assert obj.data.root_link_vel_w.torch.shape == (4, 6)
        assert obj.data.root_link_state_w.torch.shape == (4, 13)

    def test_body_state_shapes(self, obj):
        """Test body state tensor shapes (single body)."""
        # body_link_pose_w is wp.transformf so shape is (4, 1) but converts to (4, 1, 7)
        assert obj.data.body_link_pose_w.torch.shape == (4, 1, 7)
        # body_link_vel_w is wp.spatial_vectorf so shape is (4, 1) but converts to (4, 1, 6)
        assert obj.data.body_link_vel_w.torch.shape == (4, 1, 6)

    def test_body_properties(self, obj):
        """Test body property shapes."""
        assert obj.data.body_mass.shape == (4, 1)
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
        # body_link_pose_w is wp.transformf so shape is (4, 5) but converts to (4, 5, 7)
        assert collection.data.body_link_pose_w.torch.shape == (4, 5, 7)
        # body_link_vel_w is wp.spatial_vectorf so shape is (4, 5) but converts to (4, 5, 6)
        assert collection.data.body_link_vel_w.torch.shape == (4, 5, 6)
        assert collection.data.body_link_state_w.torch.shape == (4, 5, 13)

    def test_find_bodies_returns_indices(self, collection):
        """Test that find_bodies returns an int32 index tensor."""
        body_ids, names = collection.find_bodies("body_0", as_proxy=False)
        assert isinstance(body_ids, torch.Tensor)
        assert body_ids.dtype == torch.int32
        assert body_ids.device.type == collection.device
        assert body_ids.tolist() == [0]
        assert names == ["body_0"]


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
        # root_link_pose_w is wp.transformf so shape is (3,) but converts to (3, 7)
        assert obj.data.root_link_pose_w.torch.shape == (3, 7)

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
        # body_link_pose_w is wp.transformf so shape is (4, 6) but converts to (4, 6, 7)
        assert collection.data.body_link_pose_w.torch.shape == (4, 6, 7)


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
        assert torch.allclose(robot.data.joint_pos.torch, expected)

    def test_with_joint_limits(self):
        """Test setting joint limits."""

        robot = (
            MockArticulationBuilder()
            .with_num_instances(1)
            .with_joints(["j0", "j1"])
            .with_joint_limits(-1.0, 1.0)
            .build()
        )

        limits = robot.data.joint_pos_limits.torch
        assert torch.all(limits[..., 0] == -1.0)
        assert torch.all(limits[..., 1] == 1.0)
