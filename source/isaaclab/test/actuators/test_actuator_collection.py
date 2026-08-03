# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the actuator collection runtime."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.actuators import (
    ActuatorCollection,
    ActuatorControl,
    ActuatorJointProperties,
    DCMotor,
    DCMotorCfg,
    DelayedPDActuatorCfg,
    IdealPDActuator,
    IdealPDActuatorCfg,
    ImplicitActuator,
    ImplicitActuatorCfg,
)
from isaaclab.actuators.actuator_control import ArticulationActuatorControl
from isaaclab.utils.warp import ProxyArray


def _implicit_cfg() -> ImplicitActuatorCfg:
    """Create a valid implicit actuator config for collection tests."""
    return ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)


def _ideal_cfg(joints: list[str], *, stiffness: float, damping: float, effort_limit: float):
    return IdealPDActuatorCfg(
        joint_names_expr=joints,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
        velocity_limit=100.0,
    )


def _dc_cfg(
    joints: list[str],
    *,
    stiffness: float,
    damping: float,
    effort_limit: float,
    velocity_limit: float,
    saturation_effort: float,
):
    return DCMotorCfg(
        joint_names_expr=joints,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
        velocity_limit=velocity_limit,
        saturation_effort=saturation_effort,
    )


def _make_unbatched_reference(monkeypatch, actuator_type, cfgs, control):
    with monkeypatch.context() as patch:
        patch.setattr(actuator_type, "_supports_execution_aggregation", False)
        return ActuatorCollection(cfgs, control)


def _assign_deterministic_inputs(collection: ActuatorCollection, control: FakeActuatorControl) -> None:
    control.joint_pos.torch.copy_(
        torch.tensor(
            [
                [0.35, -0.80, 1.25, -1.60],
                [-0.45, 0.95, -1.35, 1.80],
            ],
            dtype=torch.float32,
        )
    )
    control.joint_vel.torch.copy_(
        torch.tensor(
            [
                [16.0, 31.0, -17.0, -32.0],
                [-18.0, -33.0, 19.0, 34.0],
            ],
            dtype=torch.float32,
        )
    )
    collection.command.position.torch.copy_(
        torch.tensor(
            [
                [1.40, -0.20, -0.75, 2.20],
                [0.15, -1.45, 2.05, -0.65],
            ],
            dtype=torch.float32,
        )
    )
    collection.command.velocity.torch.copy_(
        torch.tensor(
            [
                [-3.5, 4.25, 5.75, -6.5],
                [7.0, -8.5, -9.25, 10.75],
            ],
            dtype=torch.float32,
        )
    )
    collection.command.effort.torch.copy_(
        torch.tensor(
            [
                [2.25, -3.50, 4.75, -5.25],
                [-6.50, 7.75, -8.25, 9.50],
            ],
            dtype=torch.float32,
        )
    )


def _assert_collection_outputs_match_exactly(actual: ActuatorCollection, reference: ActuatorCollection) -> None:
    torch.testing.assert_close(
        actual.joint_command.position.torch,
        reference.joint_command.position.torch,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual.joint_command.velocity.torch,
        reference.joint_command.velocity.torch,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual.joint_command.effort.torch,
        reference.joint_command.effort.torch,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(actual.computed_torque.torch, reference.computed_torque.torch, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.applied_torque.torch, reference.applied_torque.torch, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual.soft_joint_vel_limits.torch,
        reference.soft_joint_vel_limits.torch,
        rtol=0.0,
        atol=0.0,
    )


class FakeActuatorControl(ActuatorControl):
    """Small backend-neutral control object used by collection unit tests."""

    def __init__(self, *, num_envs: int = 2, joint_names: list[str] | None = None, device: str = "cpu"):
        self._num_instances = num_envs
        self._joint_names = joint_names or ["joint_0", "joint_1", "joint_2"]
        self._device = device
        self._joint_pos = ProxyArray(wp.zeros((num_envs, len(self._joint_names)), dtype=wp.float32, device=device))
        self._joint_vel = ProxyArray(wp.zeros((num_envs, len(self._joint_names)), dtype=wp.float32, device=device))
        self.written_properties: list[tuple[str, bool]] = []
        self.native_gain_writes: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.staged_commands: list[str] = []
        self.submitted = False

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def num_joints(self) -> int:
        return len(self._joint_names)

    @property
    def num_fixed_tendons(self) -> int:
        return 0

    @property
    def device(self) -> str:
        return self._device

    @property
    def joint_pos(self) -> ProxyArray:
        return self._joint_pos

    @property
    def joint_vel(self) -> ProxyArray:
        return self._joint_vel

    def find_joints(self, name_keys: str | Sequence[str]) -> tuple[list[int], list[str]]:
        expressions = [name_keys] if isinstance(name_keys, str) else list(name_keys)
        matches = [
            (joint_id, joint_name)
            for joint_id, joint_name in enumerate(self._joint_names)
            if any(re.fullmatch(expression, joint_name) for expression in expressions)
        ]
        return [joint_id for joint_id, _ in matches], [joint_name for _, joint_name in matches]

    def resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array | None) -> torch.Tensor | wp.array:
        if env_ids is None:
            return wp.array(list(range(self.num_instances)), dtype=wp.int32, device=self.device)
        if isinstance(env_ids, torch.Tensor | wp.array):
            return env_ids
        return wp.array(list(env_ids), dtype=wp.int32, device=self.device)

    def resolve_joint_ids(self, joint_ids: Sequence[int] | torch.Tensor | wp.array | None) -> torch.Tensor | wp.array:
        if joint_ids is None:
            return wp.array(list(range(self.num_joints)), dtype=wp.int32, device=self.device)
        if isinstance(joint_ids, torch.Tensor | wp.array):
            return joint_ids
        return wp.array(list(joint_ids), dtype=wp.int32, device=self.device)

    def resolve_env_mask(self, env_mask: wp.array | None) -> wp.array:
        return (
            env_mask
            if env_mask is not None
            else wp.array([True] * self.num_instances, dtype=wp.bool, device=self.device)
        )

    def resolve_joint_mask(self, joint_mask: wp.array | None) -> wp.array:
        return (
            joint_mask
            if joint_mask is not None
            else wp.array([True] * self.num_joints, dtype=wp.bool, device=self.device)
        )

    def assert_shape_and_dtype(
        self, tensor: torch.Tensor | wp.array | float, shape: tuple[int, ...], dtype: type, name: str
    ) -> None:
        if isinstance(tensor, (float, int)):
            return
        if isinstance(tensor, torch.Tensor):
            assert tuple(tensor.shape) == shape
            return
        assert tensor.shape == shape
        assert tensor.dtype == dtype

    def assert_shape_and_dtype_mask(
        self, tensor: torch.Tensor | wp.array | float, masks: tuple[wp.array, ...], dtype: type, name: str
    ) -> None:
        self.assert_shape_and_dtype(tensor, tuple(mask.shape[0] for mask in masks), dtype, name)

    def get_default_joint_properties(self, joint_ids: torch.Tensor | wp.array | slice) -> ActuatorJointProperties:
        if isinstance(joint_ids, slice):
            num_joints = self.num_joints
        else:
            num_joints = joint_ids.shape[0]
        shape = (self.num_instances, num_joints)
        zeros = torch.zeros(shape, dtype=torch.float32, device=self.device)
        ones = torch.ones(shape, dtype=torch.float32, device=self.device)
        return ActuatorJointProperties(
            stiffness=zeros,
            damping=zeros,
            armature=zeros,
            friction=zeros,
            dynamic_friction=zeros,
            viscous_friction=zeros,
            effort_limit=ones * 100.0,
            velocity_limit=ones * 10.0,
        )

    def write_resolved_joint_properties(self, actuator, *, native_managed: bool) -> None:
        self.written_properties.append((actuator.__class__.__name__, native_managed))

    def write_native_actuator_gain(self, attr, values, env_ids, joint_ids) -> None:
        self.native_gain_writes.append((attr, values.clone(), env_ids.clone(), joint_ids.clone()))

    def stage_user_command(
        self,
        command_name: str,
        collection: ActuatorCollection,
        env_ids: torch.Tensor | wp.array | None,
        joint_ids: torch.Tensor | wp.array | None,
        env_mask: wp.array | None,
        joint_mask: wp.array | None,
    ) -> None:
        self.staged_commands.append(command_name)

    def submit_commands(self, collection: ActuatorCollection) -> None:
        self.submitted = True


class NativeFakeActuatorControl(FakeActuatorControl):
    """Control object that handles actuator execution natively."""

    @property
    def native_active(self) -> bool:
        return True

    def compute_native_actuators(self, collection: ActuatorCollection, dt: float) -> bool:
        return True


class ProxyFinderActuatorControl(FakeActuatorControl):
    """Control object whose joint finder returns cached proxy indices."""

    def find_joints(self, name_keys: str | Sequence[str]) -> tuple[ProxyArray, list[str]]:
        return ProxyArray(wp.array([0, 2], dtype=wp.int32, device=self.device)), ["joint_0", "joint_2"]


class FakeArticulationActuatorControl(ArticulationActuatorControl):
    """Concrete shared articulation-control test adapter."""

    def submit_commands(self, collection: ActuatorCollection) -> None:
        pass


class FakeArticulation:
    """Small articulation facade for shared control tests."""

    def __init__(self):
        self.num_instances = 2
        self.num_joints = 3
        self.num_fixed_tendons = 0
        self.device = "cpu"
        shape = (self.num_instances, self.num_joints)
        zeros = torch.zeros(shape, dtype=torch.float32)
        ones = torch.ones(shape, dtype=torch.float32)
        self.data = SimpleNamespace(
            joint_pos=ProxyArray(wp.zeros(shape, dtype=wp.float32, device=self.device)),
            joint_vel=ProxyArray(wp.zeros(shape, dtype=wp.float32, device=self.device)),
            joint_stiffness=SimpleNamespace(torch=zeros),
            joint_damping=SimpleNamespace(torch=zeros),
            joint_armature=SimpleNamespace(torch=zeros),
            joint_friction_coeff=SimpleNamespace(torch=zeros),
            joint_effort_limits=SimpleNamespace(torch=ones * 100.0),
            joint_vel_limits=SimpleNamespace(torch=ones * 10.0),
        )
        self.calls: list[tuple[str, dict]] = []

    def find_joints(
        self, name_keys: str | Sequence[str], *, as_proxy: bool = False
    ) -> tuple[list[int] | ProxyArray, list[str]]:
        joint_ids = list(range(self.num_joints))
        if as_proxy:
            resolved_ids = ProxyArray(wp.array(joint_ids, dtype=wp.int32, device=self.device))
        else:
            resolved_ids = joint_ids
        return resolved_ids, ["joint_0", "joint_1", "joint_2"]

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array:
        values = list(range(self.num_instances)) if env_ids is None else list(env_ids)
        return wp.array(values, dtype=wp.int32, device=self.device)

    def _resolve_joint_ids(self, joint_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array:
        values = list(range(self.num_joints)) if joint_ids is None else list(joint_ids)
        return wp.array(values, dtype=wp.int32, device=self.device)

    def _resolve_env_mask(self, env_mask: wp.array | None) -> wp.array:
        return (
            env_mask
            if env_mask is not None
            else wp.array([True] * self.num_instances, dtype=wp.bool, device=self.device)
        )

    def _resolve_joint_mask(self, joint_mask: wp.array | None) -> wp.array:
        return (
            joint_mask
            if joint_mask is not None
            else wp.array([True] * self.num_joints, dtype=wp.bool, device=self.device)
        )

    def assert_shape_and_dtype(
        self, tensor: torch.Tensor | wp.array | float, shape: tuple[int, ...], dtype: type, name: str
    ) -> None:
        pass

    def assert_shape_and_dtype_mask(
        self, tensor: torch.Tensor | wp.array | float, masks: tuple[wp.array, ...], dtype: type, name: str
    ) -> None:
        pass

    def write_joint_effort_limit_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("effort_limit", kwargs))

    def write_joint_velocity_limit_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("velocity_limit", kwargs))

    def write_joint_armature_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("armature", kwargs))

    def write_joint_friction_coefficient_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("friction", kwargs))

    def write_joint_stiffness_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("stiffness", kwargs))

    def write_joint_damping_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("damping", kwargs))


def test_articulation_control_provides_common_forwarding_and_property_writes():
    articulation = FakeArticulation()
    control = FakeArticulationActuatorControl(articulation)

    assert control.num_instances == articulation.num_instances
    assert control.num_joints == articulation.num_joints
    assert control.device == articulation.device

    defaults = control.get_default_joint_properties(slice(None))
    torch.testing.assert_close(defaults.dynamic_friction, torch.zeros(2, 3))
    torch.testing.assert_close(defaults.viscous_friction, torch.zeros(2, 3))

    actuator = SimpleNamespace(
        effort_limit_sim=torch.ones((2, 3)),
        velocity_limit_sim=torch.ones((2, 3)) * 2.0,
        armature=torch.ones((2, 3)) * 3.0,
        friction=torch.ones((2, 3)) * 4.0,
        stiffness=torch.ones((2, 3)) * 5.0,
        damping=torch.ones((2, 3)) * 6.0,
        joint_indices=slice(None),
    )

    control.write_resolved_joint_properties(actuator, native_managed=False)

    assert [name for name, _ in articulation.calls] == [
        "effort_limit",
        "velocity_limit",
        "armature",
        "friction",
        "stiffness",
        "damping",
    ]
    assert articulation.calls[-2][1]["stiffness"] == 0.0
    assert articulation.calls[-1][1]["damping"] == 0.0


def test_collection_is_mapping_like_and_read_only():
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)

    assert list(collection.keys()) == ["all"]
    assert collection["all"] is next(iter(collection.values()))
    assert list(collection.items())[0][0] == "all"
    with pytest.raises(TypeError, match="membership is fixed"):
        collection["new"] = collection["all"]


def test_same_stateless_class_builds_one_execution_batch_with_group_views():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": _ideal_cfg(["joint_0", "joint_2"], stiffness=10.0, damping=1.0, effort_limit=20.0),
            "knees": _ideal_cfg(["joint_1", "joint_3"], stiffness=30.0, damping=2.0, effort_limit=40.0),
        },
        control,
    )

    assert len(collection._execution_batches) == 1
    batch = collection._execution_batches[0]
    assert type(batch.actuator) is IdealPDActuator
    assert batch.group_names == ("hips", "knees")
    assert isinstance(collection["hips"], IdealPDActuator)
    assert collection["hips"].joint_names == ["joint_0", "joint_2"]
    assert collection["hips"].stiffness.shape == (2, 2)
    torch.testing.assert_close(batch.actuator.stiffness[:, :2], torch.full((2, 2), 10.0))
    torch.testing.assert_close(batch.actuator.stiffness[:, 2:], torch.full((2, 2), 30.0))

    collection["hips"].stiffness.fill_(17.0)
    torch.testing.assert_close(batch.actuator.stiffness[:, :2], torch.full((2, 2), 17.0))
    torch.testing.assert_close(batch.actuator.stiffness[:, 2:], torch.full((2, 2), 30.0))


def test_dc_motor_execution_batch_packs_different_saturation_efforts():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": _dc_cfg(
                ["joint_0", "joint_1"],
                stiffness=20.0,
                damping=1.0,
                effort_limit=40.0,
                velocity_limit=10.0,
                saturation_effort=60.0,
            ),
            "knees": _dc_cfg(
                ["joint_2", "joint_3"],
                stiffness=30.0,
                damping=2.0,
                effort_limit=70.0,
                velocity_limit=20.0,
                saturation_effort=120.0,
            ),
        },
        control,
    )

    batch = collection._execution_batches[0]
    assert type(batch.actuator) is DCMotor
    torch.testing.assert_close(
        batch.actuator._saturation_effort,
        torch.tensor([[60.0, 60.0, 120.0, 120.0]]).expand(2, -1),
    )


def test_ideal_pd_aggregate_matches_independent_groups_exactly(monkeypatch):
    joint_names = [f"joint_{index}" for index in range(4)]
    cfgs = {
        "hips": _ideal_cfg(["joint_0", "joint_2"], stiffness=12.0, damping=1.5, effort_limit=18.0),
        "knees": _ideal_cfg(["joint_1", "joint_3"], stiffness=27.0, damping=2.25, effort_limit=31.0),
    }
    reference_control = FakeActuatorControl(joint_names=joint_names)
    actual_control = FakeActuatorControl(joint_names=joint_names)
    reference = _make_unbatched_reference(monkeypatch, IdealPDActuator, cfgs, reference_control)
    actual = ActuatorCollection(cfgs, actual_control)
    _assign_deterministic_inputs(reference, reference_control)
    _assign_deterministic_inputs(actual, actual_control)

    reference.compute()
    actual.compute()

    _assert_collection_outputs_match_exactly(actual, reference)


def test_dc_motor_aggregate_matches_independent_groups_exactly(monkeypatch):
    joint_names = [f"joint_{index}" for index in range(4)]
    cfgs = {
        "hips": _dc_cfg(
            ["joint_0", "joint_2"],
            stiffness=14.0,
            damping=1.25,
            effort_limit=20.0,
            velocity_limit=10.0,
            saturation_effort=40.0,
        ),
        "knees": _dc_cfg(
            ["joint_1", "joint_3"],
            stiffness=23.0,
            damping=2.5,
            effort_limit=30.0,
            velocity_limit=20.0,
            saturation_effort=60.0,
        ),
    }
    reference_control = FakeActuatorControl(joint_names=joint_names)
    actual_control = FakeActuatorControl(joint_names=joint_names)
    reference = _make_unbatched_reference(monkeypatch, DCMotor, cfgs, reference_control)
    actual = ActuatorCollection(cfgs, actual_control)
    _assign_deterministic_inputs(reference, reference_control)
    _assign_deterministic_inputs(actual, actual_control)

    reference.compute()
    actual.compute()

    _assert_collection_outputs_match_exactly(actual, reference)


def test_implicit_aggregate_matches_independent_groups_exactly(monkeypatch):
    joint_names = [f"joint_{index}" for index in range(4)]
    cfgs = {
        "hips": ImplicitActuatorCfg(
            joint_names_expr=["joint_0", "joint_2"],
            stiffness=9.0,
            damping=0.75,
            effort_limit_sim=16.0,
            velocity_limit=7.0,
            velocity_limit_sim=70.0,
        ),
        "knees": ImplicitActuatorCfg(
            joint_names_expr=["joint_1", "joint_3"],
            stiffness=19.0,
            damping=1.75,
            effort_limit_sim=28.0,
            velocity_limit=11.0,
            velocity_limit_sim=110.0,
        ),
    }
    reference_control = FakeActuatorControl(joint_names=joint_names)
    actual_control = FakeActuatorControl(joint_names=joint_names)
    reference = _make_unbatched_reference(monkeypatch, ImplicitActuator, cfgs, reference_control)
    actual = ActuatorCollection(cfgs, actual_control)
    _assign_deterministic_inputs(reference, reference_control)
    _assign_deterministic_inputs(actual, actual_control)

    reference.compute()
    actual.compute()

    _assert_collection_outputs_match_exactly(actual, reference)


def test_aggregate_computes_once_and_refreshes_group_outputs(monkeypatch):
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(6)])
    collection = ActuatorCollection(
        {
            "hips": _ideal_cfg(["joint_0", "joint_3"], stiffness=8.0, damping=0.5, effort_limit=12.0),
            "knees": _ideal_cfg(["joint_1", "joint_4"], stiffness=13.0, damping=1.0, effort_limit=18.0),
            "ankles": _ideal_cfg(["joint_2", "joint_5"], stiffness=21.0, damping=1.5, effort_limit=27.0),
        },
        control,
    )
    compute_calls = 0
    scatter_calls = 0
    original_compute = IdealPDActuator.compute
    original_scatter = collection._scatter_actuator_output

    def counted_compute(self, control_action, joint_pos, joint_vel):
        nonlocal compute_calls
        compute_calls += 1
        return original_compute(self, control_action, joint_pos, joint_vel)

    def counted_scatter(actuator, control_action, joint_indices=None):
        nonlocal scatter_calls
        scatter_calls += 1
        if joint_indices is None:
            return original_scatter(actuator, control_action)
        return original_scatter(actuator, control_action, joint_indices)

    monkeypatch.setattr(IdealPDActuator, "compute", counted_compute)
    monkeypatch.setattr(collection, "_scatter_actuator_output", counted_scatter)
    collection.command.position.torch.copy_(torch.arange(12, dtype=torch.float32).reshape(2, 6) + 0.25)
    collection.command.velocity.torch.copy_(torch.arange(12, dtype=torch.float32).reshape(2, 6) * -0.5 - 0.75)
    collection.command.effort.torch.copy_(torch.arange(12, dtype=torch.float32).reshape(2, 6) + 1.5)

    collection.compute()

    assert compute_calls == 1
    assert scatter_calls == 1
    batch = collection._execution_batches[0]
    for group_name, group_slice in zip(batch.group_names, batch.group_slices):
        torch.testing.assert_close(
            collection[group_name].computed_effort,
            batch.actuator.computed_effort[:, group_slice],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            collection[group_name].applied_effort,
            batch.actuator.applied_effort[:, group_slice],
            rtol=0.0,
            atol=0.0,
        )
    first_hips_output = collection["hips"].computed_effort
    collection.command.position.torch.mul_(-1.25)
    collection.command.velocity.torch.add_(2.75)
    collection.command.effort.torch.sub_(4.5)

    collection.compute()

    assert compute_calls == 2
    assert scatter_calls == 2
    assert collection["hips"].computed_effort is not first_hips_output
    for group_name, group_slice in zip(batch.group_names, batch.group_slices):
        torch.testing.assert_close(
            collection[group_name].computed_effort,
            batch.actuator.computed_effort[:, group_slice],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            collection[group_name].applied_effort,
            batch.actuator.applied_effort[:, group_slice],
            rtol=0.0,
            atol=0.0,
        )


def test_stateful_subclasses_and_overlapping_groups_remain_unbatched():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    delayed = ActuatorCollection(
        {
            "first": DelayedPDActuatorCfg(
                joint_names_expr=["joint_0", "joint_1"], stiffness=1.0, damping=1.0, max_delay=0
            ),
            "second": DelayedPDActuatorCfg(
                joint_names_expr=["joint_2", "joint_3"], stiffness=2.0, damping=2.0, max_delay=0
            ),
        },
        control,
    )
    assert len(delayed._execution_batches) == 2

    overlapping = ActuatorCollection(
        {
            "first": _ideal_cfg(["joint_0", "joint_1"], stiffness=1.0, damping=1.0, effort_limit=10.0),
            "second": _ideal_cfg(["joint_1", "joint_2"], stiffness=2.0, damping=2.0, effort_limit=20.0),
        },
        FakeActuatorControl(joint_names=["joint_0", "joint_1", "joint_2"]),
    )
    assert len(overlapping._execution_batches) == 2

    cross_class = ActuatorCollection(
        {
            "ideal_a": _ideal_cfg(["joint_0"], stiffness=1.0, damping=1.0, effort_limit=10.0),
            "dc": _dc_cfg(
                ["joint_1", "joint_2"],
                stiffness=2.0,
                damping=2.0,
                effort_limit=20.0,
                velocity_limit=10.0,
                saturation_effort=30.0,
            ),
            "ideal_b": _ideal_cfg(["joint_1"], stiffness=3.0, damping=3.0, effort_limit=30.0),
        },
        FakeActuatorControl(joint_names=["joint_0", "joint_1", "joint_2"]),
    )
    assert len(cross_class._execution_batches) == 3


def test_runtime_gains_route_into_aggregate_and_native_hook():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": _dc_cfg(
                ["joint_0", "joint_1"],
                stiffness=20.0,
                damping=1.0,
                effort_limit=40.0,
                velocity_limit=10.0,
                saturation_effort=60.0,
            ),
            "knees": _dc_cfg(
                ["joint_2", "joint_3"],
                stiffness=30.0,
                damping=2.0,
                effort_limit=70.0,
                velocity_limit=20.0,
                saturation_effort=120.0,
            ),
        },
        control,
    )
    env_ids = torch.tensor([1], dtype=torch.long)

    collection.write_actuator_stiffness_to_sim(
        stiffness=torch.tensor([[71.0, 93.0]]),
        env_ids=env_ids,
        joint_ids=torch.tensor([0, 3], dtype=torch.long),
    )

    torch.testing.assert_close(collection["hips"].stiffness[1, 0], torch.tensor(71.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection["knees"].stiffness[1, 1], torch.tensor(93.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection.actuator_stiffness.torch[1, 0], torch.tensor(71.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection.actuator_stiffness.torch[1, 3], torch.tensor(93.0), rtol=0.0, atol=0.0)
    assert control.native_gain_writes[-1][0] == "kp"

    collection.write_actuator_damping_to_sim(
        damping=torch.tensor([[47.0, 29.0]]),
        env_ids=env_ids,
        joint_ids=torch.tensor([3, 0], dtype=torch.long),
    )

    torch.testing.assert_close(collection["knees"].damping[1, 1], torch.tensor(47.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection["hips"].damping[1, 0], torch.tensor(29.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection.actuator_damping.torch[1, 3], torch.tensor(47.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection.actuator_damping.torch[1, 0], torch.tensor(29.0), rtol=0.0, atol=0.0)
    assert control.native_gain_writes[-1][0] == "kd"


def test_native_execution_bypasses_lab_aggregation_and_keeps_group_gains_current(monkeypatch):
    control = NativeFakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": _dc_cfg(
                ["joint_0", "joint_1"],
                stiffness=20.0,
                damping=1.0,
                effort_limit=40.0,
                velocity_limit=10.0,
                saturation_effort=60.0,
            ),
            "knees": _dc_cfg(
                ["joint_2", "joint_3"],
                stiffness=30.0,
                damping=2.0,
                effort_limit=70.0,
                velocity_limit=20.0,
                saturation_effort=120.0,
            ),
        },
        control,
    )

    assert len(collection._execution_batches) == 2
    assert all(len(batch.group_names) == 1 for batch in collection._execution_batches)

    def fail_compute(*args, **kwargs):
        raise AssertionError("Lab actuator execution must be bypassed")

    monkeypatch.setattr(DCMotor, "compute", fail_compute)
    collection.compute()
    collection.write_actuator_stiffness_to_sim(
        stiffness=torch.tensor([[71.0, 93.0]]),
        env_ids=torch.tensor([1], dtype=torch.long),
        joint_ids=torch.tensor([0, 3], dtype=torch.long),
    )

    torch.testing.assert_close(collection["hips"].stiffness[1, 0], torch.tensor(71.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection["knees"].stiffness[1, 1], torch.tensor(93.0), rtol=0.0, atol=0.0)


def test_collection_exports_proxy_arrays():
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)

    assert collection.command.position.shape == (2, 3)
    assert collection.command.velocity.shape == (2, 3)
    assert collection.command.effort.shape == (2, 3)
    assert collection.joint_command.position.shape == (2, 3)
    assert collection.joint_command.velocity.shape == (2, 3)
    assert collection.joint_command.effort.shape == (2, 3)
    assert collection.computed_torque.shape == (2, 3)
    assert collection.applied_torque.shape == (2, 3)
    assert collection.gear_ratio.shape == (2, 3)


def test_collection_accepts_cached_proxy_joint_indices():
    control = ProxyFinderActuatorControl()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        collection = ActuatorCollection({"outer": _implicit_cfg()}, control)

    assert not [warning for warning in caught_warnings if warning.category is DeprecationWarning]
    torch.testing.assert_close(collection["outer"].joint_indices, torch.tensor([0, 2], dtype=torch.int32))


def test_write_command_index_updates_only_selected_cells():
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)
    value = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    collection.command.set_position_index(value=value, env_ids=[1], joint_ids=[0, 2])

    expected = torch.zeros(2, 3)
    expected[1, 0] = 1.0
    expected[1, 2] = 2.0
    torch.testing.assert_close(collection.command.position.torch.cpu(), expected)
    assert control.staged_commands == ["position"]


def test_write_command_index_accepts_signed_int64_selectors():
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)
    value = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    env_ids = torch.tensor([1], dtype=torch.int64)
    joint_ids = wp.array([0, 2], dtype=wp.int64, device="cpu")

    collection.command.set_position_index(value=value, env_ids=env_ids, joint_ids=joint_ids)

    expected = torch.zeros(2, 3)
    expected[1, 0] = 3.0
    expected[1, 2] = 4.0
    torch.testing.assert_close(collection.command.position.torch.cpu(), expected)


def test_write_command_mask_uses_full_sized_value():
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    env_mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    joint_mask = wp.array([False, True, True], dtype=wp.bool, device="cpu")

    collection.command.set_velocity_mask(value=value, env_mask=env_mask, joint_mask=joint_mask)

    expected = torch.zeros(2, 3)
    expected[0, 1:] = value[0, 1:]
    torch.testing.assert_close(collection.command.velocity.torch.cpu(), expected)
    assert control.staged_commands == ["velocity"]


def test_compute_submits_processed_commands():
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)
    value = torch.ones(2, 3, dtype=torch.float32)
    collection.command.set_position_index(value=value, full_data=True)

    collection.compute()
    collection.submit_commands()

    torch.testing.assert_close(collection.joint_command.position.torch.cpu(), value)
    assert control.submitted
