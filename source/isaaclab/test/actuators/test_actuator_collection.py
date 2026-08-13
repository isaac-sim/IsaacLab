# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the actuator collection runtime."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
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


def _implicit_cfg(**kwargs) -> ImplicitActuatorCfg:
    """Create a valid implicit actuator config for collection tests."""
    return ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0, **kwargs)


class SelectorRecordingActuator(ImplicitActuator):
    """Custom actuator that records the selector supplied to :meth:`compute`."""

    def compute(self, control_action, joint_pos, joint_vel):
        self.observed_joint_indices = control_action.joint_indices
        return super().compute(control_action, joint_pos, joint_vel)


def _ideal_cfg(joints: list[str], *, stiffness: float, damping: float, effort_limit: float):
    return IdealPDActuatorCfg(
        joint_names_expr=joints,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
        velocity_limit=100.0,
    )


def _ideal_pd_cfg(**kwargs) -> IdealPDActuatorCfg:
    """Create a minimal ideal PD actuator configuration."""
    return IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0, **kwargs)


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
        patch.delattr(actuator_type, "_EXECUTION_PARAMETER_NAMES")
        reference = ActuatorCollection(cfgs, control)
        assert len(reference._execution_batches) == len(cfgs)
        return reference


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
    torch.testing.assert_close(actual.computed_effort.torch, reference.computed_effort.torch, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual.applied_effort.torch, reference.applied_effort.torch, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        wp.to_torch(actual._soft_joint_vel_limits),
        wp.to_torch(reference._soft_joint_vel_limits),
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
        shape = (num_envs, len(self._joint_names))
        self._joint_stiffness = ProxyArray(wp.zeros(shape, dtype=wp.float32, device=device))
        self._joint_damping = ProxyArray(wp.zeros(shape, dtype=wp.float32, device=device))
        self._joint_effort_limits = ProxyArray(wp.full(shape, 100.0, dtype=wp.float32, device=device))
        zeros = torch.zeros(shape, dtype=torch.float32, device=device)
        ones = torch.ones(shape, dtype=torch.float32, device=device)
        self._current_joint_properties = ActuatorJointProperties(
            stiffness=self._joint_stiffness.torch,
            damping=self._joint_damping.torch,
            armature=zeros.clone(),
            friction=zeros.clone(),
            dynamic_friction=zeros.clone(),
            viscous_friction=zeros.clone(),
            effort_limit=self._joint_effort_limits.torch,
            velocity_limit=ones * 10.0,
        )
        self.written_properties: list[tuple[ActuatorJointProperties, torch.Tensor | slice, bool, bool]] = []
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

    @property
    def joint_stiffness(self) -> ProxyArray:
        return self._joint_stiffness

    @property
    def joint_damping(self) -> ProxyArray:
        return self._joint_damping

    @property
    def joint_effort_limits(self) -> ProxyArray:
        return self._joint_effort_limits

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
        if isinstance(joint_ids, wp.array):
            joint_ids = wp.to_torch(joint_ids).to(device=self.device, dtype=torch.long)
        properties = self._current_joint_properties
        return ActuatorJointProperties(
            stiffness=self.joint_stiffness.torch[:, joint_ids].clone(),
            damping=self.joint_damping.torch[:, joint_ids].clone(),
            armature=properties.armature[:, joint_ids].clone(),
            friction=properties.friction[:, joint_ids].clone(),
            dynamic_friction=properties.dynamic_friction[:, joint_ids].clone(),
            viscous_friction=properties.viscous_friction[:, joint_ids].clone(),
            effort_limit=self.joint_effort_limits.torch[:, joint_ids].clone(),
            velocity_limit=properties.velocity_limit[:, joint_ids].clone(),
        )

    def write_resolved_joint_properties(
        self,
        properties: ActuatorJointProperties,
        joint_ids: torch.Tensor | slice,
        *,
        implicit: bool,
        native_managed: bool,
    ) -> None:
        self.written_properties.append((properties, joint_ids, implicit, native_managed))
        self.joint_effort_limits.torch[:, joint_ids] = properties.effort_limit
        if implicit and not native_managed:
            self.joint_stiffness.torch[:, joint_ids] = properties.stiffness
            self.joint_damping.torch[:, joint_ids] = properties.damping
        else:
            self.joint_stiffness.torch[:, joint_ids] = 0.0
            self.joint_damping.torch[:, joint_ids] = 0.0

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
    def native_actuator_path_active(self) -> bool:
        return True

    def prepare_native_actuators(self, collection, actuator_cfgs) -> set[str]:
        self.prepared_actuator_cfgs = actuator_cfgs
        return set(actuator_cfgs)

    def compute_native_actuators(self, collection: ActuatorCollection, dt: float) -> bool:
        return True


class NativeGainFakeActuatorControl(NativeFakeActuatorControl):
    """Native control whose gains model controller-owned storage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = (self.num_instances, self.num_joints)
        self.native_gains = {
            "kp": torch.zeros(shape, dtype=torch.float32, device=self.device),
            "kd": torch.zeros(shape, dtype=torch.float32, device=self.device),
        }

    def get_native_actuator_gain(self, attr, joint_ids: torch.Tensor | slice) -> torch.Tensor | None:
        gains = self.native_gains.get(attr)
        return None if gains is None else gains[:, joint_ids]

    def write_native_actuator_gain(self, attr, values, env_ids, joint_ids) -> None:
        super().write_native_actuator_gain(attr, values, env_ids, joint_ids)
        self.native_gains[attr][env_ids[:, None], joint_ids] = values


def test_legacy_actuator_control_remains_concrete_without_implicit_drive_arrays():
    drive_property_names = {"joint_stiffness", "joint_damping", "joint_effort_limits"}
    legacy_members = {
        name: FakeActuatorControl.__dict__[name] for name in ActuatorControl.__abstractmethods__ - drive_property_names
    }
    legacy_control_type = type(
        "LegacyActuatorControl",
        (ActuatorControl,),
        {"__init__": FakeActuatorControl.__init__, **legacy_members},
    )

    control = legacy_control_type()

    for name in drive_property_names:
        with pytest.raises(
            NotImplementedError,
            match=rf"{name}.*Lab implicit actuator execution.*articulation-order.*ProxyArray",
        ):
            getattr(control, name)


def test_deprecated_joint_limit_aliases_warn_and_forward():
    control = NativeFakeActuatorControl()
    cfg = _ideal_pd_cfg(effort_limit_sim=12.0, velocity_limit_sim=34.0)

    with pytest.warns(DeprecationWarning, match="joint_effort_limit"):
        collection = ActuatorCollection({"motor": cfg}, control)

    assert cfg.joint_effort_limit is None
    assert cfg.joint_velocity_limit is None
    assert control.prepared_actuator_cfgs["motor"] is not cfg
    assert control.prepared_actuator_cfgs["motor"].joint_effort_limit == 12.0
    assert control.prepared_actuator_cfgs["motor"].joint_velocity_limit == 34.0
    assert collection["motor"].cfg.joint_effort_limit == 12.0
    assert collection["motor"].cfg.joint_velocity_limit == 34.0


@pytest.mark.parametrize(
    (
        "canonical_name",
        "alias_name",
        "canonical_value",
        "alias_value",
        "cfg_factory",
        "control_factory",
        "warns",
    ),
    [
        (
            "joint_effort_limit",
            "effort_limit_sim",
            12.0,
            {"joint_.*": 12.0},
            _ideal_pd_cfg,
            NativeFakeActuatorControl,
            True,
        ),
        (
            "joint_velocity_limit",
            "velocity_limit_sim",
            34.0,
            {"joint_.*": 34.0},
            _ideal_pd_cfg,
            NativeFakeActuatorControl,
            True,
        ),
        (
            "joint_effort_limit",
            "effort_limit",
            {"joint_.*": 12.0},
            12.0,
            _implicit_cfg,
            FakeActuatorControl,
            False,
        ),
    ],
)
def test_equivalent_joint_limit_aliases_prefer_canonical_value(
    canonical_name, alias_name, canonical_value, alias_value, cfg_factory, control_factory, warns
):
    cfg = cfg_factory(**{canonical_name: canonical_value, alias_name: alias_value})
    warning_context = pytest.warns(DeprecationWarning) if warns else nullcontext()
    with warning_context:
        collection = ActuatorCollection({"motor": cfg}, control_factory())

    assert getattr(collection["motor"].cfg, canonical_name) == canonical_value
    assert getattr(cfg, canonical_name) == canonical_value
    assert getattr(cfg, alias_name) == alias_value


@pytest.mark.parametrize(
    ("canonical_name", "alias_name", "cfg_factory", "control_factory", "warns"),
    [
        ("joint_effort_limit", "effort_limit_sim", _ideal_pd_cfg, NativeFakeActuatorControl, True),
        ("joint_velocity_limit", "velocity_limit_sim", _ideal_pd_cfg, NativeFakeActuatorControl, True),
        ("joint_effort_limit", "effort_limit", _implicit_cfg, FakeActuatorControl, False),
    ],
)
def test_conflicting_joint_limit_aliases_raise(canonical_name, alias_name, cfg_factory, control_factory, warns):
    cfg = cfg_factory(
        **{
            canonical_name: {"joint_0": 12.0, "joint_1": 10.0, "joint_2": 8.0},
            alias_name: {"joint_0": 12.0, "joint_1": 11.0, "joint_2": 8.0},
        }
    )

    warning_context = pytest.warns(DeprecationWarning) if warns else nullcontext()
    with warning_context, pytest.raises(ValueError, match=rf"motor.*{canonical_name}.*{alias_name}"):
        ActuatorCollection({"motor": cfg}, control_factory())


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
            joint_stiffness=ProxyArray(wp.zeros(shape, dtype=wp.float32, device=self.device)),
            joint_damping=ProxyArray(wp.zeros(shape, dtype=wp.float32, device=self.device)),
            joint_armature=SimpleNamespace(torch=zeros.clone()),
            joint_friction_coeff=SimpleNamespace(torch=zeros.clone()),
            joint_effort_limits=ProxyArray(wp.full(shape, 100.0, dtype=wp.float32, device=self.device)),
            joint_vel_limits=SimpleNamespace(torch=ones * 10.0),
        )
        self.calls: list[tuple[str, dict]] = []
        self.resolved_env_ids: list[object] = []
        self.resolved_joint_ids: list[object] = []

    def find_joints(
        self, name_keys: str | Sequence[str], *, as_proxy: bool = False
    ) -> tuple[list[int] | ProxyArray, list[str]]:
        expressions = [name_keys] if isinstance(name_keys, str) else list(name_keys)
        matches = [
            (joint_id, joint_name)
            for joint_id, joint_name in enumerate(["joint_0", "joint_1", "joint_2"])
            if any(re.fullmatch(expression, joint_name) for expression in expressions)
        ]
        joint_ids = [joint_id for joint_id, _ in matches]
        joint_names = [joint_name for _, joint_name in matches]
        if as_proxy:
            resolved_ids = ProxyArray(wp.array(joint_ids, dtype=wp.int32, device=self.device))
        else:
            resolved_ids = joint_ids
        return resolved_ids, joint_names

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array:
        self.resolved_env_ids.append(env_ids)
        values = list(range(self.num_instances)) if env_ids is None else list(env_ids)
        return wp.array(values, dtype=wp.int32, device=self.device)

    def _resolve_joint_ids(self, joint_ids: Sequence[int] | torch.Tensor | wp.array | None) -> wp.array:
        self.resolved_joint_ids.append(joint_ids)
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

    def write_joint_dynamic_friction_coefficient_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("dynamic_friction", kwargs))

    def write_joint_viscous_friction_coefficient_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("viscous_friction", kwargs))

    def write_joint_stiffness_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("stiffness", kwargs))
        self.data.joint_stiffness.torch[:, kwargs["joint_ids"]] = kwargs["stiffness"]

    def write_joint_damping_to_sim_index(self, **kwargs) -> None:
        self.calls.append(("damping", kwargs))
        self.data.joint_damping.torch[:, kwargs["joint_ids"]] = kwargs["damping"]


def test_articulation_control_provides_common_forwarding_and_property_writes():
    articulation = FakeArticulation()
    control = FakeArticulationActuatorControl(articulation)

    assert not FakeActuatorControl().native_actuator_path_active
    assert not control.native_actuator_path_active
    control._native_actuator_path_active = True
    assert control.native_actuator_path_active

    assert control.num_instances == articulation.num_instances
    assert control.num_joints == articulation.num_joints
    assert control.device == articulation.device
    assert control.joint_stiffness is articulation.data.joint_stiffness
    assert control.joint_damping is articulation.data.joint_damping
    assert control.joint_effort_limits is articulation.data.joint_effort_limits

    defaults = control.get_default_joint_properties(slice(None))
    torch.testing.assert_close(defaults.dynamic_friction, torch.zeros(2, 3))
    torch.testing.assert_close(defaults.viscous_friction, torch.zeros(2, 3))

    properties = ActuatorJointProperties(
        effort_limit=torch.ones((2, 3)),
        velocity_limit=torch.ones((2, 3)) * 2.0,
        armature=torch.ones((2, 3)) * 3.0,
        friction=torch.ones((2, 3)) * 4.0,
        dynamic_friction=torch.zeros((2, 3)),
        viscous_friction=torch.zeros((2, 3)),
        stiffness=torch.ones((2, 3)) * 5.0,
        damping=torch.ones((2, 3)) * 6.0,
    )

    control.write_resolved_joint_properties(properties, slice(None), implicit=False, native_managed=False)

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
    control.resolve_env_ids((1,))
    control.resolve_joint_ids(range(1, 3))
    assert articulation.resolved_env_ids == [[1]]
    assert articulation.resolved_joint_ids == [[1, 2]]


def test_articulation_control_projects_warp_joint_property_selectors():
    articulation = FakeArticulation()
    control = FakeArticulationActuatorControl(articulation)
    articulation.data.joint_armature.torch.copy_(
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
    )
    joint_ids = wp.array([2, 0], dtype=wp.int32, device="cpu")

    defaults = control.get_default_joint_properties(joint_ids)

    expected = torch.tensor(
        [
            [3.0, 1.0],
            [6.0, 4.0],
        ]
    )
    torch.testing.assert_close(defaults.armature, expected)


def test_native_unset_gains_capture_authored_defaults_before_solver_zero(monkeypatch):
    """Keep authored gains in a native explicit group while zeroing its solver drives."""
    articulation = FakeArticulation()
    articulation.data.joint_stiffness.torch.fill_(17.0)
    articulation.data.joint_damping.torch.fill_(3.0)
    control = FakeArticulationActuatorControl(articulation)
    monkeypatch.setattr(control, "prepare_native_actuators", lambda collection, cfgs: set(cfgs))

    collection = ActuatorCollection(
        {
            "explicit": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
                effort_limit=100.0,
                velocity_limit=10.0,
            )
        },
        control,
    )

    torch.testing.assert_close(collection["explicit"].stiffness, torch.full((2, 3), 17.0))
    torch.testing.assert_close(collection["explicit"].damping, torch.full((2, 3), 3.0))
    torch.testing.assert_close(articulation.data.joint_stiffness.torch, torch.zeros((2, 3)))
    torch.testing.assert_close(articulation.data.joint_damping.torch, torch.zeros((2, 3)))
    assert articulation.calls[-2][1]["stiffness"] == 0.0
    assert articulation.calls[-1][1]["damping"] == 0.0


def test_native_group_gains_project_live_controller_values_without_local_mirrors():
    """Read native group gains from controllers and retain local values only for unsupported controls."""
    control = NativeGainFakeActuatorControl()
    control.native_gains["kp"].copy_(torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))
    control.native_gains["kd"].copy_(torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]))

    collection = ActuatorCollection(
        {"native": _ideal_cfg([".*"], stiffness=11.0, damping=1.1, effort_limit=100.0)}, control
    )
    group = collection["native"]

    assert "_stiffness" not in group.__dict__
    assert "_damping" not in group.__dict__
    torch.testing.assert_close(group.stiffness, control.native_gains["kp"])
    torch.testing.assert_close(group.damping, control.native_gains["kd"])

    control.native_gains["kp"][1, 2] = 17.0
    control.native_gains["kd"][1, 2] = 1.7
    assert group.stiffness[1, 2] == 17.0
    assert group.damping[1, 2] == 1.7
    for attr, value in (("stiffness", torch.ones((2, 3))), ("damping", torch.ones((2, 3)))):
        with pytest.raises(AttributeError, match=rf"{attr}.*randomize_actuator_gains.*native gain"):
            setattr(group, attr, value)
    assert "_stiffness" not in group.__dict__
    assert "_damping" not in group.__dict__

    control.native_gains.pop("kd")
    unsupported = ActuatorCollection(
        {"native": _ideal_cfg([".*"], stiffness=11.0, damping=1.1, effort_limit=100.0)}, control
    )["native"]
    assert "_stiffness" in unsupported.__dict__
    assert "_damping" in unsupported.__dict__


def test_overlapping_groups_are_rejected():
    with pytest.raises(
        ValueError,
        match="Joint 'joint_1' is assigned to multiple actuator groups: 'first' and 'second'",
    ):
        ActuatorCollection(
            {
                "first": _ideal_cfg(["joint_0", "joint_1"], stiffness=1.0, damping=1.0, effort_limit=10.0),
                "second": _ideal_cfg(["joint_1", "joint_2"], stiffness=2.0, damping=2.0, effort_limit=20.0),
            },
            FakeActuatorControl(),
        )


def test_collection_is_mapping_like_and_read_only():
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)

    assert list(collection.keys()) == ["all"]
    assert collection["all"] is next(iter(collection.values()))
    assert list(collection.items())[0][0] == "all"
    with pytest.raises(TypeError, match="membership is fixed"):
        collection["new"] = collection["all"]
    with pytest.raises(TypeError):
        del collection["all"]

    assert tuple(collection) == ("all",)


def test_custom_singleton_compute_receives_original_selector():
    cfg = _implicit_cfg()
    cfg.class_type = SelectorRecordingActuator
    collection = ActuatorCollection({"all": cfg}, FakeActuatorControl())

    collection.compute()

    assert collection["all"].observed_joint_indices == slice(None)


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
    assert batch.group_names == ("hips", "knees")
    assert batch.actuator is not collection["hips"]
    assert batch.actuator is not collection["knees"]
    assert collection["hips"].joint_names == ["joint_0", "joint_2"]
    torch.testing.assert_close(batch.actuator.stiffness[:, :2], torch.full((2, 2), 10.0))
    torch.testing.assert_close(batch.actuator.stiffness[:, 2:], torch.full((2, 2), 30.0))
    assert collection["hips"].stiffness.untyped_storage().data_ptr() == batch.actuator.stiffness.data_ptr()
    collection["hips"].stiffness.fill_(17.0)
    torch.testing.assert_close(batch.actuator.stiffness[:, :2], torch.full((2, 2), 17.0))


def test_disjoint_implicit_groups_share_one_execution_batch():
    control = FakeActuatorControl(num_envs=1, joint_names=["joint_0", "joint_1", "joint_2", "joint_3"])
    collection = ActuatorCollection(
        {
            "first": ImplicitActuatorCfg(
                joint_names_expr=["joint_0", "joint_2"], stiffness=1.0, damping=1.0, joint_velocity_limit=5.0
            ),
            "second": ImplicitActuatorCfg(
                joint_names_expr=["joint_1", "joint_3"], stiffness=2.0, damping=2.0, joint_velocity_limit=6.0
            ),
        },
        control,
    )

    assert len(collection._execution_batches) == 1
    assert type(collection._execution_batches[0].actuator) is ImplicitActuator
    assert collection._execution_batches[0].group_names == ("first", "second")
    assert ImplicitActuator._EXECUTION_PARAMETER_NAMES == ("velocity_limit",)

    group = collection["first"]
    velocity_limit_snapshot = group.velocity_limit.clone()
    control._current_joint_properties.velocity_limit[:, [0, 2]] = 99.0
    control.joint_stiffness.torch[:, [0, 2]] = torch.tensor([[11.0, 13.0]])
    control.joint_damping.torch[:, [0, 2]] = torch.tensor([[2.0, 3.0]])
    control.joint_effort_limits.torch[:, [0, 2]] = torch.tensor([[7.0, 9.0]])
    collection.command.position.torch[:, [0, 2]] = torch.tensor([[1.0, 2.0]])
    collection.command.velocity.torch[:, [0, 2]] = torch.tensor([[2.0, 3.0]])

    collection.compute()

    expected = torch.tensor([[15.0, 35.0]])
    limit = torch.tensor([[7.0, 9.0]])
    assert torch.equal(group.computed_effort, expected)
    assert torch.equal(group.applied_effort, expected.clamp(-limit, limit))
    assert torch.equal(group.stiffness, torch.tensor([[11.0, 13.0]]))
    assert torch.equal(group.damping, torch.tensor([[2.0, 3.0]]))
    assert torch.equal(group.effort_limit, torch.tensor([[7.0, 9.0]]))
    assert torch.equal(group.velocity_limit, velocity_limit_snapshot)
    with pytest.raises(
        AttributeError,
        match=r"ImplicitActuator.stiffness.*write_joint_stiffness_to_sim_index.*randomize_actuator_gains",
    ):
        group.stiffness = torch.zeros_like(group.stiffness)


def test_lab_executed_explicit_groups_warn_once():
    explicit_cfgs = {
        "ideal": _ideal_cfg(["joint_0"], stiffness=1.0, damping=1.0, effort_limit=10.0),
        "delayed": DelayedPDActuatorCfg(
            joint_names_expr=["joint_1", "joint_2"],
            stiffness=1.0,
            damping=1.0,
            effort_limit=10.0,
            velocity_limit=10.0,
            max_delay=0,
        ),
    }
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        ActuatorCollection(explicit_cfgs, FakeActuatorControl())

    deprecations = [warning for warning in caught_warnings if warning.category is DeprecationWarning]
    assert len(deprecations) == 1
    assert "execution of explicit actuator models is deprecated" in str(deprecations[0].message)


def test_native_executed_explicit_group_does_not_warn():
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        ActuatorCollection(
            {"native": _ideal_cfg([".*"], stiffness=1.0, damping=1.0, effort_limit=10.0)},
            NativeFakeActuatorControl(),
        )

    assert not [warning for warning in caught_warnings if warning.category is DeprecationWarning]


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


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "actuator_cfg",
    [
        ImplicitActuatorCfg(joint_names_expr=["joint_0"], stiffness=2.0, damping=0.0),
        IdealPDActuatorCfg(
            joint_names_expr=["joint_0"],
            stiffness=2.0,
            damping=0.0,
            effort_limit=100.0,
            velocity_limit=10.0,
        ),
    ],
    ids=["implicit", "explicit"],
)
def test_actuator_batch_rebinds_cuda_state_provider_on_request(
    actuator_cfg: ImplicitActuatorCfg | IdealPDActuatorCfg,
):
    control = FakeActuatorControl(num_envs=1, joint_names=["joint_0"], device="cuda:0")
    collection = ActuatorCollection({"all": actuator_cfg}, control)
    collection.command.position.torch.fill_(3.0)

    collection.compute()
    control._joint_pos = ProxyArray(wp.full((1, 1), 2.0, dtype=wp.float32, device=control.device))
    if isinstance(actuator_cfg, ImplicitActuatorCfg):
        collection.command.velocity.torch.fill_(4.0)
        collection.command.effort.torch.fill_(5.0)
        control._joint_vel = ProxyArray(wp.full((1, 1), 1.0, dtype=wp.float32, device=control.device))
        control._joint_stiffness = ProxyArray(wp.full((1, 1), 7.0, dtype=wp.float32, device=control.device))
        control._joint_damping = ProxyArray(wp.full((1, 1), 11.0, dtype=wp.float32, device=control.device))
        control._joint_effort_limits = ProxyArray(wp.full((1, 1), 13.0, dtype=wp.float32, device=control.device))
    collection._rebind_state_inputs()

    collection.compute()

    expected_computed = 45.0 if isinstance(actuator_cfg, ImplicitActuatorCfg) else 2.0
    expected_applied = 13.0 if isinstance(actuator_cfg, ImplicitActuatorCfg) else 2.0
    torch.testing.assert_close(
        collection.computed_effort.torch,
        torch.tensor([[expected_computed]], device=control.device),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        collection.applied_effort.torch,
        torch.tensor([[expected_applied]], device=control.device),
        rtol=0.0,
        atol=0.0,
    )


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
    original_compute = IdealPDActuator.compute

    def counted_compute(self, control_action, joint_pos, joint_vel):
        nonlocal compute_calls
        compute_calls += 1
        return original_compute(self, control_action, joint_pos, joint_vel)

    monkeypatch.setattr(IdealPDActuator, "compute", counted_compute)
    collection.command.position.torch.copy_(torch.arange(12, dtype=torch.float32).reshape(2, 6) + 0.25)
    collection.command.velocity.torch.copy_(torch.arange(12, dtype=torch.float32).reshape(2, 6) * -0.5 - 0.75)
    collection.command.effort.torch.copy_(torch.arange(12, dtype=torch.float32).reshape(2, 6) + 1.5)

    collection.compute()

    assert compute_calls == 1
    batch = collection._execution_batches[0]
    first_hips_output = collection["hips"].computed_effort
    collection.command.position.torch.mul_(-1.25)
    collection.command.velocity.torch.add_(2.75)
    collection.command.effort.torch.sub_(4.5)

    collection.compute()

    assert compute_calls == 2
    assert collection["hips"].computed_effort is first_hips_output
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


def test_stateless_explicit_batch_preserves_tensor_pointers():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {
            "hips": _ideal_cfg(["joint_0", "joint_2"], stiffness=8.0, damping=0.5, effort_limit=12.0),
            "knees": _ideal_cfg(["joint_1", "joint_3"], stiffness=13.0, damping=1.0, effort_limit=18.0),
        },
        control,
    )
    _assign_deterministic_inputs(collection, control)
    batch = collection._execution_batches[0]

    collection.compute()
    pointers = (
        batch.actuator.computed_effort.data_ptr(),
        batch.actuator.applied_effort.data_ptr(),
        batch.control_action.joint_positions.data_ptr(),
        batch.control_action.joint_velocities.data_ptr(),
        batch.control_action.joint_efforts.data_ptr(),
        batch.joint_pos.data_ptr(),
        batch.joint_vel.data_ptr(),
    )
    collection.command.position.torch.mul_(-1.25)
    collection.command.velocity.torch.add_(2.75)
    collection.command.effort.torch.sub_(4.5)
    control.joint_pos.torch.add_(0.125)
    control.joint_vel.torch.sub_(0.25)

    collection.compute()

    assert pointers == (
        batch.actuator.computed_effort.data_ptr(),
        batch.actuator.applied_effort.data_ptr(),
        batch.control_action.joint_positions.data_ptr(),
        batch.control_action.joint_velocities.data_ptr(),
        batch.control_action.joint_efforts.data_ptr(),
        batch.joint_pos.data_ptr(),
        batch.joint_vel.data_ptr(),
    )


def test_stateful_subclasses_and_incompatible_classes_remain_separate():
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
            "ideal_b": _ideal_cfg(["joint_3"], stiffness=3.0, damping=3.0, effort_limit=30.0),
        },
        FakeActuatorControl(joint_names=["joint_0", "joint_1", "joint_2", "joint_3"]),
    )
    assert len(cross_class._execution_batches) == 2


def test_native_execution_bypasses_lab_aggregation(monkeypatch):
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
