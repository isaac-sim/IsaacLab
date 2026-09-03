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

import isaaclab.actuators as actuator_api
from isaaclab.actuators import (
    ActuatorCollection,
    ActuatorControl,
    DCMotor,
    DCMotorCfg,
    DelayedPDActuatorCfg,
    IdealPDActuator,
    IdealPDActuatorCfg,
    ImplicitActuator,
    ImplicitActuatorCfg,
)
from isaaclab.actuators.actuator_control import ArticulationActuatorControl
from isaaclab.actuators.newton import read_group_parameter, write_group_parameter
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
        actuator_effort_limit=effort_limit,
        actuator_velocity_limit=100.0,
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
        actuator_effort_limit=effort_limit,
        actuator_velocity_limit=velocity_limit,
        saturation_effort=saturation_effort,
    )


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
    collection.target_command.position.torch.copy_(
        torch.tensor(
            [
                [1.40, -0.20, -0.75, 2.20],
                [0.15, -1.45, 2.05, -0.65],
            ],
            dtype=torch.float32,
        )
    )
    collection.target_command.velocity.torch.copy_(
        torch.tensor(
            [
                [-3.5, 4.25, 5.75, -6.5],
                [7.0, -8.5, -9.25, 10.75],
            ],
            dtype=torch.float32,
        )
    )
    collection.target_command.effort.torch.copy_(
        torch.tensor(
            [
                [2.25, -3.50, 4.75, -5.25],
                [-6.50, 7.75, -8.25, 9.50],
            ],
            dtype=torch.float32,
        )
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
        self._current_joint_properties = {
            "stiffness": self._joint_stiffness.torch,
            "damping": self._joint_damping.torch,
            "armature": zeros.clone(),
            "friction": zeros.clone(),
            "dynamic_friction": zeros.clone(),
            "viscous_friction": zeros.clone(),
            "joint_effort_limit": self._joint_effort_limits.torch,
            "joint_velocity_limit": ones * 10.0,
        }
        self.written_properties: list[tuple[dict[str, torch.Tensor], torch.Tensor | slice, bool, bool]] = []
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

    def get_default_joint_properties(self, joint_ids: torch.Tensor | wp.array | slice) -> dict[str, torch.Tensor]:
        if isinstance(joint_ids, wp.array):
            joint_ids = wp.to_torch(joint_ids).to(device=self.device, dtype=torch.long)
        properties = self._current_joint_properties
        return {
            "stiffness": self.joint_stiffness.torch[:, joint_ids].clone(),
            "damping": self.joint_damping.torch[:, joint_ids].clone(),
            "armature": properties["armature"][:, joint_ids].clone(),
            "friction": properties["friction"][:, joint_ids].clone(),
            "dynamic_friction": properties["dynamic_friction"][:, joint_ids].clone(),
            "viscous_friction": properties["viscous_friction"][:, joint_ids].clone(),
            "joint_effort_limit": self.joint_effort_limits.torch[:, joint_ids].clone(),
            "joint_velocity_limit": properties["joint_velocity_limit"][:, joint_ids].clone(),
        }

    def write_resolved_joint_properties(
        self,
        properties: dict[str, torch.Tensor],
        joint_ids: torch.Tensor | slice,
        *,
        implicit: bool,
        native_managed: bool,
    ) -> None:
        self.written_properties.append((properties, joint_ids, implicit, native_managed))
        self.joint_effort_limits.torch[:, joint_ids] = properties["joint_effort_limit"]
        if implicit and not native_managed:
            self.joint_stiffness.torch[:, joint_ids] = properties["stiffness"]
            self.joint_damping.torch[:, joint_ids] = properties["damping"]
        else:
            self.joint_stiffness.torch[:, joint_ids] = 0.0
            self.joint_damping.torch[:, joint_ids] = 0.0

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


class _FakeNewtonActuator:
    """Newton-actuator stand-in; a plain class so the view's mapping cache can hash it."""

    def __init__(self, controller, indices):
        self.controller = controller
        self.delay = None
        self.clamping = []
        self.indices = indices


class NativeFakeActuatorControl(FakeActuatorControl):
    """Control object that handles actuator execution natively."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        size = self.num_instances * self.num_joints
        self.newton_actuator = _FakeNewtonActuator(
            controller=SimpleNamespace(),
            indices=wp.array(list(range(size)), dtype=wp.uint32, device=self.device),
        )

    @property
    def native_actuator_path_active(self) -> bool:
        return True

    def prepare_native_actuators(self, collection, actuator_cfgs) -> set[str]:
        self.prepared_actuator_cfgs = actuator_cfgs
        return set(actuator_cfgs)

    def compute_native_actuators(self, collection: ActuatorCollection, dt: float) -> bool:
        return True

    def finalize_native_actuators(self, collection):
        from isaaclab.actuators.newton.adapter import LightArticulationView, NewtonActuatorSelection

        return NewtonActuatorSelection(
            view=LightArticulationView(self.num_instances, self.num_joints, self.device),
            actuators=[self.newton_actuator],
        )


class NativeGainFakeActuatorControl(NativeFakeActuatorControl):
    """Native control backed by one Newton-shaped actuator with controller-owned storage."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        size = self.num_instances * self.num_joints
        self.newton_actuator.controller = SimpleNamespace(
            kp=wp.zeros(size, dtype=wp.float32, device=self.device),
            kd=wp.zeros(size, dtype=wp.float32, device=self.device),
        )

    @property
    def native_gains(self) -> dict[str, torch.Tensor]:
        """Live torch views over the controller-owned gain storage."""
        shape = (self.num_instances, self.num_joints)
        controller = self.newton_actuator.controller
        return {
            attr: wp.to_torch(getattr(controller, attr)).view(shape)
            for attr in ("kp", "kd")
            if hasattr(controller, attr)
        }


@pytest.mark.parametrize(
    ("cfg_factory", "with_joint_limit", "expected_joint_limit"),
    [
        (_ideal_pd_cfg, True, 34.0),
        (_implicit_cfg, True, 34.0),
        (_implicit_cfg, False, 12.0),
    ],
)
def test_deprecated_effort_limit_forwards_by_actuator_type(cfg_factory, with_joint_limit, expected_joint_limit):
    """``effort_limit`` resolves to the rated ``actuator_effort_limit`` for every actuator type.

    For implicit groups without a separate solver clamp, the rated value also reaches
    ``joint_effort_limit`` for backward compatibility.
    """
    cfg = (
        cfg_factory(effort_limit=12.0, joint_effort_limit=34.0) if with_joint_limit else cfg_factory(effort_limit=12.0)
    )
    # Lab execution keeps the group as an inspectable Lab model instance; the explicit
    # variant additionally warns about deprecated Lab execution, which pytest.warns tolerates.
    control = FakeActuatorControl()

    with pytest.warns(DeprecationWarning, match="actuator_effort_limit"):
        collection = ActuatorCollection({"motor": cfg}, control)

    # Runtime alias-property behavior (group.effort_limit get/set warnings) is covered by
    # the per-actuator suites in test_implicit_actuator.py and test_ideal_pd_actuator.py.
    group = collection["motor"]
    assert group.cfg.actuator_effort_limit == 12.0
    assert group.cfg.joint_effort_limit == expected_joint_limit
    torch.testing.assert_close(group.actuator_effort_limit, torch.full((2, 3), 12.0))
    if isinstance(group, ImplicitActuator):
        # Lab explicit models hold no solver-limit view; the solver clamp lives on articulation data.
        torch.testing.assert_close(group.joint_effort_limit, torch.full((2, 3), expected_joint_limit))


def test_constructor_resolves_deprecated_velocity_limit_alias():
    cfg = _ideal_pd_cfg(velocity_limit_sim=34.0)

    with pytest.warns(DeprecationWarning, match="joint_velocity_limit"):
        actuator = IdealPDActuator(
            cfg,
            joint_names=["joint_0", "joint_1", "joint_2"],
            joint_ids=slice(None),
            num_envs=2,
            device="cpu",
        )

    assert actuator.cfg.joint_velocity_limit == 34.0
    assert actuator.cfg.velocity_limit_sim is None


def test_implicit_actuator_separate_rated_effort_limit_is_honored():
    """A rated limit distinct from the solver clamp stays on the implicit actuator."""
    cfg = _implicit_cfg(actuator_effort_limit=12.0, joint_effort_limit=34.0)

    collection = ActuatorCollection({"motor": cfg}, FakeActuatorControl())

    group = collection["motor"]
    torch.testing.assert_close(group.actuator_effort_limit, torch.full((2, 3), 12.0))
    torch.testing.assert_close(group.joint_effort_limit, torch.full((2, 3), 34.0))


@pytest.mark.parametrize(
    ("cfg", "actuator_type", "canonical_name"),
    [
        (_ideal_pd_cfg(), IdealPDActuator, "actuator_effort_limit"),
        (_implicit_cfg(), ImplicitActuator, "joint_effort_limit"),
    ],
)
def test_constructor_effort_limit_alias_conflicts_with_explicit_infinity(cfg, actuator_type, canonical_name):
    constructor_kwargs = {
        "joint_names": ["joint_0", "joint_1", "joint_2"],
        "joint_ids": slice(None),
        "num_envs": 2,
        "device": "cpu",
        "stiffness": 0.0,
        "damping": 0.0,
    }

    with pytest.warns(DeprecationWarning, match=canonical_name):
        with pytest.raises(ValueError, match=rf"conflicting {canonical_name}.*effort_limit"):
            actuator_type(
                cfg,
                **constructor_kwargs,
                **{canonical_name: torch.inf, "effort_limit": 12.0},
            )

    with pytest.warns(DeprecationWarning, match=canonical_name):
        actuator = actuator_type(
            cfg.copy(),
            **constructor_kwargs,
            **{canonical_name: torch.full((2, 3), 12.0), "effort_limit": 12.0},
        )
    torch.testing.assert_close(getattr(actuator, canonical_name), torch.full((2, 3), 12.0))


def test_equivalent_limit_aliases_prefer_canonical_values():
    # Rows with a canonical value assert the canonical value wins over an equivalent alias.
    # Rows with canonical_value=None assert the deprecated *_sim alias forwards onto the
    # prepared copy when the canonical field is unset, without mutating the user's cfg.
    scenarios = (
        ("joint_effort_limit", "effort_limit_sim", 12.0, {"joint_.*": 12.0}, _ideal_pd_cfg, NativeFakeActuatorControl),
        (
            "joint_velocity_limit",
            "velocity_limit_sim",
            34.0,
            {"joint_.*": 34.0},
            _ideal_pd_cfg,
            NativeFakeActuatorControl,
        ),
        ("actuator_effort_limit", "effort_limit", {"joint_.*": 12.0}, 12.0, _ideal_pd_cfg, NativeFakeActuatorControl),
        ("joint_effort_limit", "effort_limit", {"joint_.*": 12.0}, 12.0, _implicit_cfg, FakeActuatorControl),
        ("joint_effort_limit", "effort_limit_sim", None, 12.0, _ideal_pd_cfg, NativeFakeActuatorControl),
        ("joint_velocity_limit", "velocity_limit_sim", None, 34.0, _ideal_pd_cfg, NativeFakeActuatorControl),
    )

    for canonical_name, alias_name, canonical_value, alias_value, cfg_factory, control_factory in scenarios:
        cfg_kwargs = {alias_name: alias_value}
        if canonical_value is not None:
            cfg_kwargs[canonical_name] = canonical_value
        cfg = cfg_factory(**cfg_kwargs)
        control = control_factory()
        with pytest.warns(DeprecationWarning):
            collection = ActuatorCollection({"motor": cfg}, control)

        prepared_cfgs = getattr(control, "prepared_actuator_cfgs", None)
        resolved_cfg = prepared_cfgs["motor"] if prepared_cfgs is not None else collection["motor"].cfg
        expected_value = canonical_value if canonical_value is not None else alias_value
        assert getattr(resolved_cfg, canonical_name) == expected_value
        # the user's cfg is never mutated: the canonical field keeps its original value.
        assert getattr(cfg, canonical_name) == canonical_value
        assert getattr(cfg, alias_name) == alias_value
        if canonical_value is None:
            assert resolved_cfg is not cfg


def test_conflicting_limit_aliases_raise():
    scenarios = (
        ("joint_effort_limit", "effort_limit_sim", _ideal_pd_cfg, NativeFakeActuatorControl),
        ("joint_velocity_limit", "velocity_limit_sim", _ideal_pd_cfg, NativeFakeActuatorControl),
        ("actuator_effort_limit", "effort_limit", _ideal_pd_cfg, NativeFakeActuatorControl),
        ("actuator_effort_limit", "effort_limit", _implicit_cfg, FakeActuatorControl),
    )

    for canonical_name, alias_name, cfg_factory, control_factory in scenarios:
        cfg = cfg_factory(
            **{
                canonical_name: {"joint_0": 12.0, "joint_1": 10.0, "joint_2": 8.0},
                alias_name: {"joint_0": 12.0, "joint_1": 11.0, "joint_2": 8.0},
            }
        )
        with (
            pytest.warns(DeprecationWarning),
            pytest.raises(ValueError, match=rf"motor.*{canonical_name}.*{alias_name}"),
        ):
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
    torch.testing.assert_close(defaults["dynamic_friction"], torch.zeros(2, 3))
    torch.testing.assert_close(defaults["viscous_friction"], torch.zeros(2, 3))

    properties = {
        "joint_effort_limit": torch.ones((2, 3)),
        "joint_velocity_limit": torch.ones((2, 3)) * 2.0,
        "armature": torch.ones((2, 3)) * 3.0,
        "friction": torch.ones((2, 3)) * 4.0,
        "dynamic_friction": torch.zeros((2, 3)),
        "viscous_friction": torch.zeros((2, 3)),
        "stiffness": torch.ones((2, 3)) * 5.0,
        "damping": torch.ones((2, 3)) * 6.0,
    }

    control.write_resolved_joint_properties(properties, slice(None), implicit=False, native_managed=False)

    # the write order is not a contract: compare the set of property writes order-insensitively.
    assert sorted(name for name, _ in articulation.calls) == sorted(
        ["effort_limit", "velocity_limit", "armature", "friction", "stiffness", "damping"]
    )
    calls_by_name = dict(articulation.calls)
    assert calls_by_name["stiffness"]["stiffness"] == 0.0
    assert calls_by_name["damping"]["damping"] == 0.0
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
    torch.testing.assert_close(defaults["armature"], expected)


def test_native_explicit_groups_zero_solver_drives_and_build_no_lab_model(monkeypatch):
    """Zero the solver drives of a native explicit group and expose the Newton actuator."""
    from isaaclab.actuators.newton.adapter import LightArticulationView, NewtonActuatorSelection

    articulation = FakeArticulation()
    articulation.data.joint_stiffness.torch.fill_(17.0)
    articulation.data.joint_damping.torch.fill_(3.0)
    control = FakeArticulationActuatorControl(articulation)
    newton_actuator = _FakeNewtonActuator(
        controller=SimpleNamespace(),
        indices=wp.array(
            list(range(articulation.num_instances * articulation.num_joints)),
            dtype=wp.uint32,
            device=articulation.device,
        ),
    )
    monkeypatch.setattr(control, "prepare_native_actuators", lambda collection, cfgs: set(cfgs))
    monkeypatch.setattr(control, "_native_actuator_path_active", True)
    monkeypatch.setattr(
        control,
        "finalize_native_actuators",
        lambda collection: NewtonActuatorSelection(
            view=LightArticulationView(articulation.num_instances, articulation.num_joints, articulation.device),
            actuators=[newton_actuator],
        ),
    )

    collection = ActuatorCollection(
        {
            "explicit": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
                actuator_effort_limit=100.0,
                actuator_velocity_limit=10.0,
            )
        },
        control,
    )

    # No Lab model is built for the Newton-executed group: the owner is exposed directly.
    assert collection["explicit"] is newton_actuator
    torch.testing.assert_close(articulation.data.joint_stiffness.torch, torch.zeros((2, 3)))
    torch.testing.assert_close(articulation.data.joint_damping.torch, torch.zeros((2, 3)))
    assert articulation.calls[-2][1]["stiffness"] == 0.0
    assert articulation.calls[-1][1]["damping"] == 0.0


def test_native_group_parameters_route_through_the_collection_door():
    """Read and write native group parameters through the collection's single parameter door."""
    control = NativeGainFakeActuatorControl()
    control.native_gains["kp"].copy_(torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))
    control.native_gains["kd"].copy_(torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]))

    collection = ActuatorCollection(
        {"native": _ideal_cfg([".*"], stiffness=11.0, damping=1.1, effort_limit=100.0)}, control
    )

    # Reads are live projections of the controller-owned storage.
    torch.testing.assert_close(
        read_group_parameter(collection, "native", "controller", "kp"), control.native_gains["kp"]
    )
    control.native_gains["kd"][1, 2] = 1.7
    assert read_group_parameter(collection, "native", "controller", "kd")[1, 2] == 1.7

    # The group's mapping entry is the owning Newton actuator: no stale Lab mirrors exist,
    # and direct modification of the controller storage is observed by the door reads.
    group = collection["native"]
    assert group is control.newton_actuator
    wp.to_torch(group.controller.kp).view(2, 3)[0, 0] = 21.0
    assert read_group_parameter(collection, "native", "controller", "kp")[0, 0] == 21.0
    wp.to_torch(group.controller.kp).view(2, 3)[0, 0] = 2.0

    # The single write path patches the controller storage in place over an env/joint selection.
    write_group_parameter(
        collection,
        "native",
        "controller",
        "kp",
        values=torch.tensor([[42.0]]),
        env_ids=torch.tensor([0]),
        joint_ids=torch.tensor([1]),
    )
    torch.testing.assert_close(control.native_gains["kp"], torch.tensor([[2.0, 42.0, 4.0], [5.0, 6.0, 7.0]]))
    write_group_parameter(collection, "native", "controller", "kd", values=torch.full((2, 3), 0.9))
    torch.testing.assert_close(control.native_gains["kd"], torch.full((2, 3), 0.9))

    with pytest.raises(ValueError, match=r"No Newton actuator exposes parameter \('controller', 'kq'\)"):
        write_group_parameter(collection, "native", "controller", "kq", values=torch.zeros((2, 3)))
    with pytest.raises(ValueError, match=r"Unknown actuator component 'gains'"):
        read_group_parameter(collection, "native", "gains", "kp")

    # A parameter the controllers do not expose raises instead of falling back to stale values.
    unsupported_control = NativeGainFakeActuatorControl()
    del unsupported_control.newton_actuator.controller.kd
    unsupported = ActuatorCollection(
        {"native": _ideal_cfg([".*"], stiffness=11.0, damping=1.1, effort_limit=100.0)}, unsupported_control
    )
    torch.testing.assert_close(read_group_parameter(unsupported, "native", "controller", "kp"), torch.zeros((2, 3)))
    with pytest.raises(ValueError, match=r"No Newton actuator exposes parameter \('controller', 'kd'\)"):
        read_group_parameter(unsupported, "native", "controller", "kd")

    # Groups that are not Newton-managed keep plain construction gains, and the door rejects them.
    plain = ActuatorCollection(
        {"plain": _ideal_cfg([".*"], stiffness=11.0, damping=1.1, effort_limit=100.0)}, FakeActuatorControl()
    )
    torch.testing.assert_close(plain["plain"].stiffness, torch.full((2, 3), 11.0))
    with pytest.raises(ValueError, match=r"'plain' is not executed by Newton actuators"):
        read_group_parameter(plain, "plain", "controller", "kp")


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
    assert hasattr(actuator_api, "ActuatorTargetCommand")
    assert hasattr(actuator_api, "ActuatorOutputCommand")

    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)
    assert isinstance(collection.target_command, actuator_api.ActuatorTargetCommand)
    assert isinstance(collection.output_command, actuator_api.ActuatorOutputCommand)

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


def test_multi_group_explicit_outputs_match_pd_formula():
    joint_names = [f"joint_{index}" for index in range(4)]
    control = FakeActuatorControl(joint_names=joint_names)
    collection = ActuatorCollection(
        {
            "hips": _ideal_cfg(["joint_0", "joint_2"], stiffness=12.0, damping=1.5, effort_limit=18.0),
            "knees": _ideal_cfg(["joint_1", "joint_3"], stiffness=27.0, damping=2.25, effort_limit=31.0),
        },
        control,
    )
    _assign_deterministic_inputs(collection, control)

    collection.compute()

    stiffness = torch.tensor([[12.0, 27.0, 12.0, 27.0]])
    damping = torch.tensor([[1.5, 2.25, 1.5, 2.25]])
    limit = torch.tensor([[18.0, 31.0, 18.0, 31.0]])
    expected_computed = (
        stiffness * (collection.target_command.position.torch - control.joint_pos.torch)
        + damping * (collection.target_command.velocity.torch - control.joint_vel.torch)
        + collection.target_command.effort.torch
    )
    expected_applied = expected_computed.clamp(-limit, limit)
    torch.testing.assert_close(collection.computed_effort.torch, expected_computed, rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection.applied_effort.torch, expected_applied, rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection["hips"].computed_effort, expected_computed[:, [0, 2]], rtol=0.0, atol=0.0)
    torch.testing.assert_close(collection["knees"].applied_effort, expected_applied[:, [1, 3]], rtol=0.0, atol=0.0)


def test_disjoint_implicit_groups_share_one_execution_batch():
    assert "is_implicit_model" in ImplicitActuator.__dict__.get("__annotations__", {})
    assert ImplicitActuator.__dict__["is_implicit_model"] is True

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

    assert collection._execution_actuators == []
    executor = collection._implicit_executor
    assert executor is not None
    assert type(executor.actuator) is ImplicitActuator
    assert executor.actuator is not collection["first"]
    assert executor.group_names == ("first", "second")

    group = collection["first"]
    velocity_limit_snapshot = group.actuator_velocity_limit.clone()
    control._current_joint_properties["joint_velocity_limit"][:, [0, 2]] = 99.0
    control.joint_stiffness.torch[:, [0, 2]] = torch.tensor([[11.0, 13.0]])
    control.joint_damping.torch[:, [0, 2]] = torch.tensor([[2.0, 3.0]])
    control.joint_effort_limits.torch[:, [0, 2]] = torch.tensor([[7.0, 9.0]])
    collection.target_command.position.torch[:, [0, 2]] = torch.tensor([[1.0, 2.0]])
    collection.target_command.velocity.torch[:, [0, 2]] = torch.tensor([[2.0, 3.0]])

    collection.compute()

    expected = torch.tensor([[15.0, 35.0]])
    limit = torch.tensor([[7.0, 9.0]])
    assert torch.equal(group.computed_effort, expected)
    assert torch.equal(group.applied_effort, expected.clamp(-limit, limit))
    assert torch.equal(group.stiffness, torch.tensor([[11.0, 13.0]]))
    assert torch.equal(group.damping, torch.tensor([[2.0, 3.0]]))
    assert torch.equal(group.joint_effort_limit, torch.tensor([[7.0, 9.0]]))
    assert torch.equal(group.actuator_velocity_limit, velocity_limit_snapshot)
    stiffness_before = group.stiffness.clone()
    with pytest.warns(
        UserWarning,
        match=r"ImplicitActuator.stiffness.*write_joint_stiffness_to_sim_index.*randomize_actuator_gains",
    ):
        group.stiffness = torch.zeros_like(group.stiffness)
    assert torch.equal(group.stiffness, stiffness_before)


def test_lab_executed_explicit_groups_warn_once():
    explicit_cfgs = {
        "ideal": _ideal_cfg(["joint_0"], stiffness=1.0, damping=1.0, effort_limit=10.0),
        "delayed": DelayedPDActuatorCfg(
            joint_names_expr=["joint_1", "joint_2"],
            stiffness=1.0,
            damping=1.0,
            actuator_effort_limit=10.0,
            actuator_velocity_limit=10.0,
            max_delay=0,
        ),
    }
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        ActuatorCollection(explicit_cfgs, FakeActuatorControl())

    deprecations = [warning for warning in caught_warnings if warning.category is DeprecationWarning]
    assert len(deprecations) == 1
    assert "execution of explicit actuator models is deprecated" in str(deprecations[0].message)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "actuator_cfg",
    [
        ImplicitActuatorCfg(joint_names_expr=["joint_0"], stiffness=2.0, damping=0.0),
        IdealPDActuatorCfg(
            joint_names_expr=["joint_0"],
            stiffness=2.0,
            damping=0.0,
            actuator_effort_limit=100.0,
            actuator_velocity_limit=10.0,
        ),
    ],
    ids=["implicit", "explicit"],
)
def test_actuator_batch_rebinds_cuda_state_provider_on_request(
    actuator_cfg: ImplicitActuatorCfg | IdealPDActuatorCfg,
):
    control = FakeActuatorControl(num_envs=1, joint_names=["joint_0"], device="cuda:0")
    collection = ActuatorCollection({"all": actuator_cfg}, control)
    collection.target_command.position.torch.fill_(3.0)

    collection.compute()
    control._joint_pos = ProxyArray(wp.full((1, 1), 2.0, dtype=wp.float32, device=control.device))
    if isinstance(actuator_cfg, ImplicitActuatorCfg):
        collection.target_command.velocity.torch.fill_(4.0)
        collection.target_command.effort.torch.fill_(5.0)
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


def test_partial_coverage_explicit_group_reads_fresh_commands_each_compute():
    control = FakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    collection = ActuatorCollection(
        {"hips": _ideal_cfg(["joint_0", "joint_2"], stiffness=10.0, damping=0.0, effort_limit=1000.0)},
        control,
    )
    collection.target_command.position.torch[:, [0, 2]] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    collection.compute()

    expected_first = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    torch.testing.assert_close(collection.computed_effort.torch[:, [0, 2]].cpu(), expected_first, rtol=0.0, atol=0.0)

    collection.target_command.position.torch.mul_(2.0)
    collection.compute()

    torch.testing.assert_close(
        collection.computed_effort.torch[:, [0, 2]].cpu(), expected_first * 2.0, rtol=0.0, atol=0.0
    )


def test_native_execution_bypasses_lab_aggregation(monkeypatch):
    control = NativeFakeActuatorControl(joint_names=[f"joint_{index}" for index in range(4)])
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
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

    assert not [warning for warning in caught_warnings if warning.category is DeprecationWarning]

    assert collection._implicit_executor is None
    assert collection._execution_actuators == []

    def fail_compute(*args, **kwargs):
        raise AssertionError("Lab actuator execution must be bypassed")

    monkeypatch.setattr(DCMotor, "compute", fail_compute)
    collection.compute()


def test_collection_accepts_cached_proxy_joint_indices():
    control = ProxyFinderActuatorControl()
    collection = ActuatorCollection({"outer": _implicit_cfg()}, control)

    torch.testing.assert_close(collection["outer"].joint_indices, torch.tensor([0, 2], dtype=torch.int32))


@pytest.mark.parametrize("command_name", ["position", "velocity", "effort"])
def test_write_command_index_supports_selectors_and_submission(command_name):
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)
    setter = getattr(collection.target_command, f"set_{command_name}_index")
    command_buffer = getattr(collection.target_command, command_name)
    value = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    setter(value=value, env_ids=[1], joint_ids=[0, 2])

    expected = torch.zeros(2, 3)
    expected[1, 0] = 1.0
    expected[1, 2] = 2.0
    torch.testing.assert_close(command_buffer.torch.cpu(), expected)
    assert control.staged_commands == [command_name]

    command_buffer.torch.zero_()
    value = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    env_ids = torch.tensor([1], dtype=torch.int64)
    joint_ids = wp.array([0, 2], dtype=wp.int64, device="cpu")

    setter(value=value, env_ids=env_ids, joint_ids=joint_ids)

    expected = torch.zeros(2, 3)
    expected[1, 0] = 3.0
    expected[1, 2] = 4.0
    torch.testing.assert_close(command_buffer.torch.cpu(), expected)

    collection.compute()
    collection.submit_commands()

    torch.testing.assert_close(getattr(collection.output_command, command_name).torch.cpu(), expected)
    assert control.submitted


@pytest.mark.parametrize("command_name", ["position", "velocity", "effort"])
def test_write_command_mask_uses_full_sized_value(command_name):
    control = FakeActuatorControl()
    collection = ActuatorCollection({"all": _implicit_cfg()}, control)
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    env_mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    joint_mask = wp.array([False, True, True], dtype=wp.bool, device="cpu")

    getattr(collection.target_command, f"set_{command_name}_mask")(
        value=value, env_mask=env_mask, joint_mask=joint_mask
    )

    expected = torch.zeros(2, 3)
    expected[0, 1:] = value[0, 1:]
    torch.testing.assert_close(getattr(collection.target_command, command_name).torch.cpu(), expected)
    assert control.staged_commands == [command_name]

    with pytest.raises(TypeError, match="wp.bool"):
        getattr(collection.target_command, f"set_{command_name}_mask")(
            value=value, env_mask=wp.array([1, 0], dtype=wp.int32, device="cpu"), joint_mask=joint_mask
        )
