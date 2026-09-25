# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none


"""Real-backend tests for the OVPhysX Articulation.

Mirrors :mod:`isaaclab_physx.test.assets.test_articulation` 1-to-1: same set
of test functions, names, parametrizations, and assertions.

OVPhysX runs kitless under ``./scripts/run_ovphysx.sh`` so there is no
``AppLauncher`` boot — :class:`~isaaclab.sim.SimulationContext` is driven
directly via ``build_simulation_context(sim_cfg=SimulationCfg(physics=OvPhysxCfg(), ...))``
which works because :func:`isaaclab.app.has_kit` returns False in this
environment.

PhysX-specific ``cube_object.root_view.set_X(...)`` / ``get_X(...)`` calls are
adapted to OVPhysX by going through
:attr:`~isaaclab_ov.assets.Articulation.root_view`, an
:class:`~isaaclab_ov.sim.views.OvPhysxView` over the per-tensor-type bindings
(``root_view.get_attribute(tensor_type)`` /
:meth:`~isaaclab_ov.assets.Articulation._get_binding`), and the public setters
(:meth:`set_masses_index`, :meth:`set_coms_index`, :meth:`set_inertias_index`).
Reads use the data-class properties (``cube_object.data.body_mass``,
``body_inertia``, ``body_com_pose_b``).

Process-global device lock
--------------------------

The OVPhysX runtime fixes device mode (CPU vs GPU) when the process creates
its first ``ovphysx.PhysX`` instance and cannot switch it without a process
restart. :class:`~isaaclab_ov.physics.OvPhysxManager` tracks
this on ``_locked_device`` and raises :exc:`RuntimeError` if a later
:class:`SimulationContext` requests a different device.  The
``_ovphysx_skip_other_device`` autouse fixture below preempts that error in
parametrized tests by ``pytest.skip``-ing on the unlocked device, so the
session finishes cleanly when only one device is exercised.

CI note
-------
Because the lock is process-global, full coverage requires **two separate
``./scripts/run_ovphysx.sh -m pytest`` invocations** -- once with ``-k 'cpu'``
and once with ``-k 'cuda:0'``. Until the wheel exposes a way to reset Carbonite
device state, this is the supported pattern.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import warp as wp

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.test.utils import test_devices
from isaaclab.test.utils.articulation_ordering import (
    ANYMAL_C_PHYSX_JOINT_NAMES,
    BRANCHING_MJWARP_BODY_NAMES,
    BRANCHING_MJWARP_JOINT_NAMES,
    BRANCHING_PHYSX_BODY_NAMES,
    BRANCHING_PHYSX_JOINT_NAMES,
    PANDA_ROOT_PRESERVING_REVERSED_BODY_NAMES,
)

# The OVPhysX runtime wheel is optional. Skip gracefully when it is not installed;
# CI jobs that need OVPhysX coverage install it explicitly.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets import Articulation  # noqa: E402
from isaaclab_ov.assets.articulation.actuator_control import OvPhysxActuatorControl  # noqa: E402
from isaaclab_ov.assets.articulation.articulation_data import ArticulationData  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402
from isaaclab_physx.sim.schemas import PhysxJointCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab.utils.string as string_utils  # noqa: E402
from isaaclab.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, get_articulation_name_ordering  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.warp.launch_cache import _WarpLaunchCache  # noqa: E402

##
# Pre-defined configs
##
from isaaclab_assets import ANYMAL_C_CFG, CARTPOLE_CFG, FRANKA_PANDA_CFG, SHADOW_HAND_PHYSX_CFG  # isort:skip

wp.init()

pytestmark = pytest.mark.device_split


def test_prepare_native_actuators_leaves_implicit_only_articulation_on_standard_path(monkeypatch):
    """Keep implicit-only articulations on the unchanged solver-drive path."""
    from isaaclab_ov.assets.articulation import actuator_control

    runtime_prepare_calls = []
    runtime = SimpleNamespace(
        prepare=lambda *args, **kwargs: runtime_prepare_calls.append(True), wrapper=None, adapter=None
    )
    articulation = SimpleNamespace(
        _sim_cfg=SimpleNamespace(use_newton_actuators=True),
        cfg=SimpleNamespace(prim_path="/World/Robot"),
    )
    monkeypatch.setattr(actuator_control, "PhysxActuatorRuntime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(actuator_control, "find_first_matching_prim", lambda _: None)

    control = OvPhysxActuatorControl(articulation)
    native_groups = control.prepare_native_actuators(
        collection=None,
        actuator_cfgs={"implicit": ImplicitActuatorCfg(joint_names_expr=["joint"], stiffness=10.0, damping=1.0)},
    )

    assert native_groups == set()
    assert not control.native_actuator_path_active
    assert not articulation._has_newton_actuators
    assert runtime_prepare_calls == []


def test_cached_read_launches_reset_on_ordering_and_invalidation():
    """Ordering installation and simulation invalidation should discard recorded reads."""

    class MinimalData(ArticulationData):
        def __dir__(self):
            return []

    class Buffer:
        timestamp = 1.0

    data = MinimalData.__new__(MinimalData)
    read_launch_cache = Mock()
    data._read_launch_cache = read_launch_cache
    data._configure_ordering_buffers = lambda: None
    data._make_jacobian_body_user_to_backend = lambda: object()
    data.joint_ordering = None
    data._body_com_jacobian_w = Buffer()
    data._mass_matrix = Buffer()
    data._gravity_compensation_forces = Buffer()

    data._apply_ordering_maps_after_resolve()

    read_launch_cache.clear.assert_called_once_with()
    assert data._body_com_jacobian_w.timestamp == -1.0
    assert data._mass_matrix.timestamp == -1.0
    assert data._gravity_compensation_forces.timestamp == -1.0

    data._is_primed = True
    data._sim_timestamp = 1.0
    data._invalidate_initialize_callback(None)

    assert read_launch_cache.clear.call_count == 2
    assert data._is_primed is False
    assert data._sim_timestamp == 0.0


def test_static_property_reads_are_not_invalidated_by_simulation_steps():
    """Joint properties and body mass/inertia should be read once per invalidation, not once per step.

    On OVPhysX these are blocking CPU-only binding reads whose cost scales with the number of
    environments, so re-reading them every physics step made per-step consumers (such as the
    native actuator telemetry sync) host-bound. State buffers must still refresh every step.
    """

    class Buffer:
        def __init__(self, shape):
            self.data = wp.zeros(shape, dtype=wp.float32, device="cpu")
            self.timestamp = -1.0

    data = ArticulationData.__new__(ArticulationData)
    data.device = "cpu"
    data.num_instances = 1
    data.num_joints = 2
    data._sim_timestamp = 1.0
    data.body_ordering = None
    data._get_binding = lambda tensor_type: object()
    reads: list[int] = []
    data._binding_read = lambda tensor_type, dst: reads.append(tensor_type)

    # Joint properties: one read across several steps, one more after explicit invalidation
    # (simulation reinitialization).
    data.joint_ordering = None
    stiffness = Buffer((1, 2))
    for _ in range(3):
        data._sim_timestamp += 1.0
        data._read_joint_property_binding(TT.DOF_STIFFNESS, stiffness, None)
    assert reads.count(TT.DOF_STIFFNESS) == 1
    stiffness.timestamp = -1.0
    data._read_joint_property_binding(TT.DOF_STIFFNESS, stiffness, None)
    assert reads.count(TT.DOF_STIFFNESS) == 2

    # Body properties behave the same; body state buffers still refresh every step.
    mass, link_pose = Buffer((1, 2)), Buffer((1, 2))
    for _ in range(3):
        data._sim_timestamp += 1.0
        data._refresh_reordered_body_buffer(mass, None, TT.BODY_MASS, static=True)
        data._refresh_reordered_body_buffer(link_pose, None, TT.LINK_POSE)
    assert reads.count(TT.BODY_MASS) == 1
    assert reads.count(TT.LINK_POSE) == 3
    mass.timestamp = -1.0
    data._refresh_reordered_body_buffer(mass, None, TT.BODY_MASS, static=True)
    assert reads.count(TT.BODY_MASS) == 2

    # Under a non-identity joint ordering the property is gathered once, then served from cache.
    data.joint_ordering = SimpleNamespace(user_to_backend=wp.array([1, 0], dtype=wp.int32, device="cpu"))
    data._read_launch_cache = _WarpLaunchCache("cpu")
    user_buffer, backend_buffer = Buffer((1, 2)), Buffer((1, 2))
    data._binding_read = lambda tensor_type, dst: (reads.append(tensor_type), dst.assign([[1.0, 2.0]]))
    for _ in range(3):
        data._sim_timestamp += 1.0
        data._read_joint_property_binding(TT.DOF_DAMPING, user_buffer, backend_buffer)
    assert reads.count(TT.DOF_DAMPING) == 1
    torch.testing.assert_close(wp.to_torch(user_buffer.data), torch.tensor([[2.0, 1.0]]))


def _read_binding_to_torch(articulation: Articulation, tensor_type: int, device: str | torch.device) -> torch.Tensor:
    """Read an OVPhysX attribute into a torch tensor on *device*.

    Test-side adapter for the verbatim PhysX mirror.  PhysX cross-checks the
    data class against the simulation via ``articulation.root_view.get_X()``
    accessors; on OVPhysX we go through the equivalent
    :meth:`~isaaclab_ov.sim.views.OvPhysxView.get_attribute`, which returns a
    freshly allocated ``float32`` array on the attribute's native device (CPU for
    CPU-only property types), then move the result to *device*.
    """
    arr = articulation.root_view.get_attribute(tensor_type)
    return wp.to_torch(arr).to(device)


# Session-locked device.  Set on the first parametrized test that runs and
# never reassigned -- ovphysx's process-global device lock means subsequent
# tests on the other device must skip.
_LOCKED_DEVICE: list[str | None] = [None]


@pytest.fixture(autouse=True)
def _ovphysx_skip_other_device(request):
    """Skip tests whose ``device`` parameter mismatches the session-locked device.

    The OVPhysX runtime locks process-global device mode when the process
    creates its first ``ovphysx.PhysX`` instance, so any test parametrized to a
    different device after the first ``sim.reset()`` would hit the manager's
    :exc:`RuntimeError`. We detect the locked device on the
    first encounter and skip subsequent tests on the other device with a clear
    message so the run finishes cleanly rather than producing spurious failures.
    """
    callspec = getattr(request.node, "callspec", None)
    device = callspec.params.get("device") if callspec is not None else None
    if device is None:
        # Test does not parametrize on device (e.g. test_warmup_attach_stage_not_called_for_cpu).
        return
    locked = _LOCKED_DEVICE[0]
    if locked is None:
        _LOCKED_DEVICE[0] = device
        return
    if device != locked:
        pytest.skip(
            f"ovphysx process-global device lock is held by '{locked}'; cannot run '{device}' "
            "tests in the same session.  Run pytest twice (once per device) for full coverage."
        )


def _ovphysx_sim_context(device: str, **kwargs):
    """Wrapper around :func:`build_simulation_context` that injects OVPhysX cfg.

    PhysX tests pass ``device=device`` directly and let
    :func:`build_simulation_context` build a default :class:`SimulationCfg`.
    OVPhysX needs ``physics=OvPhysxCfg()`` set on the cfg so the manager
    dispatches to OVPhysX rather than PhysX, so we build the cfg here and
    pass it through.  ``gravity_enabled`` is consumed locally (it is ignored
    by ``build_simulation_context`` once a ``sim_cfg`` is provided).
    ``add_ground_plane``, ``auto_add_lighting``, and other kwargs continue
    to flow through ``build_simulation_context`` as before.
    """
    dt = kwargs.pop("dt", 1.0 / 60.0)
    gravity_enabled = kwargs.pop("gravity_enabled", True)
    use_newton_actuators = kwargs.pop("use_newton_actuators", False)
    gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
    sim_cfg = SimulationCfg(
        physics=OvPhysxCfg(),
        device=device,
        dt=dt,
        gravity=gravity,
        use_newton_actuators=use_newton_actuators,
    )
    return build_simulation_context(device=device, sim_cfg=sim_cfg, **kwargs)


def generate_articulation_cfg(
    articulation_type: str,
    stiffness: float | None = 10.0,
    damping: float | None = 2.0,
    actuator_velocity_limit: float | None = None,
    actuator_effort_limit: float | None = None,
    joint_velocity_limit: float | None = None,
    joint_effort_limit: float | None = None,
) -> ArticulationCfg:
    """Generate an articulation configuration.

    Args:
        articulation_type: Type of articulation to generate.
            It should be one of: "humanoid", "panda", "anymal", "shadow_hand", "single_joint_implicit",
            "single_joint_explicit".
        stiffness: Stiffness value for the articulation's actuators. Only currently used for "humanoid".
            Defaults to 10.0.
        damping: Damping value for the articulation's actuators. Only currently used for "humanoid".
            Defaults to 2.0.
        actuator_velocity_limit: Velocity limit for the actuators. Only currently used for "single_joint_implicit"
            and "single_joint_explicit".
        actuator_effort_limit: Effort limit for explicit actuators. Only currently used for
            "single_joint_explicit".
        joint_velocity_limit: Velocity limit for the actuators (set into the simulation).
            Only currently used for "single_joint_implicit" and "single_joint_explicit".
        joint_effort_limit: Effort limit for the actuators (set into the simulation).
            Only currently used for "single_joint_implicit" and "single_joint_explicit".

    Returns:
        The articulation configuration for the requested articulation type.

    """
    if articulation_type == "humanoid":
        articulation_cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/Humanoid/humanoid_instanceable.usd"
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.34)),
            actuators={"body": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=stiffness, damping=damping)},
        )
    elif articulation_type == "panda":
        articulation_cfg = FRANKA_PANDA_CFG
    elif articulation_type == "anymal":
        articulation_cfg = ANYMAL_C_CFG
    elif articulation_type == "shadow_hand":
        articulation_cfg = SHADOW_HAND_PHYSX_CFG
    elif articulation_type == "single_joint_implicit":
        articulation_cfg = ArticulationCfg(
            # we set 80.0 default for max force because default in USD is 10e10 which makes testing annoying.
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
                joint_drive_props=[
                    sim_utils.UsdPhysicsDriveCfg(max_force=80.0),
                    PhysxJointCfg(max_joint_velocity=5.0),
                ],
            ),
            actuators={
                "joint": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    joint_effort_limit=joint_effort_limit,
                    joint_velocity_limit=joint_velocity_limit,
                    actuator_velocity_limit=actuator_velocity_limit,
                    stiffness=2000.0,
                    damping=100.0,
                ),
            },
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos=({"RevoluteJoint": 1.5708}),
                rot=(0.7071081, 0, 0, 0.7071055),
            ),
        )
    elif articulation_type == "single_joint_explicit":
        # we set 80.0 default for max force because default in USD is 10e10 which makes testing annoying.
        articulation_cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
                joint_drive_props=[
                    sim_utils.UsdPhysicsDriveCfg(max_force=80.0),
                    PhysxJointCfg(max_joint_velocity=5.0),
                ],
            ),
            actuators={
                "joint": IdealPDActuatorCfg(
                    joint_names_expr=[".*"],
                    joint_effort_limit=joint_effort_limit,
                    joint_velocity_limit=joint_velocity_limit,
                    actuator_effort_limit=actuator_effort_limit,
                    actuator_velocity_limit=actuator_velocity_limit,
                    stiffness=0.0,
                    damping=10.0,
                ),
            },
        )
    else:
        raise ValueError(
            f"Invalid articulation type: {articulation_type}, valid options are 'humanoid', 'panda', 'anymal',"
            " 'shadow_hand', 'single_joint_implicit' or 'single_joint_explicit'."
        )

    return articulation_cfg


def generate_articulation(
    articulation_cfg: ArticulationCfg, num_articulations: int, device: str
) -> tuple[Articulation, torch.tensor]:
    """Generate an articulation from a configuration.

    Handles the creation of the articulation, the environment prims and the articulation's environment
    translations

    Args:
        articulation_cfg: Articulation configuration.
        num_articulations: Number of articulations to generate.
        device: Device to use for the tensors.

    Returns:
        The articulation and environment translations.

    """
    # Generate translations of 2.5 m in x for each articulation
    translations = torch.zeros(num_articulations, 3, device=device)
    translations[:, 0] = torch.arange(num_articulations) * 2.5

    # Create Top-level Xforms, one for each articulation
    for i in range(num_articulations):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=translations[i][:3])
    articulation = Articulation(articulation_cfg.replace(prim_path="/World/Env_[^/]*/Robot"))

    return articulation, translations


@pytest.mark.parametrize("device", ["cuda:0"])
def test_newton_native_explicit_actuator_submits_ovphysx_effort(device):
    """Run a Newton-native explicit actuator through the current OVPhysX state and effort binding."""
    stiffness, damping, actuator_effort_limit = 20.0, 1.0, 80.0
    with _ovphysx_sim_context(device=device, gravity_enabled=False, use_newton_actuators=True) as sim:
        sim._app_control_on_stop_handle = None
        articulation_cfg = generate_articulation_cfg("single_joint_explicit").replace(
            actuators={
                "joint": IdealPDActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=stiffness,
                    damping=damping,
                    actuator_effort_limit=actuator_effort_limit,
                )
            }
        )
        articulation, _ = generate_articulation(articulation_cfg, 1, device)
        sim.reset()

        initial_pos = articulation.data.joint_pos.torch.clone()
        target = initial_pos + 0.5
        articulation.actuators.target_command.set_position_index(value=target)
        articulation.write_data_to_sim()

        assert articulation._actuator_control.native_actuator_path_active
        assert articulation.newton_actuator_adapter is not None
        assert torch.any(articulation.actuators.computed_effort.torch != 0.0)
        assert torch.any(articulation.actuators.applied_effort.torch != 0.0)
        torch.testing.assert_close(
            _read_binding_to_torch(articulation, TT.DOF_ACTUATION_FORCE, device),
            articulation.actuators.applied_effort.torch,
        )

        sim.step()
        articulation.update(sim.cfg.dt)
        # Use raw OV bindings so the observation cannot refresh the public state shadow.
        current_pos = _read_binding_to_torch(articulation, TT.DOF_POSITION, device)
        current_vel = _read_binding_to_torch(articulation, TT.DOF_VELOCITY, device)
        assert not torch.allclose(current_pos, initial_pos)

        articulation.write_data_to_sim()
        expected_effort = torch.clamp(
            stiffness * (target - current_pos) - damping * current_vel,
            -actuator_effort_limit,
            actuator_effort_limit,
        )
        torch.testing.assert_close(articulation.actuators.applied_effort.torch, expected_effort)


@pytest.mark.parametrize(
    "module_name",
    [
        "isaaclab_physx.assets.articulation.actuator_control",
        "isaaclab_ov.assets.articulation.actuator_control",
    ],
)
def test_host_actuator_control_import_does_not_probe_optional_newton_runtime(monkeypatch, module_name):
    """Import host controls without probing an unrequested Newton optional dependency."""
    original_find_spec = importlib.util.find_spec

    def reject_newton_probe(name, *args, **kwargs):
        if name.startswith("isaaclab_newton"):
            raise AssertionError("host actuator-control import eagerly probed Newton")
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", reject_newton_probe)
    importlib.reload(importlib.import_module(module_name))


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("use_newton_actuators", [False, True])
def test_ovphysx_effort_binding_excludes_implicit_pd(device, use_newton_actuators):
    """Submit implicit feedforward and explicit PD effort without submitting implicit PD telemetry."""
    stiffness, effort_limit = 20.0, 400.0
    with _ovphysx_sim_context(device=device, gravity_enabled=False, use_newton_actuators=use_newton_actuators) as sim:
        sim._app_control_on_stop_handle = None
        articulation_cfg = CARTPOLE_CFG.replace(
            actuators={
                "cart": ImplicitActuatorCfg(
                    joint_names_expr=["slider_to_cart"],
                    joint_effort_limit=effort_limit,
                    stiffness=stiffness,
                    damping=0.0,
                ),
                "pole": IdealPDActuatorCfg(
                    joint_names_expr=["cart_to_pole"],
                    stiffness=stiffness,
                    damping=0.0,
                    actuator_effort_limit=effort_limit,
                ),
            }
        )
        articulation, _ = generate_articulation(articulation_cfg, 1, device)
        sim.reset()

        joint_names = articulation.joint_names
        cart_id = joint_names.index("slider_to_cart")
        pole_id = joint_names.index("cart_to_pole")
        initial_pos = articulation.data.joint_pos.torch.clone()
        position_target = initial_pos.clone()
        position_target[:, cart_id] += 0.25
        position_target[:, pole_id] += 0.5
        feedforward = torch.zeros_like(initial_pos)
        feedforward[:, cart_id] = 1.5
        feedforward[:, pole_id] = -0.75
        articulation.actuators.target_command.set_position_index(value=position_target)
        articulation.actuators.target_command.set_effort_index(value=feedforward)
        articulation.write_data_to_sim()

        expected_pd_effort = torch.clamp(
            stiffness * (position_target - initial_pos) + feedforward, -effort_limit, effort_limit
        )
        applied_effort = articulation.actuators.applied_effort.torch
        torch.testing.assert_close(applied_effort, expected_pd_effort)
        assert torch.all(applied_effort[:, cart_id] != feedforward[:, cart_id])
        expected_force = expected_pd_effort.clone()
        expected_force[:, cart_id] = feedforward[:, cart_id]
        backend_to_user = [joint_names.index(name) for name in articulation.root_view.dof_names]
        backend_force = _read_binding_to_torch(articulation, TT.DOF_ACTUATION_FORCE, device)
        torch.testing.assert_close(backend_force, expected_force[:, backend_to_user])

        expected_stiffness = torch.zeros_like(initial_pos)
        expected_stiffness[:, cart_id] = stiffness
        torch.testing.assert_close(
            _read_binding_to_torch(articulation, TT.DOF_STIFFNESS, device),
            expected_stiffness[:, backend_to_user],
        )
        assert articulation._actuator_control.native_actuator_path_active == use_newton_actuators


@pytest.mark.parametrize("device", ["cuda:0"])
def test_newton_native_actuator_reset_and_gain_event_are_environment_selective(device):
    """Reset and randomize only the selected OVPhysX native-controller environment."""
    from isaaclab.envs.mdp.events import randomize_actuator_gains  # noqa: PLC0415
    from isaaclab.managers import EventTermCfg, SceneEntityCfg  # noqa: PLC0415

    class Env:
        def __init__(self, asset):
            self.scene = self
            self.num_envs = asset.num_instances
            self.device = asset.device
            self._asset = asset

        def __getitem__(self, name):
            assert name == "robot"
            return self._asset

    with _ovphysx_sim_context(device=device, use_newton_actuators=True) as sim:
        sim._app_control_on_stop_handle = None
        articulation_cfg = generate_articulation_cfg("single_joint_explicit").replace(
            actuators={
                "joint": DelayedPDActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=20.0,
                    damping=1.0,
                    actuator_effort_limit=80.0,
                    min_delay=1,
                    max_delay=1,
                )
            }
        )
        articulation, _ = generate_articulation(articulation_cfg, 2, device)
        sim.reset()
        for _ in range(3):
            articulation.write_data_to_sim()
            sim.step()
            articulation.update(sim.cfg.dt)

        adapter = articulation.newton_actuator_adapter
        stateful_pairs = [
            state
            for actuator, state in zip(adapter.actuators, adapter._states_a)
            if state is not None and getattr(state, "delay_state", None) is not None
        ]
        assert len(stateful_pairs) == 1
        articulation.reset(env_ids=torch.tensor([0], device=device, dtype=torch.long))
        assert stateful_pairs[0].delay_state.num_pushes.numpy().tolist() == [0, 1]

        env = Env(articulation)
        asset_cfg = SceneEntityCfg("robot")
        event_params = {
            "asset_cfg": asset_cfg,
            "stiffness_distribution_params": (101.0, 101.0),
            "damping_distribution_params": (3.0, 3.0),
            "operation": "abs",
            "distribution": "uniform",
        }
        event = randomize_actuator_gains(EventTermCfg(func=randomize_actuator_gains, params=event_params), env)
        event(env, env_ids=torch.tensor([0], device=device), **event_params)

        from isaaclab.actuators.newton import read_group_parameter

        stiffness = read_group_parameter(articulation.actuators, "joint", "controller", "kp")
        damping = read_group_parameter(articulation.actuators, "joint", "controller", "kd")
        torch.testing.assert_close(stiffness, torch.tensor([[101.0], [20.0]], device=device))
        torch.testing.assert_close(damping, torch.tensor([[3.0], [1.0]], device=device))


@pytest.fixture
def sim(request):
    """Create simulation context with the specified device."""
    device = request.getfixturevalue("device")
    if "gravity_enabled" in request.fixturenames:
        gravity_enabled = request.getfixturevalue("gravity_enabled")
    else:
        gravity_enabled = True  # default to gravity enabled
    if "add_ground_plane" in request.fixturenames:
        add_ground_plane = request.getfixturevalue("add_ground_plane")
    else:
        add_ground_plane = False  # default to no ground plane
    dt = request.getfixturevalue("dt") if "dt" in request.fixturenames else 1.0 / 60.0
    with _ovphysx_sim_context(
        device=device, auto_add_lighting=True, gravity_enabled=gravity_enabled, add_ground_plane=add_ground_plane, dt=dt
    ) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("gravity_enabled", [True])
@pytest.mark.parametrize("user_ordering", [False, True])
# Keep integration-time pose/velocity differences below the dynamics tolerance.
@pytest.mark.parametrize("dt", [1e-4])
def test_reversed_joint_dynamics_use_public_joint_basis(sim, device, gravity_enabled, user_ordering, dt):
    """Check velocity, kinetic energy and gravity in the public joint basis."""
    articulation = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(Path(__file__).parent / "data" / "articulation_ordering_branching.usda")
            ),
            actuators={},
            joint_ordering=BRANCHING_MJWARP_JOINT_NAMES if user_ordering else None,
            body_ordering=BRANCHING_MJWARP_BODY_NAMES if user_ordering else None,
        )
    )
    UsdPhysics.FixedJoint.Define(sim.stage, "/World/Robot/fixed_root").GetBody1Rel().SetTargets(["/World/Robot/base"])
    joint = UsdPhysics.RevoluteJoint.Get(sim.stage, "/World/Robot/left_elbow")
    body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
    joint.GetBody0Rel().SetTargets(body1)
    joint.GetBody1Rel().SetTargets(body0)
    for prim in Usd.PrimRange(sim.stage.GetPrimAtPath("/World/Robot")):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Set("Y")
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(prim)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0.2, 0.0, 0.0))
            # Isotropic inertia makes the energy check independent of body rotation.
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1))
    sim.reset()

    velocity = torch.zeros((1, articulation.num_joints), device=device)
    velocity[:, articulation.find_joints("left_shoulder")[0][0]] = 0.4
    velocity[:, articulation.find_joints("left_elbow")[0][0]] = 0.7
    articulation.write_joint_velocity_to_sim_index(velocity=velocity)
    sim.step()
    articulation.update(dt)

    joint_velocity = articulation.data.joint_vel.torch
    predicted_velocity = torch.einsum("nbij,nj->nbi", articulation.data.body_com_jacobian_w.torch, joint_velocity)
    torch.testing.assert_close(predicted_velocity, articulation.data.body_com_vel_w.torch[:, 1:], atol=1e-5, rtol=1e-5)

    generalized_energy = 0.5 * torch.einsum(
        "ni,nij,nj->n", joint_velocity, articulation.data.mass_matrix.torch, joint_velocity
    )
    body_velocity = articulation.data.body_com_vel_w.torch
    body_inertia = articulation.data.body_inertia.torch.reshape(1, articulation.num_bodies, 3, 3)
    body_energy = 0.5 * (
        (articulation.data.body_mass.torch.unsqueeze(-1) * body_velocity[..., :3].square()).sum((-1, -2))
        + torch.einsum("nbi,nbij,nbj->n", body_velocity[..., 3:], body_inertia, body_velocity[..., 3:])
    )
    torch.testing.assert_close(generalized_energy, body_energy, atol=1e-5, rtol=1e-5)

    gravity = torch.tensor(sim.cfg.gravity, device=device)
    body_weight = articulation.data.body_mass.torch[:, 1:, None] * gravity
    expected_compensation = -torch.einsum(
        "nbij,nbi->nj", articulation.data.body_com_jacobian_w.torch[:, :, :3], body_weight
    )
    assert torch.all(expected_compensation.abs() > 0.1)
    torch.testing.assert_close(
        articulation.data.gravity_compensation_forces.torch, expected_compensation, atol=1e-5, rtol=1e-5
    )


def test_joint_dof_sign_resolution_traverses_instance_proxies():
    """Resolve reversed joints inside an instanceable articulation."""
    source_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(source_stage, "/Robot")
    UsdGeom.Xform.Define(source_stage, "/Robot/base")
    UsdGeom.Xform.Define(source_stage, "/Robot/link")
    joint = UsdPhysics.RevoluteJoint.Define(source_stage, "/Robot/joint")
    joint.GetBody0Rel().SetTargets(["/Robot/link"])
    joint.GetBody1Rel().SetTargets(["/Robot/base"])
    stage = Usd.Stage.CreateInMemory()
    instance = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    instance.GetReferences().AddReference(source_stage.GetRootLayer().identifier, "/Robot")
    instance.SetInstanceable(True)

    articulation = Mock(
        cfg=Mock(prim_path="/World/Robot"),
        _joint_names=["joint"],
        _body_names=["base", "link"],
    )

    assert Articulation._resolve_joint_dof_signs(articulation, stage) == (-1,)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_live_anymal_c_manual_joint_ordering_reorders_joint_targets(sim, device):
    """Write nonidentity-ordered joint targets into their intended backend columns."""
    stiffness, damping = 10.0, 2.0
    backend_joint_names = ANYMAL_C_PHYSX_JOINT_NAMES
    joint_ordering = (*backend_joint_names[1:], backend_joint_names[0])
    articulation_cfg = generate_articulation_cfg("anymal").replace(
        joint_ordering=joint_ordering,
        actuators={"legs": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=stiffness, damping=damping)},
    )
    articulation, _ = generate_articulation(articulation_cfg, 1, device=device)
    sim.reset()

    ordering = articulation.joint_ordering
    assert ordering is not None
    user_to_backend = torch.as_tensor(ordering.user_to_backend_indices, dtype=torch.long, device=device)
    backend_to_user = torch.as_tensor(ordering.backend_to_user_indices, dtype=torch.long, device=device)
    assert not torch.equal(user_to_backend, backend_to_user)

    joint_index = torch.arange(articulation.num_joints, dtype=torch.float32, device=device).unsqueeze(0)
    position_target = -0.25 + 0.031 * joint_index
    velocity_target = 0.07 + 0.017 * joint_index
    effort_target = torch.where(joint_index.remainder(2) == 0, 0.0, 0.13 * joint_index)
    joint_pos = articulation.data.joint_pos.torch.clone()
    joint_vel = articulation.data.joint_vel.torch.clone()
    articulation.set_joint_position_target_index(target=position_target)
    articulation.set_joint_velocity_target_index(target=velocity_target)
    articulation.actuators.target_command.set_effort_index(value=effort_target)
    articulation.write_data_to_sim()

    backend_position_target = _read_binding_to_torch(articulation, TT.DOF_POSITION_TARGET, device)
    backend_velocity_target = _read_binding_to_torch(articulation, TT.DOF_VELOCITY_TARGET, device)
    torch.testing.assert_close(backend_position_target, position_target[:, backend_to_user])
    torch.testing.assert_close(backend_velocity_target, velocity_target[:, backend_to_user])
    torch.testing.assert_close(
        _read_binding_to_torch(articulation, TT.DOF_ACTUATION_FORCE, device), effort_target[:, backend_to_user]
    )
    computed_effort = (
        stiffness * (position_target - joint_pos) + damping * (velocity_target - joint_vel) + effort_target
    )
    limits = articulation.data.joint_effort_limits.torch
    expected_telemetry = torch.clamp(computed_effort, min=-limits, max=limits)
    torch.testing.assert_close(articulation.actuators.applied_effort.torch, expected_telemetry)
    assert not torch.allclose(expected_telemetry, effort_target)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_reversed_joint_ordering_joint_state_index_writes_backend_order(sim, device):
    """Read dynamics and write full, partial, and position-only joint state through a nonidentity joint axis."""
    articulation_cfg = generate_articulation_cfg("anymal").replace(
        joint_ordering=tuple(reversed(ANYMAL_C_PHYSX_JOINT_NAMES))
    )
    articulation, _ = generate_articulation(articulation_cfg, 2, device=device)
    sim.reset()

    ordering = articulation.joint_ordering
    assert ordering is not None
    num_joints = articulation.num_joints
    user_to_backend = torch.as_tensor(ordering.user_to_backend_indices, dtype=torch.long, device=device)
    backend_to_user = torch.as_tensor(ordering.backend_to_user_indices, dtype=torch.long, device=device)

    # Floating-base columns stay leading while the actuated-joint axes are gathered.
    generalized_user_to_backend = torch.cat((torch.arange(6, device=device), 6 + user_to_backend))
    raw_jacobian = _read_binding_to_torch(articulation, TT.JACOBIAN, device).reshape(
        2, articulation.num_bodies, 6, num_joints + 6
    )
    raw_mass_matrix = _read_binding_to_torch(articulation, TT.MASS_MATRIX, device)
    raw_gravity = _read_binding_to_torch(articulation, TT.GRAVITY_FORCE, device)
    torch.testing.assert_close(
        articulation.data.body_com_jacobian_w.torch,
        raw_jacobian[:, :, :, generalized_user_to_backend],
    )
    torch.testing.assert_close(
        articulation.data.mass_matrix.torch,
        raw_mass_matrix[:, generalized_user_to_backend, :][:, :, generalized_user_to_backend],
    )
    torch.testing.assert_close(
        articulation.data.gravity_compensation_forces.torch,
        raw_gravity[:, generalized_user_to_backend],
    )

    backend_pos_before = torch.arange(2 * num_joints, dtype=torch.float32, device=device).reshape(2, num_joints)
    backend_vel_before = backend_pos_before + 100.0
    articulation.root_view.set_attribute(TT.DOF_POSITION, wp.from_torch(backend_pos_before.contiguous()))
    articulation.root_view.set_attribute(TT.DOF_VELOCITY, wp.from_torch(backend_vel_before.contiguous()))
    for buffer in (
        articulation.data._joint_pos_buf,
        articulation.data._joint_vel_buf,
        articulation.data._joint_pos_backend,
        articulation.data._joint_vel_backend,
    ):
        if buffer is not None:
            buffer.timestamp = -1.0

    torch.testing.assert_close(articulation.data.joint_pos.torch, backend_pos_before[:, user_to_backend])
    torch.testing.assert_close(articulation.data.joint_vel.torch, backend_vel_before[:, user_to_backend])

    def write_and_check(position, velocity, env_ids, joint_ids, expected_public_pos, expected_public_vel):
        articulation.write_joint_state_to_sim_index(
            position=position,
            velocity=velocity,
            env_ids=env_ids,
            joint_ids=joint_ids,
        )
        torch.testing.assert_close(articulation.data.joint_pos.torch, expected_public_pos)
        torch.testing.assert_close(articulation.data.joint_vel.torch, expected_public_vel)
        torch.testing.assert_close(
            _read_binding_to_torch(articulation, TT.DOF_POSITION, device),
            expected_public_pos[:, backend_to_user],
        )
        torch.testing.assert_close(
            _read_binding_to_torch(articulation, TT.DOF_VELOCITY, device),
            expected_public_vel[:, backend_to_user],
        )

    # Full write.
    position = torch.arange(2 * num_joints, dtype=torch.float32, device=device).reshape(2, num_joints) + 200.0
    velocity = position + 100.0
    write_and_check(position, velocity, None, None, position, velocity)

    # Partial write keeps every unselected public and backend entry.
    joint_ids = [0, 2]
    expected_public_pos = position.clone()
    expected_public_vel = velocity.clone()
    partial_position = torch.tensor([[401.0, 403.0]], device=device)
    partial_velocity = torch.tensor([[501.0, 503.0]], device=device)
    expected_public_pos[1, joint_ids] = partial_position[0]
    expected_public_vel[1, joint_ids] = partial_velocity[0]
    write_and_check(partial_position, partial_velocity, [1], joint_ids, expected_public_pos, expected_public_vel)

    # A position-only partial write (a distinct kernel) preserves every unselected backend joint.
    backend_before = wp.to_torch(articulation.root_view.get_attribute(TT.DOF_POSITION)).clone()
    backend_joint_id = ordering.user_to_backend_indices[0]
    selected_value = backend_before[0, backend_joint_id] + 0.001
    articulation.write_joint_position_to_sim_index(
        position=selected_value.reshape(1, 1),
        env_ids=wp.array([0], dtype=wp.int32, device=device),
        joint_ids=wp.array([0], dtype=wp.int32, device=device),
    )
    expected_backend = backend_before.clone()
    expected_backend[0, backend_joint_id] = selected_value
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_attribute(TT.DOF_POSITION)), expected_backend, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_branching_fixture_physx_ordering_is_identity_on_ovphysx(sim, device):
    """Take the same-backend identity fast path for ``joint_ordering="physx"`` on OVPhysX.

    Live coverage for the same-backend symbolic-convention path: OVPhysX is a
    PhysX-family backend whose articulation view is already in PhysX (breadth-first) order, so
    requesting ``physx`` must expose the public joint/body axes verbatim in backend order with no
    reorder map.
    """
    articulation = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(Path(__file__).parent / "data" / "articulation_ordering_branching.usda")
            ),
            actuators={},
            joint_ordering="physx",
            body_ordering="physx",
        )
    )
    sim.reset()
    assert articulation.is_initialized

    # OVPhysX exposes the native breadth-first PhysX order on the backend axis.
    assert tuple(articulation.backend_joint_names) == BRANCHING_PHYSX_JOINT_NAMES
    assert tuple(articulation.backend_body_names) == BRANCHING_PHYSX_BODY_NAMES

    # Same-backend preset: the public axis equals the backend axis and no reorder map is created.
    assert tuple(articulation.joint_names) == tuple(articulation.backend_joint_names)
    assert tuple(articulation.body_names) == tuple(articulation.backend_body_names)
    assert tuple(articulation.joint_names) == BRANCHING_PHYSX_JOINT_NAMES
    assert tuple(articulation.body_names) == BRANCHING_PHYSX_BODY_NAMES
    assert articulation.joint_ordering is None
    assert articulation.body_ordering is None


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_articulation_dynamics_fixed_base_match_raw_ovphysx_bindings(sim, device):
    """Expose fixed-base computed dynamics through the backend-agnostic data API."""
    articulation, _ = generate_articulation(generate_articulation_cfg("panda"), 1, device=device)
    sim.reset()

    num_generalized_dofs = articulation.num_joints
    num_jacobian_bodies = articulation.num_bodies - 1
    raw_jacobian = _read_binding_to_torch(articulation, TT.JACOBIAN, device).reshape(
        1, num_jacobian_bodies, 6, num_generalized_dofs
    )
    raw_mass_matrix = _read_binding_to_torch(articulation, TT.MASS_MATRIX, device)
    raw_gravity = _read_binding_to_torch(articulation, TT.GRAVITY_FORCE, device)

    body_com_jacobian = articulation.data.body_com_jacobian_w
    mass_matrix = articulation.data.mass_matrix
    gravity = articulation.data.gravity_compensation_forces
    assert body_com_jacobian is articulation.data.body_com_jacobian_w
    assert mass_matrix is articulation.data.mass_matrix
    assert gravity is articulation.data.gravity_compensation_forces

    torch.testing.assert_close(body_com_jacobian.torch, raw_jacobian)
    torch.testing.assert_close(mass_matrix.torch, raw_mass_matrix)
    torch.testing.assert_close(gravity.torch, raw_gravity)
    assert articulation.data.body_link_jacobian_w.torch.shape == raw_jacobian.shape
    assert torch.isfinite(articulation.data.body_link_jacobian_w.torch).all()
    torch.testing.assert_close(mass_matrix.torch, mass_matrix.torch.transpose(-1, -2), rtol=1e-5, atol=1e-5)

    joint_position = articulation.data.joint_pos.torch.clone()
    joint_position[:, 1] += 0.2
    articulation.write_joint_position_to_sim_index(position=joint_position)
    updated_raw_jacobian = _read_binding_to_torch(articulation, TT.JACOBIAN, device).reshape(
        1, num_jacobian_bodies, 6, num_generalized_dofs
    )
    updated_raw_mass_matrix = _read_binding_to_torch(articulation, TT.MASS_MATRIX, device)
    updated_raw_gravity = _read_binding_to_torch(articulation, TT.GRAVITY_FORCE, device)
    assert not torch.allclose(updated_raw_jacobian, raw_jacobian)
    assert not torch.allclose(updated_raw_mass_matrix, raw_mass_matrix)
    torch.testing.assert_close(articulation.data.body_com_jacobian_w.torch, updated_raw_jacobian)
    torch.testing.assert_close(articulation.data.mass_matrix.torch, updated_raw_mass_matrix)
    torch.testing.assert_close(articulation.data.gravity_compensation_forces.torch, updated_raw_gravity)

    joint_velocity = torch.linspace(-0.2, 0.2, articulation.num_joints, dtype=torch.float32, device=device).unsqueeze(0)
    articulation.write_joint_velocity_to_sim_index(velocity=joint_velocity)
    expected_com_velocity = torch.einsum("nbij,nj->nbi", articulation.data.body_com_jacobian_w.torch, joint_velocity)
    expected_link_velocity = torch.einsum("nbij,nj->nbi", articulation.data.body_link_jacobian_w.torch, joint_velocity)
    torch.testing.assert_close(
        expected_com_velocity, articulation.data.body_com_vel_w.torch[:, 1:], atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        expected_link_velocity, articulation.data.body_link_vel_w.torch[:, 1:], atol=1e-5, rtol=1e-4
    )

    # Model-property writes refresh the computed dynamics without advancing simulation time.
    data = articulation.data
    # COM offsets affect COM Jacobians, mass matrices, and gravity forces.
    data.body_com_jacobian_w
    data.mass_matrix
    data.gravity_compensation_forces
    coms = _read_binding_to_torch(articulation, TT.BODY_COM_POSE, device)
    coms[:, -1, 0] += 0.01
    articulation.set_coms_index(coms=wp.from_torch(coms.contiguous(), dtype=wp.transformf))
    assert data._body_com_jacobian_w.timestamp < data._sim_timestamp
    assert data._mass_matrix.timestamp < data._sim_timestamp
    assert data._gravity_compensation_forces.timestamp < data._sim_timestamp
    raw_jacobian = _read_binding_to_torch(articulation, TT.JACOBIAN, device).reshape(
        1, num_jacobian_bodies, 6, num_generalized_dofs
    )
    torch.testing.assert_close(data.body_com_jacobian_w.torch, raw_jacobian)
    torch.testing.assert_close(data.mass_matrix.torch, _read_binding_to_torch(articulation, TT.MASS_MATRIX, device))
    torch.testing.assert_close(
        data.gravity_compensation_forces.torch,
        _read_binding_to_torch(articulation, TT.GRAVITY_FORCE, device),
    )

    # Mass affects the mass matrix and gravity forces, but not the kinematic Jacobian.
    masses = data.body_mass.torch.clone()
    masses[:, -1] *= 1.1
    articulation.set_masses_index(masses=masses)
    assert data._mass_matrix.timestamp < data._sim_timestamp
    assert data._gravity_compensation_forces.timestamp < data._sim_timestamp
    torch.testing.assert_close(data.mass_matrix.torch, _read_binding_to_torch(articulation, TT.MASS_MATRIX, device))
    torch.testing.assert_close(
        data.gravity_compensation_forces.torch,
        _read_binding_to_torch(articulation, TT.GRAVITY_FORCE, device),
    )

    # Inertia and armature each affect only the generalized mass matrix.
    inertias = data.body_inertia.torch.clone()
    inertias[:, -1, [0, 4, 8]] *= 1.1
    articulation.set_inertias_index(inertias=inertias)
    assert data._mass_matrix.timestamp < data._sim_timestamp
    torch.testing.assert_close(data.mass_matrix.torch, _read_binding_to_torch(articulation, TT.MASS_MATRIX, device))

    armature = data.joint_armature.torch.clone()
    armature[:, -1] += 0.01
    articulation.write_joint_armature_to_sim_index(armature=armature)
    assert data._mass_matrix.timestamp < data._sim_timestamp
    torch.testing.assert_close(data.mass_matrix.torch, _read_binding_to_torch(articulation, TT.MASS_MATRIX, device))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_articulation_dynamics_reorder_body_rows_and_joint_axes(sim, device):
    """Resolve depth-first MJWarp order for ``"mjwarp"`` on OVPhysX and gather computed dynamics into it.

    OVPhysX is a PhysX-family backend (native breadth-first order), so requesting ``mjwarp`` triggers a
    temporary Newton USD discovery of the depth-first order and reorders the public joint/body axes to it.
    The MJWarp/DFS ground truth is the same tuple isaaclab_newton's
    ``test_mjwarp_ordering_resolver_matches_newton_backend_names`` pins for its live Newton backend.
    """
    articulation = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(Path(__file__).parent / "data" / "articulation_ordering_branching.usda")
            ),
            actuators={},
            joint_ordering="mjwarp",
            body_ordering="mjwarp",
        )
    )
    sim.reset()
    assert articulation.is_initialized

    # OVPhysX exposes the native breadth-first PhysX order on the backend axis.
    assert tuple(articulation.backend_joint_names) == BRANCHING_PHYSX_JOINT_NAMES
    assert tuple(articulation.backend_body_names) == BRANCHING_PHYSX_BODY_NAMES

    # Cross-backend Newton discovery resolves the depth-first MJWarp order and reorders the public axis.
    assert get_articulation_name_ordering(articulation, "mjwarp", kind="joint") == BRANCHING_MJWARP_JOINT_NAMES
    assert get_articulation_name_ordering(articulation, "mjwarp", kind="body") == BRANCHING_MJWARP_BODY_NAMES
    assert tuple(articulation.joint_names) == BRANCHING_MJWARP_JOINT_NAMES
    assert tuple(articulation.body_names) == BRANCHING_MJWARP_BODY_NAMES

    joint_ordering = articulation.joint_ordering
    body_ordering = articulation.body_ordering
    assert joint_ordering is not None
    assert body_ordering is not None
    joint_user_to_backend = torch.as_tensor(joint_ordering.user_to_backend_indices, dtype=torch.long, device=device)
    body_offset = 1 if articulation.is_fixed_base else 0
    body_user_to_backend = torch.as_tensor(
        [
            backend_body_id - body_offset
            for backend_body_id in body_ordering.user_to_backend_indices
            if not body_offset or backend_body_id != 0
        ],
        dtype=torch.long,
        device=device,
    )
    generalized_user_to_backend = torch.cat(
        (
            torch.arange(articulation.num_base_dofs, device=device),
            articulation.num_base_dofs + joint_user_to_backend,
        )
    )

    raw_jacobian = _read_binding_to_torch(articulation, TT.JACOBIAN, device).reshape(
        1,
        articulation.num_bodies - body_offset,
        6,
        articulation.num_joints + articulation.num_base_dofs,
    )
    raw_mass_matrix = _read_binding_to_torch(articulation, TT.MASS_MATRIX, device)
    raw_gravity = _read_binding_to_torch(articulation, TT.GRAVITY_FORCE, device)

    expected_jacobian = raw_jacobian[:, body_user_to_backend, :, :][:, :, :, generalized_user_to_backend]
    expected_mass_matrix = raw_mass_matrix[:, generalized_user_to_backend, :][:, :, generalized_user_to_backend]
    expected_gravity = raw_gravity[:, generalized_user_to_backend]
    torch.testing.assert_close(articulation.data.body_com_jacobian_w.torch, expected_jacobian)
    torch.testing.assert_close(articulation.data.mass_matrix.torch, expected_mass_matrix)
    torch.testing.assert_close(articulation.data.gravity_compensation_forces.torch, expected_gravity)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("add_ground_plane", [True])
def test_initialization_floating_base_non_root(sim, num_articulations, device, add_ground_plane):
    """Test initialization for a floating-base with articulation root on a rigid body.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is not fixed base
    3. All buffers have correct shapes
    4. The articulation can be simulated

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid", stiffness=0.0, damping=0.0)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    # Play sim
    sim.reset()

    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that is fixed base
    assert not articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert articulation.data.root_pos_w.torch.shape == (num_articulations, 3)
    assert articulation.data.root_quat_w.torch.shape == (num_articulations, 4)
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 21)

    # Cross-check binding shapes against cached counts.  PhysX does this via
    # ``root_view.max_dofs == shared_metatype.dof_count``; on OVPhysX
    # ``root_view`` is an ``OvPhysxView`` over the per-tensor-type bindings, so the equivalent
    # invariant is that each per-DOF / per-link binding's shape agrees with
    # the count cached on the asset.
    for tt in (TT.DOF_POSITION, TT.DOF_VELOCITY, TT.DOF_STIFFNESS):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_joints
    for tt in (TT.BODY_MASS, TT.BODY_COM_POSE):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_bodies
    # Body-name ordering check is degenerate on OVPhysX: ``body_names`` is
    # sourced from binding metadata (``sample.body_names``), so the PhysX
    # ``link_paths[0]`` round-trip is a no-op here and is omitted.
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg
        assert actuator.joint_indices == slice(None)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("add_ground_plane", [True])
def test_initialization_floating_base(sim, num_articulations, device, add_ground_plane):
    """Test initialization for a floating-base with articulation root on provided prim path.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is not fixed base
    3. All buffers have correct shapes
    4. The articulation can be simulated

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal", stiffness=0.0, damping=0.0)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that floating base
    assert not articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert articulation.data.root_pos_w.torch.shape == (num_articulations, 3)
    assert articulation.data.root_quat_w.torch.shape == (num_articulations, 4)
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 12)
    assert articulation.data.body_mass.torch.shape == (num_articulations, articulation.num_bodies)
    assert articulation.data.body_inertia.torch.shape == (num_articulations, articulation.num_bodies, 9)

    # Cross-check binding shapes against cached counts.  PhysX does this via
    # ``root_view.max_dofs == shared_metatype.dof_count``; on OVPhysX
    # ``root_view`` is an ``OvPhysxView`` over the per-tensor-type bindings, so the equivalent
    # invariant is that each per-DOF / per-link binding's shape agrees with
    # the count cached on the asset.
    for tt in (TT.DOF_POSITION, TT.DOF_VELOCITY, TT.DOF_STIFFNESS):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_joints
    for tt in (TT.BODY_MASS, TT.BODY_COM_POSE):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_bodies
    # Body-name ordering check is degenerate on OVPhysX: ``body_names`` is
    # sourced from binding metadata (``sample.body_names``), so the PhysX
    # ``link_paths[0]`` round-trip is a no-op here and is omitted.
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_initialization_fixed_base(sim, num_articulations, device):
    """Test initialization for fixed base.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is fixed base
    3. All buffers have correct shapes
    4. The articulation maintains its default state

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="panda")
    articulation, translations = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that fixed base
    assert articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert articulation.data.root_pos_w.torch.shape == (num_articulations, 3)
    assert articulation.data.root_quat_w.torch.shape == (num_articulations, 4)
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 9)
    assert articulation.data.body_mass.torch.shape == (num_articulations, articulation.num_bodies)
    assert articulation.data.body_inertia.torch.shape == (num_articulations, articulation.num_bodies, 9)

    # Cross-check binding shapes against cached counts.  PhysX does this via
    # ``root_view.max_dofs == shared_metatype.dof_count``; on OVPhysX
    # ``root_view`` is an ``OvPhysxView`` over the per-tensor-type bindings, so the equivalent
    # invariant is that each per-DOF / per-link binding's shape agrees with
    # the count cached on the asset.
    for tt in (TT.DOF_POSITION, TT.DOF_VELOCITY, TT.DOF_STIFFNESS):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_joints
    for tt in (TT.BODY_MASS, TT.BODY_COM_POSE):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_bodies
    # Body-name ordering check is degenerate on OVPhysX: ``body_names`` is
    # sourced from binding metadata (``sample.body_names``), so the PhysX
    # ``link_paths[0]`` round-trip is a no-op here and is omitted.
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg
        assert isinstance(actuator.joint_indices, torch.Tensor)
        assert actuator.joint_indices.dtype == torch.int32
        assert actuator.joint_indices.device == torch.device(device)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)

        # check that the root is at the correct state - its default state as it is fixed base
        default_root_pose = articulation.data.default_root_pose.torch.clone()
        default_root_vel = articulation.data.default_root_vel.torch.clone()
        default_root_pose[:, :3] = default_root_pose[:, :3] + translations

        torch.testing.assert_close(articulation.data.root_link_pose_w.torch, default_root_pose)
        torch.testing.assert_close(articulation.data.root_com_vel_w.torch, default_root_vel)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("add_ground_plane", [True])
def test_initialization_fixed_base_single_joint(sim, num_articulations, device, add_ground_plane):
    """Test initialization for fixed base articulation with a single joint.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is fixed base
    3. All buffers have correct shapes
    4. The articulation maintains its default state

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="single_joint_implicit")
    articulation, translations = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that fixed base
    assert articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert articulation.data.root_pos_w.torch.shape == (num_articulations, 3)
    assert articulation.data.root_quat_w.torch.shape == (num_articulations, 4)
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 1)
    assert articulation.data.body_mass.torch.shape == (num_articulations, articulation.num_bodies)
    assert articulation.data.body_inertia.torch.shape == (num_articulations, articulation.num_bodies, 9)

    # Cross-check binding shapes against cached counts.  PhysX does this via
    # ``root_view.max_dofs == shared_metatype.dof_count``; on OVPhysX
    # ``root_view`` is an ``OvPhysxView`` over the per-tensor-type bindings, so the equivalent
    # invariant is that each per-DOF / per-link binding's shape agrees with
    # the count cached on the asset.
    for tt in (TT.DOF_POSITION, TT.DOF_VELOCITY, TT.DOF_STIFFNESS):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_joints
    for tt in (TT.BODY_MASS, TT.BODY_COM_POSE):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_bodies
    # Body-name ordering check is degenerate on OVPhysX: ``body_names`` is
    # sourced from binding metadata (``sample.body_names``), so the PhysX
    # ``link_paths[0]`` round-trip is a no-op here and is omitted.
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)

        # check that the root is at the correct state - its default state as it is fixed base
        default_root_pose = articulation.data.default_root_pose.torch.clone()
        default_root_vel = articulation.data.default_root_vel.torch.clone()
        default_root_pose[:, :3] = default_root_pose[:, :3] + translations

        torch.testing.assert_close(articulation.data.root_link_pose_w.torch, default_root_pose)
        torch.testing.assert_close(articulation.data.root_com_vel_w.torch, default_root_vel)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_fixed_tendon_position_target_writes_offset(sim, num_articulations, device):
    """Initialize the fixed-base tendon hand and land tendon length targets on the selected cells only.

    A target lands in the simulation as ``rest_length - target``. The index form commands every tendon of
    environment 0; the mask form commands tendon 0 of environment 1. Every other cell must keep its initial offset.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="shadow_hand")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    sim.reset()
    assert articulation.is_initialized
    assert articulation.is_fixed_base
    assert articulation.data.root_pos_w.torch.shape == (num_articulations, 3)
    assert articulation.data.root_quat_w.torch.shape == (num_articulations, 4)
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 24)
    assert articulation.data.body_mass.torch.shape == (num_articulations, articulation.num_bodies)
    assert articulation.data.body_inertia.torch.shape == (num_articulations, articulation.num_bodies, 9)
    for tt in (TT.DOF_POSITION, TT.DOF_VELOCITY, TT.DOF_STIFFNESS):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_joints
    for tt in (TT.BODY_MASS, TT.BODY_COM_POSE):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_bodies
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert actuator.is_implicit_model == is_implicit_model_cfg

    num_tendons = articulation.num_fixed_tendons
    assert num_tendons > 0
    rest_length = articulation.data.fixed_tendon_rest_length.torch.clone()
    initial_offset = articulation.data.fixed_tendon_offset.torch.clone()

    index_target = torch.full((1, num_tendons), 0.3, dtype=torch.float32, device=device)
    articulation.set_fixed_tendon_position_target_index(target=index_target, env_ids=[0])
    # Distinct per-cell values: a uniform target cannot catch the mask form reading the wrong
    # cell, because every wrong read returns the same number.
    mask_target = (
        0.7
        + 0.1 * torch.arange(num_articulations, dtype=torch.float32, device=device).unsqueeze(1)
        + 0.01 * torch.arange(num_tendons, dtype=torch.float32, device=device).unsqueeze(0)
    )
    env_mask = wp.array([False, True], dtype=wp.bool, device=device)
    tendon_mask = wp.array([i == 0 for i in range(num_tendons)], dtype=wp.bool, device=device)
    articulation.set_fixed_tendon_position_target_mask(
        target=mask_target, fixed_tendon_mask=tendon_mask, env_mask=env_mask
    )

    articulation.write_data_to_sim()
    sim.step()
    articulation.update(sim.cfg.dt)

    expected = initial_offset.clone()
    expected[0] = rest_length[0] - 0.3
    expected[1, 0] = rest_length[1, 0] - mask_target[1, 0]
    torch.testing.assert_close(articulation.data.fixed_tendon_offset.torch, expected)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("add_ground_plane", [True])
def test_initialization_floating_base_made_fixed_base(sim, num_articulations, device, add_ground_plane):
    """Test initialization for a floating-base articulation made fixed-base using schema properties.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is fixed base after modification
    3. All buffers have correct shapes
    4. The articulation maintains its default state

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal").copy()
    # Fix root link by making it kinematic
    articulation_cfg.spawn.fix_root_link = True
    articulation, translations = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that is fixed base
    assert articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert articulation.data.root_pos_w.torch.shape == (num_articulations, 3)
    assert articulation.data.root_quat_w.torch.shape == (num_articulations, 4)
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 12)

    # Cross-check binding shapes against cached counts.  PhysX does this via
    # ``root_view.max_dofs == shared_metatype.dof_count``; on OVPhysX
    # ``root_view`` is an ``OvPhysxView`` over the per-tensor-type bindings, so the equivalent
    # invariant is that each per-DOF / per-link binding's shape agrees with
    # the count cached on the asset.
    for tt in (TT.DOF_POSITION, TT.DOF_VELOCITY, TT.DOF_STIFFNESS):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_joints
    for tt in (TT.BODY_MASS, TT.BODY_COM_POSE):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_bodies
    # Body-name ordering check is degenerate on OVPhysX: ``body_names`` is
    # sourced from binding metadata (``sample.body_names``), so the PhysX
    # ``link_paths[0]`` round-trip is a no-op here and is omitted.

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)

        # check that the root is at the correct state - its default state as it is fixed base
        default_root_pose = articulation.data.default_root_pose.torch.clone()
        default_root_vel = articulation.data.default_root_vel.torch.clone()
        default_root_pose[:, :3] = default_root_pose[:, :3] + translations

        torch.testing.assert_close(articulation.data.root_link_pose_w.torch, default_root_pose)
        torch.testing.assert_close(articulation.data.root_com_vel_w.torch, default_root_vel)


@pytest.mark.parametrize("device", ["cpu"])
def test_fragment_fix_root_reenables_existing_joint(sim, device):
    """The fragment path must normalize OVPhysX topology even when a disabled fixed joint exists."""
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal").copy()
    articulation_cfg.spawn.articulation_props = []
    articulation_cfg.spawn.fix_root_link = None
    articulation, _ = generate_articulation(articulation_cfg, num_articulations=1, device=device)

    stage = sim.stage
    asset_path = "/World/Env_0/Robot"
    root = sim_utils.get_first_matching_child_prim(
        asset_path, lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI), stage=stage
    )
    assert root is not None and root.HasAPI(UsdPhysics.RigidBodyAPI)
    old_root_path = root.GetPath().pathString
    final_root = root.GetParent()

    joint = UsdPhysics.FixedJoint.Define(stage, f"{old_root_path}/PreAuthoredFixedJoint")
    joint.CreateBody1Rel().SetTargets([root.GetPath()])
    joint.CreateJointEnabledAttr().Set(False)

    assert sim_utils.apply_articulation_root_properties(f"{asset_path}(/.*)?", [], stage, fix_root_link=True)
    assert joint.GetJointEnabledAttr().Get() is True
    world_joints = []
    for prim in sim_utils.get_all_matching_child_prims(
        asset_path, lambda prim: prim.IsA(UsdPhysics.FixedJoint), stage=stage
    ):
        usd_joint = UsdPhysics.Joint(prim)
        has_body_0 = bool(usd_joint.GetBody0Rel().GetTargets())
        has_body_1 = bool(usd_joint.GetBody1Rel().GetTargets())
        if has_body_0 != has_body_1:
            world_joints.append(prim)
    assert world_joints == [joint.GetPrim()]
    assert final_root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not root.HasAPI(UsdPhysics.ArticulationRootAPI)

    sim.reset()
    assert articulation.is_initialized
    assert articulation.is_fixed_base


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("add_ground_plane", [True])
def test_initialization_fixed_base_made_floating_base(sim, num_articulations, device, add_ground_plane):
    """Test initialization for fixed base made floating-base using schema properties.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is floating base after modification
    3. All buffers have correct shapes
    4. The articulation can be simulated

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    # Copy so the shared panda cfg stays fixed-base for later tests.
    articulation_cfg = generate_articulation_cfg(articulation_type="panda").copy()
    # Unfix root link by making it non-kinematic
    articulation_cfg.spawn.fix_root_link = False
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that is floating base
    assert not articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert articulation.data.root_pos_w.torch.shape == (num_articulations, 3)
    assert articulation.data.root_quat_w.torch.shape == (num_articulations, 4)
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 9)

    # Cross-check binding shapes against cached counts.  PhysX does this via
    # ``root_view.max_dofs == shared_metatype.dof_count``; on OVPhysX
    # ``root_view`` is an ``OvPhysxView`` over the per-tensor-type bindings, so the equivalent
    # invariant is that each per-DOF / per-link binding's shape agrees with
    # the count cached on the asset.
    for tt in (TT.DOF_POSITION, TT.DOF_VELOCITY, TT.DOF_STIFFNESS):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_joints
    for tt in (TT.BODY_MASS, TT.BODY_COM_POSE):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_bodies
    # Body-name ordering check is degenerate on OVPhysX: ``body_names`` is
    # sourced from binding metadata (``sample.body_names``), so the PhysX
    # ``link_paths[0]`` round-trip is a no-op here and is omitted.

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("add_ground_plane", [True])
def test_out_of_range_default_joint_pos(sim, num_articulations, device, add_ground_plane):
    """Test that the default joint position from configuration is out of range.

    This test verifies that:
    1. The articulation fails to initialize when joint positions are out of range
    2. The error is properly handled

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    # Create articulation
    articulation_cfg = generate_articulation_cfg(articulation_type="panda").copy()
    articulation_cfg.init_state.joint_pos = {
        "panda_joint1": 10.0,
        "panda_joint[2, 4]": -20.0,
    }

    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    # Play sim
    with pytest.raises(ValueError):
        sim.reset()


@pytest.mark.parametrize("device", test_devices())
def test_out_of_range_default_joint_vel(sim, device):
    """Test that the default joint velocity from configuration is out of range.

    This test verifies that:
    1. The articulation fails to initialize when joint velocities are out of range
    2. The error is properly handled
    """
    articulation_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Robot")
    articulation_cfg.init_state.joint_vel = {
        "panda_joint1": 100.0,
        "panda_joint[2, 4]": -60.0,
    }
    articulation = Articulation(articulation_cfg)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    # Play sim
    with pytest.raises(ValueError):
        sim.reset()


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("add_ground_plane", [True])
def test_joint_pos_limits(sim, num_articulations, device, add_ground_plane):
    """Test write_joint_limits_to_sim API and when default pos falls outside of the new limits.

    This test verifies that:
    1. Joint limits can be set correctly
    2. Default positions are preserved when setting new limits
    3. Joint limits can be set with indexing
    4. Invalid joint positions are properly handled

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    # Create articulation
    articulation_cfg = generate_articulation_cfg(articulation_type="panda")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized

    # Get current default joint pos
    default_joint_pos = articulation._data.default_joint_pos.torch.clone()

    # Set new joint limits
    limits = torch.zeros(num_articulations, articulation.num_joints, 2, device=device)
    limits[..., 0] = (torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0) * -1.0
    limits[..., 1] = torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0
    articulation.write_joint_position_limit_to_sim_index(limits=limits)

    # Check new limits are in place
    torch.testing.assert_close(articulation._data.joint_pos_limits.torch, limits)
    torch.testing.assert_close(articulation._data.default_joint_pos.torch, default_joint_pos)

    # Set new joint limits with indexing
    env_ids = torch.arange(1, device=device, dtype=torch.int32)
    joint_ids = torch.arange(2, device=device, dtype=torch.int32)
    limits = torch.zeros(env_ids.shape[0], joint_ids.shape[0], 2, device=device)
    limits[..., 0] = (torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) + 5.0) * -1.0
    limits[..., 1] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) + 5.0
    articulation.write_joint_position_limit_to_sim_index(limits=limits, env_ids=env_ids, joint_ids=joint_ids)

    # Check new limits are in place
    torch.testing.assert_close(articulation._data.joint_pos_limits.torch[env_ids][:, joint_ids], limits)
    torch.testing.assert_close(articulation._data.default_joint_pos.torch, default_joint_pos)

    # Set new joint limits that invalidate default joint pos
    limits = torch.zeros(num_articulations, articulation.num_joints, 2, device=device)
    limits[..., 0] = torch.rand(num_articulations, articulation.num_joints, device=device) * -0.1
    limits[..., 1] = torch.rand(num_articulations, articulation.num_joints, device=device) * 0.1
    articulation.write_joint_position_limit_to_sim_index(limits=limits)

    # Check if all values are within the bounds
    default_joint_pos_torch = articulation._data.default_joint_pos.torch
    within_bounds = (default_joint_pos_torch >= limits[..., 0]) & (default_joint_pos_torch <= limits[..., 1])
    assert torch.all(within_bounds)

    # Set new joint limits that invalidate default joint pos with indexing
    limits = torch.zeros(env_ids.shape[0], joint_ids.shape[0], 2, device=device)
    limits[..., 0] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) * -0.1
    limits[..., 1] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) * 0.1
    articulation.write_joint_position_limit_to_sim_index(limits=limits, env_ids=env_ids, joint_ids=joint_ids)

    # Check if all values are within the bounds
    default_joint_pos_torch = articulation._data.default_joint_pos.torch
    within_bounds = (default_joint_pos_torch[env_ids][:, joint_ids] >= limits[..., 0]) & (
        default_joint_pos_torch[env_ids][:, joint_ids] <= limits[..., 1]
    )
    assert torch.all(within_bounds)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_external_force_on_single_body(sim, num_articulations, device):
    """Test application of external force on the base of the articulation.

    This test verifies that:
    1. External forces can be applied to specific bodies
    2. The forces affect the articulation's motion correctly
    3. The articulation responds to the forces as expected

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)
    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies("base")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 1] = 1000.0

    # Now we are ready!
    for _ in range(5):
        # reset root state
        articulation.write_root_pose_to_sim_index(root_pose=articulation.data.default_root_pose.torch.clone())
        articulation.write_root_velocity_to_sim_index(root_velocity=articulation.data.default_root_vel.torch.clone())
        # reset dof state
        joint_pos, joint_vel = (
            articulation.data.default_joint_pos.torch,
            articulation.data.default_joint_vel.torch,
        )
        articulation.write_joint_position_to_sim_index(position=joint_pos)
        articulation.write_joint_velocity_to_sim_index(velocity=joint_vel)
        # reset articulation
        articulation.reset()
        # apply force
        articulation.permanent_wrench_composer.set_forces_and_torques_index(
            forces=external_wrench_b[..., :3], torques=external_wrench_b[..., 3:], body_ids=body_ids
        )
        # perform simulation
        for _ in range(100):
            # apply action to the articulation
            articulation.set_joint_position_target_index(target=articulation.data.default_joint_pos.torch.clone())
            articulation.write_data_to_sim()
            # perform step
            sim.step()
            # update buffers
            articulation.update(sim.cfg.dt)
        # check condition that the articulations have fallen down
        for i in range(num_articulations):
            assert articulation.data.root_pos_w.torch[i, 2].item() < 0.2


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_external_force_on_single_body_at_position(sim, num_articulations, device):
    """Test application of external force on the base of the articulation at a given position.

    This test verifies that:
    1. External forces can be applied to specific bodies at a given position
    2. External forces are calculated and composed correctly
    3. The forces affect the articulation's motion correctly
    4. The articulation responds to the forces as expected

    The global-frame position path is covered by :func:`test_external_force_on_multiple_bodies_at_position`;
    a fixed world-frame force here would lift the robot instead of tipping it over.

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)
    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies("base")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 2] = 500.0
    external_wrench_positions_b = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    external_wrench_positions_b[..., 1] = 1.0

    desired_force = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    desired_force[..., 2] = 1000.0
    desired_torque = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    desired_torque[..., 0] = 1000.0

    # Now we are ready!
    for i in range(5):
        # reset root state
        root_pose = articulation.data.default_root_pose.torch.clone()
        root_pose[0, 0] = 2.5  # space them apart by 2.5m

        articulation.write_root_pose_to_sim_index(root_pose=root_pose)
        articulation.write_root_velocity_to_sim_index(root_velocity=articulation.data.default_root_vel.torch.clone())
        # reset dof state
        joint_pos, joint_vel = (
            articulation.data.default_joint_pos.torch,
            articulation.data.default_joint_vel.torch,
        )
        articulation.write_joint_position_to_sim_index(position=joint_pos)
        articulation.write_joint_velocity_to_sim_index(velocity=joint_vel)
        # reset articulation
        articulation.reset()
        # apply force
        is_global = False

        if i % 2 == 0:
            body_com_pos_w = articulation.data.body_com_pos_w.torch[:, body_ids, :3]
            # is_global = True
            external_wrench_positions_b[..., 0] = 0.0
            external_wrench_positions_b[..., 1] = 1.0
            external_wrench_positions_b[..., 2] = 0.0
            external_wrench_positions_b += body_com_pos_w
        else:
            external_wrench_positions_b[..., 0] = 0.0
            external_wrench_positions_b[..., 1] = 1.0
            external_wrench_positions_b[..., 2] = 0.0

        articulation.permanent_wrench_composer.set_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=external_wrench_positions_b,
            body_ids=body_ids,
            is_global=is_global,
        )
        articulation.permanent_wrench_composer.add_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=external_wrench_positions_b,
            body_ids=body_ids,
            is_global=is_global,
        )
        # perform simulation
        for _ in range(100):
            # apply action to the articulation
            articulation.set_joint_position_target_index(target=articulation.data.default_joint_pos.torch.clone())
            articulation.write_data_to_sim()
            # perform step
            sim.step()
            # update buffers
            articulation.update(sim.cfg.dt)
        # check condition that the articulations have fallen down
        for i in range(num_articulations):
            assert articulation.data.root_pos_w.torch[i, 2].item() < 0.2


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_external_force_on_multiple_bodies(sim, num_articulations, device):
    """Test application of external force on the legs of the articulation.

    This test verifies that:
    1. External forces can be applied to multiple bodies
    2. The forces affect the articulation's motion correctly
    3. The articulation responds to the forces as expected

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies(".*_SHANK")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 1] = 100.0

    # Now we are ready!
    for _ in range(5):
        # reset root state
        articulation.write_root_pose_to_sim_index(root_pose=articulation.data.default_root_pose.torch.clone())
        articulation.write_root_velocity_to_sim_index(root_velocity=articulation.data.default_root_vel.torch.clone())
        # reset dof state
        joint_pos, joint_vel = (
            articulation.data.default_joint_pos.torch,
            articulation.data.default_joint_vel.torch,
        )
        articulation.write_joint_position_to_sim_index(position=joint_pos)
        articulation.write_joint_velocity_to_sim_index(velocity=joint_vel)
        # reset articulation
        articulation.reset()
        # apply force
        articulation.permanent_wrench_composer.set_forces_and_torques_index(
            forces=external_wrench_b[..., :3], torques=external_wrench_b[..., 3:], body_ids=body_ids
        )
        # perform simulation
        for _ in range(100):
            # apply action to the articulation
            articulation.set_joint_position_target_index(target=articulation.data.default_joint_pos.torch.clone())
            articulation.write_data_to_sim()
            # perform step
            sim.step()
            # update buffers
            articulation.update(sim.cfg.dt)
        # check condition
        for i in range(num_articulations):
            # since there is a moment applied on the articulation, the articulation should rotate
            assert articulation.data.root_ang_vel_w.torch[i, 2].item() > 0.1


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_external_force_on_multiple_bodies_at_position(sim, num_articulations, device):
    """Test application of external force on the legs of the articulation at a given position.

    This test verifies that:
    1. External forces can be applied to multiple bodies at a given position
    2. External forces can be applied to multiple bodies in the global frame
    3. External forces are calculated and composed correctly
    4. The forces affect the articulation's motion correctly
    5. The articulation responds to the forces as expected

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, translations = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies(".*_SHANK")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 2] = 500.0
    external_wrench_positions_b = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    external_wrench_positions_b[..., 1] = 1.0

    desired_force = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    desired_force[..., 2] = 1000.0
    desired_torque = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    desired_torque[..., 0] = 1000.0

    # Now we are ready!
    for i in range(5):
        # Preserve environment separation when converting the default root pose to world coordinates.
        root_pose = articulation.data.default_root_pose.torch.clone()
        root_pose[:, :3] += translations
        articulation.write_root_pose_to_sim_index(root_pose=root_pose)
        articulation.write_root_velocity_to_sim_index(root_velocity=articulation.data.default_root_vel.torch.clone())
        # reset dof state
        joint_pos, joint_vel = (
            articulation.data.default_joint_pos.torch,
            articulation.data.default_joint_vel.torch,
        )
        articulation.write_joint_position_to_sim_index(position=joint_pos)
        articulation.write_joint_velocity_to_sim_index(velocity=joint_vel)
        # reset articulation
        articulation.reset()

        is_global = False
        if i % 2 == 0:
            body_com_pos_w = articulation.data.body_com_pos_w.torch[:, body_ids, :3]
            is_global = True
            external_wrench_positions_b[..., 0] = 0.0
            external_wrench_positions_b[..., 1] = 1.0
            external_wrench_positions_b[..., 2] = 0.0
            external_wrench_positions_b += body_com_pos_w
        else:
            external_wrench_positions_b[..., 0] = 0.0
            external_wrench_positions_b[..., 1] = 1.0
            external_wrench_positions_b[..., 2] = 0.0

        # apply force
        articulation.permanent_wrench_composer.set_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=external_wrench_positions_b,
            body_ids=body_ids,
            is_global=is_global,
        )
        articulation.permanent_wrench_composer.add_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            positions=external_wrench_positions_b,
            body_ids=body_ids,
            is_global=is_global,
        )
        # perform simulation
        for _ in range(100):
            # apply action to the articulation
            articulation.set_joint_position_target_index(target=articulation.data.default_joint_pos.torch.clone())
            articulation.write_data_to_sim()
            # perform step
            sim.step()
            # update buffers
            articulation.update(sim.cfg.dt)
        # check condition
        for i in range(num_articulations):
            # the response axis depends on the link frames, so check that the articulation rotates
            assert torch.linalg.vector_norm(articulation.data.root_ang_vel_w.torch[i]).item() > 0.1


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_loading_gains_from_usd(sim, num_articulations, device):
    """Test that gains are loaded from USD file if actuator model has them as None.

    This test verifies that:
    1. Gains are loaded correctly from USD file
    2. Default gains are applied when not specified
    3. The gains match the expected values

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid", stiffness=None, damping=None)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Play sim
    sim.reset()

    # Expected gains
    # -- Stiffness values
    expected_stiffness = {
        ".*_waist.*": 20.0,
        ".*_upper_arm.*": 10.0,
        "pelvis": 10.0,
        ".*_lower_arm": 2.0,
        ".*_thigh:0": 10.0,
        ".*_thigh:1": 20.0,
        ".*_thigh:2": 10.0,
        ".*_shin": 5.0,
        ".*_foot.*": 2.0,
    }
    indices_list, _, values_list = string_utils.resolve_matching_names_values(
        expected_stiffness, articulation.joint_names
    )
    expected_stiffness = torch.zeros(articulation.num_instances, articulation.num_joints, device=articulation.device)
    expected_stiffness[:, indices_list] = torch.tensor(values_list, device=articulation.device)
    # -- Damping values
    expected_damping = {
        ".*_waist.*": 5.0,
        ".*_upper_arm.*": 5.0,
        "pelvis": 5.0,
        ".*_lower_arm": 1.0,
        ".*_thigh:0": 5.0,
        ".*_thigh:1": 5.0,
        ".*_thigh:2": 5.0,
        ".*_shin": 0.1,
        ".*_foot.*": 1.0,
    }
    indices_list, _, values_list = string_utils.resolve_matching_names_values(
        expected_damping, articulation.joint_names
    )
    expected_damping = torch.zeros_like(expected_stiffness)
    expected_damping[:, indices_list] = torch.tensor(values_list, device=articulation.device)

    # Check that gains are loaded from USD file
    torch.testing.assert_close(articulation.actuators["body"].stiffness, expected_stiffness)
    torch.testing.assert_close(articulation.actuators["body"].damping, expected_damping)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("joint_limit", [1e5, None])
def test_setting_velocity_and_effort_limits_write_to_solver(sim, device, joint_limit):
    """Test that the resolved joint velocity and effort limits reach the PhysX tensor-API solver.

    The full limit-resolution matrix (config override vs. USD default, implicit and explicit
    actuators, actuator-limit soft fallback) is covered on the Newton backend and at unit
    level. This smoke test only verifies the PhysX tensor-API write path: the configured limits (or the
    USD-authored defaults when unset) land in the native solver buffers and match
    ``data.joint_vel_limits`` and ``data.joint_effort_limits``.
    """
    joint_velocity_limit = joint_effort_limit = joint_limit
    articulation_cfg = generate_articulation_cfg(
        articulation_type="single_joint_implicit",
        joint_velocity_limit=joint_velocity_limit,
        joint_effort_limit=joint_effort_limit,
    )
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg,
        num_articulations=1,
        device=device,
    )
    # Play sim
    sim.reset()

    # read the values set into the simulation
    physx_vel_limit = _read_binding_to_torch(articulation, TT.DOF_MAX_VELOCITY, device)
    # check data buffer
    torch.testing.assert_close(articulation.data.joint_vel_limits.torch, physx_vel_limit)
    # the solver clamp comes from joint_velocity_limit when set, otherwise the USD-authored value
    if joint_velocity_limit is None:
        limit = next(
            p.max_joint_velocity for p in articulation_cfg.spawn.joint_drive_props if isinstance(p, PhysxJointCfg)
        )
    else:
        limit = joint_velocity_limit
    expected_velocity_limit = torch.full_like(physx_vel_limit, limit)
    torch.testing.assert_close(physx_vel_limit, expected_velocity_limit)

    # obtain the physx effort limits
    physx_effort_limit = _read_binding_to_torch(articulation, TT.DOF_MAX_FORCE, device)
    # check data buffer
    torch.testing.assert_close(articulation.data.joint_effort_limits.torch, physx_effort_limit)
    # the solver keeps the USD-authored limit unless the user overrides it explicitly
    if joint_effort_limit is None:
        limit = next(
            p.max_force for p in articulation_cfg.spawn.joint_drive_props if isinstance(p, sim_utils.UsdPhysicsDriveCfg)
        )
    else:
        limit = joint_effort_limit
    expected_effort_limit = torch.full_like(physx_effort_limit, limit)
    torch.testing.assert_close(physx_effort_limit, expected_effort_limit)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_reset(sim, num_articulations, device):
    """Test that reset method works properly."""
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
    )

    # Play the simulator
    sim.reset()

    # Now we are ready!
    # reset articulation
    articulation.reset()

    # Reset should zero external forces and torques
    assert not articulation._instantaneous_wrench_composer.active
    assert not articulation._permanent_wrench_composer.active
    assert torch.count_nonzero(articulation._instantaneous_wrench_composer.composed_force.torch) == 0
    assert torch.count_nonzero(articulation._instantaneous_wrench_composer.composed_torque.torch) == 0
    assert torch.count_nonzero(articulation._permanent_wrench_composer.composed_force.torch) == 0
    assert torch.count_nonzero(articulation._permanent_wrench_composer.composed_torque.torch) == 0

    num_bodies = articulation.num_bodies
    articulation.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.ones((num_articulations, num_bodies, 3), device=device),
        torques=torch.ones((num_articulations, num_bodies, 3), device=device),
    )
    articulation.instantaneous_wrench_composer.add_forces_and_torques_index(
        forces=torch.ones((num_articulations, num_bodies, 3), device=device),
        torques=torch.ones((num_articulations, num_bodies, 3), device=device),
    )
    articulation.reset(env_ids=torch.tensor([0], device=device))
    assert articulation._instantaneous_wrench_composer.active
    assert articulation._permanent_wrench_composer.active
    assert torch.count_nonzero(articulation._instantaneous_wrench_composer.composed_force.torch) == num_bodies * 3
    assert torch.count_nonzero(articulation._instantaneous_wrench_composer.composed_torque.torch) == num_bodies * 3
    assert torch.count_nonzero(articulation._permanent_wrench_composer.composed_force.torch) == num_bodies * 3
    assert torch.count_nonzero(articulation._permanent_wrench_composer.composed_torque.torch) == num_bodies * 3


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("add_ground_plane", [True])
def test_apply_joint_command(sim, num_articulations, device, add_ground_plane):
    """Test applying of joint position target functions correctly for a robotic arm."""
    articulation_cfg = generate_articulation_cfg(articulation_type="panda")
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
    )

    # Play the simulator
    sim.reset()

    for _ in range(100):
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

    # reset dof state
    joint_pos = articulation.data.default_joint_pos.torch.clone()
    joint_pos[:, 3] = 0.0

    # apply action to the articulation
    articulation.set_joint_position_target_index(target=joint_pos)
    articulation.write_data_to_sim()

    for _ in range(100):
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

    # The elbow, wrist, and finger joints track the commanded target. The shoulder joints sag under
    # gravity with these gains, so they are not checked. Without the target write, the drives pull
    # every joint toward zero instead (the wrist ends near 0.3 rad instead of 3.0 rad).
    torch.testing.assert_close(articulation.data.joint_pos.torch[:, 3:], joint_pos[:, 3:], atol=0.1, rtol=0.0)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_body_root_state(sim, num_articulations, device):
    """Test for reading the `body_state_w` property.

    This test verifies that:
    1. Body states can be read correctly
    2. States are correct with a COM offset from the link frame
    3. States are consistent across different devices

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type="single_joint_implicit")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)
    env_idx = torch.tensor([x for x in range(num_articulations)], device=device, dtype=torch.int32)
    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10, "Possible reference leak for articulation"
    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized, "Articulation is not initialized"
    # Check that fixed base
    assert articulation.is_fixed_base, "Articulation is not a fixed base"

    # Resolve body indices by name (ordering may differ across physics backends)
    root_idx = articulation.body_names.index("CenterPivot")
    arm_idx = articulation.body_names.index("Arm")

    # change center of mass offset from link frame
    offset = [0.5, 0.0, 0.0]

    # create com offsets — apply offset to the Arm body
    num_bodies = articulation.num_bodies
    com = _read_binding_to_torch(articulation, TT.BODY_COM_POSE, device)
    link_offset = [1.0, 0.0, 0.0]  # the offset from CenterPivot to Arm frames
    new_com = torch.tensor(offset, device=device).repeat(num_articulations, 1, 1)
    com[:, arm_idx, :3] = new_com.squeeze(-2)
    # PhysX uses ``root_view.set_coms``; OVPhysX wraps the wheel
    # ``BODY_COM_POSE`` write in :meth:`set_coms_index` (wp.transformf contract).
    articulation.set_coms_index(
        coms=wp.from_torch(com.contiguous(), dtype=wp.transformf),
        env_ids=wp.from_torch(env_idx, dtype=wp.int32),
    )

    # check they are set
    torch.testing.assert_close(_read_binding_to_torch(articulation, TT.BODY_COM_POSE, device), com)

    for i in range(50):
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

        # get state properties
        root_link_vel_w = articulation.data.root_link_vel_w.torch
        root_com_pose_w = articulation.data.root_com_pose_w.torch
        root_com_vel_w = articulation.data.root_com_vel_w.torch
        body_link_pose_w = articulation.data.body_link_pose_w.torch
        body_link_vel_w = articulation.data.body_link_vel_w.torch
        body_com_pose_w = articulation.data.body_com_pose_w.torch
        body_com_vel_w = articulation.data.body_com_vel_w.torch

        # get joint state
        joint_pos = articulation.data.joint_pos.torch.unsqueeze(-1)
        joint_vel = articulation.data.joint_vel.torch.unsqueeze(-1)

        # LINK state
        # angular velocity should be the same for both COM and link frames
        torch.testing.assert_close(root_com_vel_w[..., 3:], root_link_vel_w[..., 3:])
        torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])

        # lin_vel arm
        lin_vel_gt = torch.zeros(num_articulations, num_bodies, 3, device=device)
        vx = -(link_offset[0]) * joint_vel * torch.sin(joint_pos)
        vy = torch.zeros(num_articulations, 1, 1, device=device)
        vz = (link_offset[0]) * joint_vel * torch.cos(joint_pos)
        lin_vel_gt[:, arm_idx, :] = torch.cat([vx, vy, vz], dim=-1).squeeze(-2)

        # linear velocity of root link should be zero
        torch.testing.assert_close(lin_vel_gt[:, root_idx, :], root_link_vel_w[..., :3], atol=1e-3, rtol=1e-1)
        # linear velocity of pendulum link should be
        torch.testing.assert_close(lin_vel_gt, body_link_vel_w[..., :3], atol=1e-3, rtol=1e-1)

        # ang_vel
        torch.testing.assert_close(root_com_vel_w[..., 3:], root_link_vel_w[..., 3:])
        torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])

        # COM state
        # position and orientation shouldn't match for the _state_com_w but everything else will
        # OVStage determines the runtime link pose from the joint frames, which may differ
        # from the authored USD Xform. Verify the COM offset relative to that runtime pose.
        pos_gt = body_link_pose_w[..., :3].clone()
        px = offset[0] * torch.cos(joint_pos)
        py = torch.zeros(num_articulations, 1, 1, device=device)
        pz = offset[0] * torch.sin(joint_pos)
        pos_gt[:, arm_idx, :] += torch.cat([px, py, pz], dim=-1).squeeze(-2)
        torch.testing.assert_close(pos_gt[:, root_idx, :], root_com_pose_w[..., :3], atol=1e-3, rtol=1e-1)
        torch.testing.assert_close(pos_gt, body_com_pose_w[..., :3], atol=1e-3, rtol=1e-1)

        # orientation
        com_quat_b = articulation.data.body_com_quat_b.torch
        com_quat_w = math_utils.quat_mul(body_link_pose_w[..., 3:], com_quat_b)
        torch.testing.assert_close(com_quat_w, body_com_pose_w[..., 3:])
        torch.testing.assert_close(com_quat_w[:, root_idx, :], root_com_pose_w[..., 3:])

        # angular velocity should be the same for both COM and link frames
        torch.testing.assert_close(root_com_vel_w[..., 3:], root_link_vel_w[..., 3:])
        torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("state_location", ["com", "link"])
@pytest.mark.parametrize("gravity_enabled", [False])
def test_write_root_state(sim, num_articulations, device, state_location, gravity_enabled):
    """Test the setters for root_state using both the link frame and center of mass as reference frame.

    This test verifies that:
    1. Root states can be written correctly
    2. The other frame follows the written one through the COM offset
    3. States can be written for both COM and link frames
    4. States are consistent across different devices

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
        state_location: Whether to test COM or link frame
    """
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)
    env_idx = torch.tensor([x for x in range(num_articulations)], device=device, dtype=torch.int32)

    # Play sim
    sim.reset()

    # change center of mass offset from link frame
    offset = torch.tensor([1.0, 0.0, 0.0]).repeat(num_articulations, 1, 1)

    # create com offsets
    com = _read_binding_to_torch(articulation, TT.BODY_COM_POSE, device)
    new_com = offset.to(device)
    com[:, 0, :3] = new_com.squeeze(-2)
    # See test_body_root_state for the PhysX → OVPhysX setter substitution.
    articulation.set_coms_index(
        coms=wp.from_torch(com.contiguous(), dtype=wp.transformf),
        env_ids=wp.from_torch(env_idx, dtype=wp.int32),
    )

    # check they are set
    torch.testing.assert_close(_read_binding_to_torch(articulation, TT.BODY_COM_POSE, device), com)

    rand_state = torch.zeros(num_articulations, 13, device=device)
    rand_state[..., :7] = articulation.data.default_root_pose.torch
    rand_state[..., :3] += env_pos
    # make quaternion a unit vector
    rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

    # Root-body COM frame in the link frame, from the values written above.
    com_pos_b, com_quat_b = com[:, 0, :3], com[:, 0, 3:7]
    link_pos_in_com, link_quat_in_com = math_utils.subtract_frame_transforms(com_pos_b, com_quat_b)

    env_idx = env_idx.to(device)
    for i in range(10):
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

        if state_location == "com":
            if i % 2 == 0:
                articulation.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7])
                articulation.write_root_com_velocity_to_sim_index(root_velocity=rand_state[..., 7:])
            else:
                articulation.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_idx)
                articulation.write_root_com_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=env_idx)
        elif state_location == "link":
            if i % 2 == 0:
                articulation.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7])
                articulation.write_root_link_velocity_to_sim_index(root_velocity=rand_state[..., 7:])
            else:
                articulation.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_idx)
                articulation.write_root_link_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=env_idx)

        if state_location == "com":
            torch.testing.assert_close(rand_state[..., :7], articulation.data.root_com_pose_w.torch)
            torch.testing.assert_close(rand_state[..., 7:], articulation.data.root_com_vel_w.torch)
            # The link frame is the written COM frame composed with the inverse COM offset.
            expected_link_pos, expected_link_quat = math_utils.combine_frame_transforms(
                rand_state[..., :3], rand_state[..., 3:7], link_pos_in_com, link_quat_in_com
            )
            torch.testing.assert_close(articulation.data.root_link_pos_w.torch, expected_link_pos)
            torch.testing.assert_close(articulation.data.root_link_quat_w.torch, expected_link_quat)
        elif state_location == "link":
            torch.testing.assert_close(rand_state[..., :7], articulation.data.root_link_pose_w.torch)
            torch.testing.assert_close(rand_state[..., 7:], articulation.data.root_link_vel_w.torch)
            # The COM frame is the written link frame composed with the COM offset.
            expected_com_pos, expected_com_quat = math_utils.combine_frame_transforms(
                rand_state[..., :3], rand_state[..., 3:7], com_pos_b, com_quat_b
            )
            torch.testing.assert_close(articulation.data.root_com_pos_w.torch, expected_com_pos)
            torch.testing.assert_close(articulation.data.root_com_quat_w.torch, expected_com_quat)


@pytest.mark.parametrize("device", ["cpu"])
def test_com_orientation_write_invalidates_static_inertia_cache_with_body_ordering(sim, device):
    """Partial COM writes and COM rotations follow non-identity body order.

    COM pose is a CPU-resident OVPhysX binding, and the raw ``root_view.set_attribute`` restore
    below forbids cross-device staging, so this test is CPU-only.
    """
    sim._app_control_on_stop_handle = None
    articulation_cfg = FRANKA_PANDA_CFG.replace(body_ordering=PANDA_ROOT_PRESERVING_REVERSED_BODY_NAMES)
    articulation, _ = generate_articulation(articulation_cfg, 1, device=device)

    sim.reset()
    articulation.update(sim.cfg.dt)
    assert articulation.body_ordering is not None

    public_body_id = 1
    backend_body_id = articulation.body_ordering.user_to_backend_indices[public_body_id]
    assert backend_body_id != public_body_id

    # A partial ordered COM write lands on the selected backend body and preserves every other one.
    backend_before = _read_binding_to_torch(articulation, TT.BODY_COM_POSE, device).clone()
    assert torch.unique(backend_before[0], dim=0).shape[0] > 1
    articulation.data._body_com_pose_b.timestamp = -1.0
    backend_staging = articulation.data._body_com_pose_b_backend
    if backend_staging is not None:
        backend_staging.timestamp = -1.0
    selected_com = backend_before[0, backend_body_id].clone()
    selected_com[0] += 0.001
    articulation.set_coms_index(
        coms=wp.from_torch(selected_com.reshape(1, 1, 7).contiguous(), dtype=wp.transformf),
        env_ids=wp.array([0], dtype=wp.int32, device=device),
        body_ids=wp.array([public_body_id], dtype=wp.int32, device=device),
    )
    backend_after = _read_binding_to_torch(articulation, TT.BODY_COM_POSE, device).clone()
    articulation.root_view.set_attribute(TT.BODY_COM_POSE, wp.from_torch(backend_before.contiguous()))
    noop_after = _read_binding_to_torch(articulation, TT.BODY_COM_POSE, device).clone()
    unselected_body_mask = torch.ones(backend_before.shape[1], dtype=torch.bool, device=device)
    unselected_body_mask[backend_body_id] = False
    assert torch.equal(noop_after[..., :3], backend_before[..., :3])
    assert torch.equal(backend_after[0, backend_body_id, :3], selected_com[:3])
    assert torch.equal(backend_after[0, unselected_body_mask, :3], backend_before[0, unselected_body_mask, :3])
    # Bound semantic orientation equality by the native setter's float32 no-op normalization.
    native_orientation_atol = torch.max(
        torch.abs(noop_after[0, unselected_body_mask, 3:7] - backend_before[0, unselected_body_mask, 3:7])
    ).item()
    assert native_orientation_atol <= torch.finfo(backend_before.dtype).eps
    torch.testing.assert_close(
        backend_after[0, unselected_body_mask, 3:7],
        backend_before[0, unselected_body_mask, 3:7],
        rtol=0.0,
        atol=native_orientation_atol,
    )
    assert torch.equal(backend_after[..., 3:7], noop_after[..., 3:7])

    # A COM rotation refreshes the static inertia cache.
    coms = articulation.data.body_com_pose_b.torch[:, public_body_id : public_body_id + 1].clone()
    coms[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    articulation.set_coms_index(coms=wp.from_torch(coms.contiguous(), dtype=wp.transformf), body_ids=[public_body_id])

    principal_inertia = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]]], device=device)
    articulation.set_inertias_index(inertias=principal_inertia, body_ids=[public_body_id])
    torch.testing.assert_close(
        articulation.data.body_inertia.torch[:, public_body_id : public_body_id + 1], principal_inertia
    )

    coms[..., 3:7] = torch.tensor([0.0, 0.0, 0.70710677, 0.70710677], device=device)
    articulation.set_coms_index(coms=wp.from_torch(coms.contiguous(), dtype=wp.transformf), body_ids=[public_body_id])
    sim.step()
    articulation.update(sim.cfg.dt)
    expected_rotated_inertia = torch.tensor([[[2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0]]], device=device)
    torch.testing.assert_close(
        articulation.data.body_inertia.torch[:, public_body_id : public_body_id + 1],
        expected_rotated_inertia,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_root_link_vel_w_refreshes_fk_before_body_com_vel_w_read(sim, device):
    """Reading ``root_link_vel_w`` must run FK before ``body_com_vel_w`` sees a "fresh" buffer.

    Regression test for a bug where ``root_link_vel_w`` read the ``LINK_VELOCITY`` binding without
    first calling ``_ensure_fk_fresh()``, unlike the sibling ``body_com_vel_w`` / ``body_link_pose_w``
    getters. ``_read_binding_into_buf`` stamps a buffer's timestamp as fresh unconditionally, so a
    ``root_link_vel_w`` read performed right after ``write_joint_velocity_to_sim_index`` (which sets
    ``_fk_timestamp = -1.0`` to force a refresh) would mark the shared velocity buffer fresh *before*
    FK actually ran. A subsequent ``body_com_vel_w`` read then sees the buffer already fresh and skips
    its own re-read, silently returning pre-FK data.

    The OVPhysX kitless backend recomputes ``LINK_VELOCITY`` eagerly on every attribute read
    regardless of whether ``update_articulations_kinematic`` was called, so comparing the numeric
    value of ``body_com_vel_w`` before and after the fix would pass either way here. The invariant
    that actually catches the bug is that ``_fk_timestamp`` must be current by the time
    ``root_link_vel_w`` finishes reading, so every dependent buffer it marks fresh is trustworthy.
    """
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type="single_joint_implicit")
    articulation, _ = generate_articulation(articulation_cfg, 2, device=device)

    sim.reset()
    articulation.update(sim.cfg.dt)

    # Prime the derived buffers before the write so their TimestampedBuffers are populated; otherwise
    # the reads below would trivially be "first reads" regardless of the cache-invalidation bug.
    articulation.data.root_link_vel_w
    articulation.data.body_com_vel_w

    joint_vel = torch.full((2, articulation.num_joints), 3.0, device=device)
    articulation.write_joint_velocity_to_sim_index(velocity=joint_vel)

    # The velocity write forces a kinematic refresh on the next FK-dependent read.
    assert articulation.data._fk_timestamp < 0.0

    articulation.data.root_link_vel_w
    # `root_link_vel_w` must have triggered the FK refresh itself -- it cannot rely on a later
    # `body_com_vel_w` read to do so, because it already marks the shared velocity buffer fresh.
    assert articulation.data._fk_timestamp == articulation.data._sim_timestamp

    body_com_vel_w = articulation.data.body_com_vel_w.torch
    assert torch.linalg.norm(body_com_vel_w[:, 1, :]) > 1e-3


@pytest.mark.parametrize("device", test_devices())
def test_setting_articulation_root_prim_path(sim, device):
    """Test that the articulation root prim path can be set explicitly."""
    sim._app_control_on_stop_handle = None
    # Create articulation
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
    articulation_cfg.articulation_root_prim_path = "/torso"
    articulation, _ = generate_articulation(articulation_cfg, 1, device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation._is_initialized


@pytest.mark.parametrize("device", test_devices())
def test_setting_invalid_articulation_root_prim_path(sim, device):
    """Test that the articulation root prim path can be set explicitly."""
    sim._app_control_on_stop_handle = None
    # Create articulation
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
    articulation_cfg.articulation_root_prim_path = "/non_existing_prim_path"
    articulation, _ = generate_articulation(articulation_cfg, 1, device=device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    # Play sim
    with pytest.raises(RuntimeError):
        sim.reset()


@pytest.mark.parametrize("device", test_devices())
def test_write_joint_state_to_sim_index_partial(sim, device, mocker):
    """Test fused joint-state writes with partial environment and joint indices.

    The first write uses unsorted int64 selectors; the deprecated combined writer must delegate
    to the public index writers.
    """
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, _ = generate_articulation(articulation_cfg, 2, device)
    sim.reset()

    original_joint_pos = articulation.data.joint_pos.torch.clone()
    original_joint_vel = articulation.data.joint_vel.torch.clone()
    _ = articulation.data.body_link_pose_w
    _ = articulation.data.body_com_vel_w
    pose_timestamp = articulation.data._body_link_pose_w.timestamp
    velocity_timestamp = articulation.data._body_com_vel_w.timestamp

    previous_joint_vel = wp.to_torch(articulation.data._previous_joint_vel)
    joint_acc = wp.to_torch(articulation.data._joint_acc.data)
    previous_joint_vel.fill_(3.0)
    joint_acc.fill_(4.0)
    articulation.data._joint_acc.timestamp = -1.0

    position = torch.tensor([[0.1, -0.1]], device=device)
    velocity = torch.tensor([[0.2, -0.2]], device=device)
    # int64 selectors with the joints in reverse order; the payload follows the selector order.
    articulation.write_joint_state_to_sim_index(
        position=position[:, [1, 0]],
        velocity=velocity[:, [1, 0]],
        env_ids=torch.tensor([1], dtype=torch.int64, device=device),
        joint_ids=torch.tensor([2, 0], dtype=torch.int64, device=device),
        skip_forward=True,
    )

    expected_joint_pos = original_joint_pos.clone()
    expected_joint_vel = original_joint_vel.clone()
    expected_previous_joint_vel = torch.full_like(previous_joint_vel, 3.0)
    expected_joint_acc = torch.full_like(joint_acc, 4.0)
    expected_joint_pos[1, [0, 2]] = position[0]
    expected_joint_vel[1, [0, 2]] = velocity[0]
    expected_previous_joint_vel[1, [0, 2]] = velocity[0]
    expected_joint_acc[1, [0, 2]] = 0.0

    torch.testing.assert_close(articulation.data.joint_pos.torch, expected_joint_pos)
    torch.testing.assert_close(articulation.data.joint_vel.torch, expected_joint_vel)
    torch.testing.assert_close(previous_joint_vel, expected_previous_joint_vel)
    torch.testing.assert_close(joint_acc, expected_joint_acc)
    assert articulation.data._joint_acc.timestamp == articulation.data._sim_timestamp
    assert articulation.data._body_link_pose_w.timestamp == pose_timestamp
    assert articulation.data._body_com_vel_w.timestamp == velocity_timestamp
    torch.testing.assert_close(_read_binding_to_torch(articulation, TT.DOF_POSITION, device), expected_joint_pos)
    torch.testing.assert_close(_read_binding_to_torch(articulation, TT.DOF_VELOCITY, device), expected_joint_vel)

    articulation.write_joint_state_to_sim_index(position=position, velocity=velocity, env_ids=[1], joint_ids=[0, 2])
    assert articulation.data._body_link_pose_w.timestamp < articulation.data._sim_timestamp
    assert articulation.data._body_com_vel_w.timestamp < articulation.data._sim_timestamp

    # The deprecated combined API stays a thin composition of the public writers.
    position = torch.tensor([[0.1]], device=device)
    velocity = torch.tensor([[0.2]], device=device)
    position_writer = mocker.patch.object(articulation, "write_joint_position_to_sim_index")
    velocity_writer = mocker.patch.object(articulation, "write_joint_velocity_to_sim_index")
    with pytest.warns(DeprecationWarning):
        articulation.write_joint_state_to_sim(position=position, velocity=velocity, joint_ids=[0], env_ids=[0])
    position_writer.assert_called_once_with(position=position, joint_ids=[0], env_ids=[0])
    velocity_writer.assert_called_once_with(velocity=velocity, joint_ids=[0], env_ids=[0])


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("gravity_enabled", [False])
def test_write_joint_state_data_consistency(sim, num_articulations, device, gravity_enabled):
    """Test the setters for root_state using both the link frame and center of mass as reference frame.

    This test verifies that after write_joint_state_to_sim operations:
    1. state, com_state, link_state value consistency
    2. body_pose, link
    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)
    env_idx = torch.tensor([x for x in range(num_articulations)])

    # Play sim
    sim.reset()

    limits = torch.zeros(num_articulations, articulation.num_joints, 2, device=device)
    limits[..., 0] = (torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0) * -1.0
    limits[..., 1] = torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0
    articulation.write_joint_position_limit_to_sim_index(limits=limits)

    from torch.distributions import Uniform

    joint_pos_limits = articulation.data.joint_pos_limits.torch
    joint_vel_limits = articulation.data.joint_vel_limits.torch
    pos_dist = Uniform(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    vel_dist = Uniform(-joint_vel_limits, joint_vel_limits)

    original_body_link_pose_w = articulation.data.body_link_pose_w.torch.clone()
    original_body_com_vel_w = articulation.data.body_com_vel_w.torch.clone()

    rand_joint_pos = pos_dist.sample()
    rand_joint_vel = vel_dist.sample()

    articulation.write_joint_state_to_sim_index(position=rand_joint_pos, velocity=rand_joint_vel)
    # make sure valued updated
    body_link_pose_w = articulation.data.body_link_pose_w.torch
    body_com_vel_w = articulation.data.body_com_vel_w.torch
    original_body_states = torch.cat([original_body_link_pose_w, original_body_com_vel_w], dim=-1)
    body_state_w = torch.cat([body_link_pose_w, body_com_vel_w], dim=-1)
    assert torch.count_nonzero(original_body_states[:, 1:] != body_state_w[:, 1:]) > (
        len(original_body_states[:, 1:]) / 2
    )
    # validate body - link consistency
    body_link_vel_w = articulation.data.body_link_vel_w.torch
    # skip lin_vel because it differs from link frame, this should be fine because we are only checking
    # if velocity update is triggered, which can be determined by comparing angular velocity
    torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])

    # validate link - com conistency
    body_com_pos_b = articulation.data.body_com_pos_b.torch
    body_com_quat_b = articulation.data.body_com_quat_b.torch
    expected_com_pos, expected_com_quat = math_utils.combine_frame_transforms(
        body_link_pose_w[..., :3].view(-1, 3),
        body_link_pose_w[..., 3:].view(-1, 4),
        body_com_pos_b.view(-1, 3),
        body_com_quat_b.view(-1, 4),
    )
    body_com_pos_w = articulation.data.body_com_pos_w.torch
    body_com_quat_w = articulation.data.body_com_quat_w.torch
    torch.testing.assert_close(expected_com_pos.view(len(env_idx), -1, 3), body_com_pos_w)
    torch.testing.assert_close(expected_com_quat.view(len(env_idx), -1, 4), body_com_quat_w)

    # validate body - com consistency
    body_com_lin_vel_w = articulation.data.body_com_lin_vel_w.torch
    body_com_ang_vel_w = articulation.data.body_com_ang_vel_w.torch
    torch.testing.assert_close(body_com_vel_w[..., :3], body_com_lin_vel_w)
    torch.testing.assert_close(body_com_vel_w[..., 3:], body_com_ang_vel_w)

    # validate pos_w, quat_w, pos_b, quat_b is consistent with pose_w and pose_b
    expected_com_pose_w = torch.cat((body_com_pos_w, body_com_quat_w), dim=2)
    expected_com_pose_b = torch.cat((body_com_pos_b, body_com_quat_b), dim=2)
    body_pos_w = articulation.data.body_pos_w.torch
    body_quat_w = articulation.data.body_quat_w.torch
    expected_body_pose_w = torch.cat((body_pos_w, body_quat_w), dim=2)
    body_link_pos_w = articulation.data.body_link_pos_w.torch
    body_link_quat_w = articulation.data.body_link_quat_w.torch
    expected_body_link_pose_w = torch.cat((body_link_pos_w, body_link_quat_w), dim=2)
    body_com_pose_w = articulation.data.body_com_pose_w.torch
    body_com_pose_b = articulation.data.body_com_pose_b.torch
    body_pose_w = articulation.data.body_pose_w.torch
    body_link_pose_w_fresh = articulation.data.body_link_pose_w.torch
    torch.testing.assert_close(body_com_pose_w, expected_com_pose_w)
    torch.testing.assert_close(body_com_pose_b, expected_com_pose_b)
    torch.testing.assert_close(body_pose_w, expected_body_pose_w)
    torch.testing.assert_close(body_link_pose_w_fresh, expected_body_link_pose_w)

    # validate pose_w is consistent with individual properties
    body_vel_w = articulation.data.body_vel_w.torch
    body_com_vel_w_fresh = articulation.data.body_com_vel_w.torch
    torch.testing.assert_close(body_pose_w, body_link_pose_w)
    torch.testing.assert_close(body_vel_w, body_com_vel_w)
    torch.testing.assert_close(body_vel_w, body_com_vel_w_fresh)


@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_write_joint_frictions_to_sim(sim, num_articulations, device, add_ground_plane):
    """Write joint friction coefficients and read them back from the simulation and public getters."""
    articulation_cfg = generate_articulation_cfg(articulation_type="panda")
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
    )

    # Play the simulator
    sim.reset()

    # apply action to the articulation
    dynamic_friction = torch.rand(num_articulations, articulation.num_joints, device=device)
    viscous_friction = torch.rand(num_articulations, articulation.num_joints, device=device)
    friction = torch.rand(num_articulations, articulation.num_joints, device=device)

    # Guarantee that the dynamic friction is not greater than the static friction
    dynamic_friction = torch.min(dynamic_friction, friction)

    # The static friction must be set first to be sure the dynamic friction is not greater than static
    # when both are set.
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=friction,
        joint_dynamic_friction_coeff=dynamic_friction,
        joint_viscous_friction_coeff=viscous_friction,
    )
    articulation.write_data_to_sim()
    sim.step()
    articulation.update(sim.cfg.dt)

    friction_props_from_sim = _read_binding_to_torch(articulation, TT.DOF_FRICTION_PROPERTIES, "cpu")
    joint_friction_coeff_sim = friction_props_from_sim[:, :, 0]
    joint_dynamic_friction_coeff_sim = friction_props_from_sim[:, :, 1]
    joint_viscous_friction_coeff_sim = friction_props_from_sim[:, :, 2]
    assert torch.allclose(joint_dynamic_friction_coeff_sim, dynamic_friction.cpu())
    assert torch.allclose(joint_viscous_friction_coeff_sim, viscous_friction.cpu())
    assert torch.allclose(joint_friction_coeff_sim, friction.cpu())

    # The public getters expose the same coefficients.
    torch.testing.assert_close(articulation.data.joint_friction_coeff.torch, friction)
    torch.testing.assert_close(articulation.data.joint_dynamic_friction_coeff.torch, dynamic_friction)
    torch.testing.assert_close(articulation.data.joint_viscous_friction_coeff.torch, viscous_friction)


@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_set_material_properties(sim, num_articulations, device, add_ground_plane):
    """Randomize per-shape articulation materials through the OVPhysX material event.

    OVPhysX exposes per-collision-shape material as the
    ``articulation_shape_friction_and_restitution`` tensor binding (shape ``[N, S, 3]`` =
    static friction, dynamic friction, restitution), addressed through the
    :class:`~isaaclab_ov.sim.views.OvPhysxView`. The binding is CPU-native, so the
    buffer lives in host memory.
    """
    from isaaclab.envs.mdp.events import _RandomizeRigidBodyMaterialOvPhysx  # noqa: PLC0415

    articulation_cfg = generate_articulation_cfg(articulation_type="panda")
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
    )

    # Play the simulator
    sim.reset()

    # The ranges exclude the asset's default material, so values inside them prove the write happened.
    static_range, dynamic_range, restitution_range = (1.5, 2.0), (1.2, 1.4), (0.6, 0.8)
    view = articulation.root_view
    materials_before = wp.to_torch(view.get_attribute(TT.SHAPE_FRICTION_AND_RESTITUTION)).clone()
    assert (materials_before[..., 0] < static_range[0]).all(), f"default material overlaps: {materials_before}"
    params = {
        "static_friction_range": static_range,
        "dynamic_friction_range": dynamic_range,
        "restitution_range": restitution_range,
        "num_buckets": 16,
    }
    asset_cfg = SimpleNamespace(body_ids=slice(None))
    env = SimpleNamespace()  # unused by the OVPhysX implementation
    randomize = _RandomizeRigidBodyMaterialOvPhysx(SimpleNamespace(params=params), env, articulation, asset_cfg)

    # Randomize only the last environment; the others keep their materials.
    randomize(env, torch.tensor([num_articulations - 1], device=device), *params.values(), asset_cfg)
    sim.step()
    articulation.update(sim.cfg.dt)

    materials = wp.to_torch(view.get_attribute(TT.SHAPE_FRICTION_AND_RESTITUTION))
    torch.testing.assert_close(materials[:-1], materials_before[:-1])
    eps = 1e-5
    for component, (lo, hi) in enumerate((static_range, dynamic_range, restitution_range)):
        values = materials[-1, :, component]
        assert ((values >= lo - eps) & (values <= hi + eps)).all(), values


@wp.kernel
def _occupy_stream_kernel(iterations: int, out: wp.array(dtype=wp.float32)):
    total = float(0.0)
    for i in range(iterations):
        total += wp.sin(float(i))
    out[0] = total


@pytest.mark.parametrize("device", ["cuda:0"])
def test_cpu_only_property_writes_wait_for_pinned_host_staging(sim, device):
    """Land CPU-only property writes while the device stream is busy.

    OVPhysX keeps joint and body properties on the host even for a GPU simulation, so the writers
    stage the device-resident data, environment indices, and masks through pinned host buffers
    before calling the CPU setter. Warp issues that device-to-host copy asynchronously on the
    device stream, so a long kernel queued ahead of the copy must not let the setter consume the
    previous contents of the pinned buffers.
    """
    articulation, _ = generate_articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(Path(__file__).parent / "data" / "articulation_ordering_branching.usda")
            ),
            actuators={},
        ),
        num_articulations=4,
        device=device,
    )
    sim.reset()
    num_envs, num_joints = articulation.num_instances, articulation.num_joints

    stiffness_before = _read_binding_to_torch(articulation, TT.DOF_STIFFNESS, device)
    damping_before = _read_binding_to_torch(articulation, TT.DOF_DAMPING, device)
    masses_before = _read_binding_to_torch(articulation, TT.BODY_MASS, device)
    joint_values = torch.arange(1, num_envs * num_joints + 1, device=device, dtype=torch.float32)
    joint_values = joint_values.reshape(num_envs, num_joints)
    env_ids = torch.tensor([1, 3], device=device)
    env_mask = wp.array([False, True, False, True], dtype=wp.bool, device=device)
    scratch = wp.zeros(1, dtype=wp.float32, device=device)

    def occupy_device_stream(iterations: int = 500_000):
        wp.launch(_occupy_stream_kernel, dim=1, inputs=[iterations], outputs=[scratch], device=device)

    # Warm up every kernel involved so that module compilation cannot drain the stream
    # between the writes below and the kernel that keeps it busy.
    occupy_device_stream(iterations=1)
    articulation.write_joint_stiffness_to_sim_index(stiffness=stiffness_before[env_ids], env_ids=env_ids)
    articulation.write_joint_damping_to_sim_mask(damping=damping_before, env_mask=env_mask)
    articulation.set_masses_mask(masses=masses_before, env_mask=env_mask)
    wp.synchronize_device(device)

    occupy_device_stream()
    articulation.write_joint_stiffness_to_sim_index(stiffness=joint_values[env_ids], env_ids=env_ids)
    occupy_device_stream()
    articulation.write_joint_damping_to_sim_mask(damping=joint_values, env_mask=env_mask)
    occupy_device_stream()
    articulation.set_masses_mask(masses=masses_before * 2.0, env_mask=env_mask)
    wp.synchronize_device(device)

    expected_stiffness = stiffness_before.clone()
    expected_stiffness[env_ids] = joint_values[env_ids]
    expected_damping = damping_before.clone()
    expected_damping[env_ids] = joint_values[env_ids]
    expected_masses = masses_before.clone()
    expected_masses[env_ids] = masses_before[env_ids] * 2.0
    torch.testing.assert_close(_read_binding_to_torch(articulation, TT.DOF_STIFFNESS, device), expected_stiffness)
    torch.testing.assert_close(_read_binding_to_torch(articulation, TT.DOF_DAMPING, device), expected_damping)
    torch.testing.assert_close(_read_binding_to_torch(articulation, TT.BODY_MASS, device), expected_masses)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
