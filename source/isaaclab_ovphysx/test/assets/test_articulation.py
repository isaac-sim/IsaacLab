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
:attr:`~isaaclab_ovphysx.assets.Articulation.root_view`, an
:class:`~isaaclab_ovphysx.sim.views.OvPhysxView` over the per-tensor-type bindings
(``root_view.get_attribute(tensor_type)`` /
:meth:`~isaaclab_ovphysx.assets.Articulation._get_binding`), and the public setters
(:meth:`set_masses_index`, :meth:`set_coms_index`, :meth:`set_inertias_index`).
Reads use the data-class properties (``cube_object.data.body_mass``,
``body_inertia``, ``body_com_pose_b``).

Process-global device lock
--------------------------

The OVPhysX runtime fixes device mode (CPU vs GPU) when the process creates
its first ``ovphysx.PhysX`` instance and cannot switch it without a process
restart. :class:`~isaaclab_ovphysx.physics.OvPhysxManager` tracks
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

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import warp as wp

from pxr import Usd, UsdGeom, UsdPhysics

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

from isaaclab_ovphysx import tensor_types as TT  # noqa: E402
from isaaclab_ovphysx.assets import Articulation  # noqa: E402
from isaaclab_ovphysx.assets.articulation.articulation_data import ArticulationData  # noqa: E402
from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab.utils.string as string_utils  # noqa: E402
from isaaclab.actuators import ActuatorBase, IdealPDActuatorCfg, ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, get_articulation_name_ordering  # noqa: E402
from isaaclab.assets.articulation import ordering_kernels  # noqa: E402
from isaaclab.envs.mdp.terminations import joint_effort_out_of_limit  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.version import get_isaac_sim_version, has_kit  # noqa: E402
from isaaclab.utils.warp.launch_cache import _WarpLaunchCache  # noqa: E402

##
# Pre-defined configs
##
from isaaclab_assets import ANYMAL_C_CFG, FRANKA_PANDA_CFG, SHADOW_HAND_CFG  # isort:skip

wp.init()

pytestmark = pytest.mark.device_split


_OMNI_PHYSX_SCHEMAS_GAP_REASON = (
    "Schema-level fixed-joint creation in :mod:`isaaclab.sim.schemas` imports the Kit-only "
    "``omni.physx.scripts.utils`` module, which is not shipped by the ovphysx wheel."
)

_SPATIAL_TENDON_OVSTAGE_GAP_REASON = (
    "OVPhysX 0.5.9 segfaults while attaching OVStage scenes containing spatial tendon schemas."
)


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


def test_generalized_dynamics_reorder_uses_public_joint_order():
    """OVPhysX dynamics reads should gather both matrix joint axes into public order."""

    class Buffer:
        def __init__(self):
            self.data = wp.zeros((1, 2, 2), dtype=wp.float32, device="cpu")
            self.timestamp = -1.0

    data = ArticulationData.__new__(ArticulationData)
    data.device = "cpu"
    data._sim_timestamp = 1.0
    data._read_launch_cache = _WarpLaunchCache("cpu")
    data.joint_ordering = object()
    data._jacobian_joint_user_to_backend = wp.array([1, 0], dtype=wp.int32, device="cpu")
    data._joint_dof_signs = wp.ones(2, dtype=wp.int32, device="cpu")
    data._has_reversed_joints = False
    data._num_base_dofs = 0

    backend_values = wp.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=wp.float32, device="cpu")
    backend_buffer = wp.zeros_like(backend_values)
    buffer = Buffer()

    def read_binding(tensor_type, dst):
        dst.assign(backend_values)

    data._binding_read = read_binding
    data._refresh_generalized_dynamics_buffer(
        buffer,
        backend_buffer,
        TT.MASS_MATRIX,
        ordering_kernels.reorder_mass_matrix_backend_to_user,
    )

    torch.testing.assert_close(
        wp.to_torch(buffer.data),
        torch.tensor([[[4.0, 3.0], [2.0, 1.0]]]),
    )
    assert buffer.timestamp == 1.0


def _read_binding_to_torch(articulation: Articulation, tensor_type: int, device: str | torch.device) -> torch.Tensor:
    """Read an OVPhysX attribute into a torch tensor on *device*.

    Test-side adapter for the verbatim PhysX mirror.  PhysX cross-checks the
    data class against the simulation via ``articulation.root_view.get_X()``
    accessors; on OVPhysX we go through the equivalent
    :meth:`~isaaclab_ovphysx.sim.views.OvPhysxView.get_attribute`, which returns a
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
    gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, dt=dt, gravity=gravity)
    return build_simulation_context(device=device, sim_cfg=sim_cfg, **kwargs)


def generate_articulation_cfg(
    articulation_type: str,
    stiffness: float | None = 10.0,
    damping: float | None = 2.0,
    velocity_limit: float | None = None,
    effort_limit: float | None = None,
    velocity_limit_sim: float | None = None,
    effort_limit_sim: float | None = None,
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
        velocity_limit: Velocity limit for the actuators. Only currently used for "single_joint_implicit"
            and "single_joint_explicit".
        effort_limit: Effort limit for the actuators. Only currently used for "single_joint_implicit"
            and "single_joint_explicit".
        velocity_limit_sim: Velocity limit for the actuators (set into the simulation).
            Only currently used for "single_joint_implicit" and "single_joint_explicit".
        effort_limit_sim: Effort limit for the actuators (set into the simulation).
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
        articulation_cfg = SHADOW_HAND_CFG
    elif articulation_type == "single_joint_implicit":
        articulation_cfg = ArticulationCfg(
            # we set 80.0 default for max force because default in USD is 10e10 which makes testing annoying.
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
                joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_effort=80.0, max_velocity=5.0),
            ),
            actuators={
                "joint": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit_sim=effort_limit_sim,
                    velocity_limit_sim=velocity_limit_sim,
                    effort_limit=effort_limit,
                    velocity_limit=velocity_limit,
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
                joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_effort=80.0, max_velocity=5.0),
            ),
            actuators={
                "joint": IdealPDActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit_sim=effort_limit_sim,
                    velocity_limit_sim=velocity_limit_sim,
                    effort_limit=effort_limit,
                    velocity_limit=velocity_limit,
                    stiffness=0.0,
                    damping=10.0,
                ),
            },
        )
    elif articulation_type == "spatial_tendon_test_asset":
        # we set 80.0 default for max force because default in USD is 10e10 which makes testing annoying.
        articulation_cfg = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/IsaacLab/Tests/spatial_tendons.usd",
            ),
            actuators={
                "joint": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=2000.0,
                    damping=100.0,
                ),
            },
        )
    else:
        raise ValueError(
            f"Invalid articulation type: {articulation_type}, valid options are 'humanoid', 'panda', 'anymal',"
            " 'shadow_hand', 'single_joint_implicit', 'single_joint_explicit' or 'spatial_tendon_test_asset'."
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
    articulation = Articulation(articulation_cfg.replace(prim_path="/World/Env_.*/Robot"))

    return articulation, translations


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
    with _ovphysx_sim_context(
        device=device, auto_add_lighting=True, gravity_enabled=gravity_enabled, add_ground_plane=add_ground_plane
    ) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("gravity_enabled", [False])
def test_write_joint_state_accepts_int64_selector(sim, device, gravity_enabled):
    """Write joint state with int64 selectors."""
    articulation_cfg = generate_articulation_cfg(articulation_type="panda")
    articulation, _ = generate_articulation(articulation_cfg, 2, device=device)
    sim.reset()
    assert articulation.num_joints >= 2

    env_ids = torch.tensor([1, 0], dtype=torch.int64, device=device)
    joint_ids = torch.tensor([articulation.num_joints - 1, 0], dtype=torch.int64, device=device)
    position = torch.tensor([[0.21, 0.11], [0.22, 0.12]], device=device)
    velocity = torch.tensor([[1.21, 1.11], [1.22, 1.12]], device=device)
    expected_position = articulation.data.joint_pos.torch.clone()
    expected_velocity = articulation.data.joint_vel.torch.clone()

    articulation.write_joint_state_to_sim_index(
        position=position, velocity=velocity, env_ids=env_ids, joint_ids=joint_ids
    )

    expected_position[env_ids[:, None], joint_ids[None, :]] = position
    expected_velocity[env_ids[:, None], joint_ids[None, :]] = velocity
    torch.testing.assert_close(articulation.data.joint_pos.torch, expected_position)
    torch.testing.assert_close(articulation.data.joint_vel.torch, expected_velocity)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("gravity_enabled", [False])
def test_reversed_joint_dynamics_use_public_joint_basis(sim, device, gravity_enabled):
    """Keep dynamics tensors consistent with public joint velocity."""
    articulation = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(Path(__file__).parent / "data" / "articulation_ordering_branching.usda")
            ),
            actuators={},
        )
    )
    UsdPhysics.FixedJoint.Define(sim.stage, "/World/Robot/fixed_root").GetBody1Rel().SetTargets(["/World/Robot/base"])
    joint = UsdPhysics.RevoluteJoint.Get(sim.stage, "/World/Robot/left_elbow")
    body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
    joint.GetBody0Rel().SetTargets(body1)
    joint.GetBody1Rel().SetTargets(body0)
    sim.reset()

    velocity = torch.zeros((1, articulation.num_joints), device=device)
    velocity[:, articulation.find_joints("left_shoulder")[0][0]] = 0.4
    velocity[:, articulation.find_joints("left_elbow")[0][0]] = 0.7
    articulation.write_joint_velocity_to_sim_index(velocity=velocity)
    sim.step()
    articulation.update(sim.cfg.dt)

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


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_live_anymal_c_manual_joint_ordering_preserves_unselected_backend_state(sim, num_articulations, device):
    """Test that a partial ordered write preserves every unselected backend joint."""
    articulation_cfg = generate_articulation_cfg("anymal").replace(
        joint_ordering=tuple(reversed(ANYMAL_C_PHYSX_JOINT_NAMES))
    )
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()

    joint_ordering = articulation.joint_ordering
    assert joint_ordering is not None
    backend_seed = torch.arange(1, articulation.num_joints + 1, dtype=torch.float32, device=device).reshape(1, -1)
    backend_seed *= 0.001
    articulation.root_view.set_attribute(TT.DOF_POSITION, wp.from_torch(backend_seed))
    backend_before = wp.to_torch(articulation.root_view.get_attribute(TT.DOF_POSITION)).clone()
    torch.testing.assert_close(backend_before, backend_seed, rtol=0.0, atol=0.0)
    backend_joint_id = joint_ordering.user_to_backend_indices[0]
    selected_value = backend_before[0, backend_joint_id] + 0.001

    articulation.write_joint_position_to_sim_index(
        position=selected_value.reshape(1, 1),
        env_ids=wp.array([0], dtype=wp.int32, device=device),
        joint_ids=wp.array([0], dtype=wp.int32, device=device),
    )

    backend_after = wp.to_torch(articulation.root_view.get_attribute(TT.DOF_POSITION)).clone()
    expected = backend_before.clone()
    expected[0, backend_joint_id] = selected_value
    torch.testing.assert_close(backend_after, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_live_anymal_c_manual_joint_ordering_reorders_joint_targets(sim, device):
    """Write nonidentity-ordered joint targets into their intended backend columns."""
    backend_joint_names = ANYMAL_C_PHYSX_JOINT_NAMES
    joint_ordering = (*backend_joint_names[1:], backend_joint_names[0])
    articulation_cfg = generate_articulation_cfg("anymal").replace(
        joint_ordering=joint_ordering,
        actuators={"legs": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=2.0)},
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
    articulation.set_joint_position_target_index(target=position_target)
    articulation.set_joint_velocity_target_index(target=velocity_target)
    articulation.write_data_to_sim()

    backend_position_target = _read_binding_to_torch(articulation, TT.DOF_POSITION_TARGET, device)
    backend_velocity_target = _read_binding_to_torch(articulation, TT.DOF_VELOCITY_TARGET, device)
    torch.testing.assert_close(backend_position_target, position_target[:, backend_to_user])
    torch.testing.assert_close(backend_velocity_target, velocity_target[:, backend_to_user])


@pytest.mark.parametrize("device", ["cpu"])
def test_live_anymal_c_manual_joint_ordering_reorders_joint_friction_properties(sim, device):
    """Read every friction component from backend order into public joint order."""
    backend_joint_names = ANYMAL_C_PHYSX_JOINT_NAMES
    joint_ordering = (*backend_joint_names[1:], backend_joint_names[0])
    articulation_cfg = generate_articulation_cfg("anymal").replace(joint_ordering=joint_ordering)
    articulation, _ = generate_articulation(articulation_cfg, 1, device=device)
    sim.reset()

    ordering = articulation.joint_ordering
    assert ordering is not None
    user_to_backend = torch.as_tensor(ordering.user_to_backend_indices, dtype=torch.long, device=device)
    joint_index = torch.arange(articulation.num_joints, dtype=torch.float32, device=device).unsqueeze(0)
    backend_friction = torch.stack(
        (20.0 + joint_index, 10.0 + 0.5 * joint_index, 1.0 + 0.25 * joint_index),
        dim=-1,
    )
    articulation.root_view.set_attribute(
        TT.DOF_FRICTION_PROPERTIES,
        wp.from_torch(backend_friction.contiguous()),
    )
    articulation.data._joint_friction_props_buf.timestamp = -1.0
    articulation.data._joint_friction_props_backend.timestamp = -1.0

    expected = backend_friction[:, user_to_backend]
    torch.testing.assert_close(articulation.data.joint_friction_coeff.torch, expected[..., 0])
    torch.testing.assert_close(articulation.data.joint_dynamic_friction_coeff.torch, expected[..., 1])
    torch.testing.assert_close(articulation.data.joint_viscous_friction_coeff.torch, expected[..., 2])


@pytest.mark.parametrize("selection", ["full", "partial"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_reversed_joint_ordering_joint_state_index_writes_backend_order(sim, selection, device):
    """Write full and partial indexed joint state through a nonidentity public joint axis."""
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

    public_pos_before = articulation.data.joint_pos.torch.clone()
    public_vel_before = articulation.data.joint_vel.torch.clone()
    torch.testing.assert_close(public_pos_before, backend_pos_before[:, user_to_backend])
    torch.testing.assert_close(public_vel_before, backend_vel_before[:, user_to_backend])

    if selection == "full":
        position = torch.arange(2 * num_joints, dtype=torch.float32, device=device).reshape(2, num_joints) + 200.0
        velocity = position + 100.0
        env_ids = None
        joint_ids = None
        expected_public_pos = position
        expected_public_vel = velocity
        expected_backend_pos = position[:, backend_to_user]
        expected_backend_vel = velocity[:, backend_to_user]
    else:
        position = torch.tensor([[201.0, 203.0]], device=device)
        velocity = torch.tensor([[301.0, 303.0]], device=device)
        env_ids = [1]
        joint_ids = [0, 2]
        expected_public_pos = public_pos_before.clone()
        expected_public_vel = public_vel_before.clone()
        expected_public_pos[1, joint_ids] = position[0]
        expected_public_vel[1, joint_ids] = velocity[0]
        expected_backend_pos = backend_pos_before.clone()
        expected_backend_vel = backend_vel_before.clone()
        backend_joint_ids = user_to_backend[joint_ids]
        expected_backend_pos[1, backend_joint_ids] = position[0]
        expected_backend_vel[1, backend_joint_ids] = velocity[0]

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
        expected_backend_pos,
    )
    torch.testing.assert_close(
        _read_binding_to_torch(articulation, TT.DOF_VELOCITY, device),
        expected_backend_vel,
    )


@pytest.mark.parametrize("num_articulations", [1])
# COM pose is a CPU-resident OVPhysX binding (``_CPU_ONLY_TYPES``) even on a GPU sim, and this test
# restores it via the low-level ``root_view.set_attribute`` which forbids cross-device staging, so it
# is inherently CPU-only (aligning it to ``cuda:0`` would fail on the CPU-native COM binding).
@pytest.mark.parametrize("device", ["cpu"])
def test_live_panda_manual_body_ordering_preserves_unselected_coms(sim, num_articulations, device):
    """Test that a partial ordered COM write preserves every unselected backend body."""
    articulation_cfg = FRANKA_PANDA_CFG.replace(body_ordering=PANDA_ROOT_PRESERVING_REVERSED_BODY_NAMES)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()

    body_ordering = articulation.body_ordering
    assert body_ordering is not None
    backend_before = _read_binding_to_torch(articulation, TT.BODY_COM_POSE, device).clone()
    assert torch.unique(backend_before[0], dim=0).shape[0] > 1

    articulation.data._body_com_pose_b.timestamp = -1.0
    backend_staging = articulation.data._body_com_pose_b_backend
    if backend_staging is not None:
        backend_staging.timestamp = -1.0

    public_body_id = 1
    backend_body_id = body_ordering.user_to_backend_indices[public_body_id]
    assert backend_body_id != public_body_id
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


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_reversed_body_ordering_wrench_composes_from_backend_pose_without_shadow_refresh(
    sim, num_articulations, device
):
    """Reversed body ordering: an external wrench composes from the backend-order link pose and
    ``write_data_to_sim`` no longer refreshes the public ``body_link_pose_w`` shadow.
    """
    articulation_cfg = FRANKA_PANDA_CFG.replace(body_ordering=PANDA_ROOT_PRESERVING_REVERSED_BODY_NAMES)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()

    body_ordering = articulation.body_ordering
    assert body_ordering is not None

    # Apply a body-frame wrench to a single named body (public order).
    body_ids, _ = articulation.find_bodies("panda_hand")
    public_body_id = body_ids[0]
    backend_body_id = int(body_ordering.user_to_backend_indices[public_body_id])
    assert backend_body_id != public_body_id  # exercises the reorder

    force_b = torch.zeros(articulation.num_instances, len(body_ids), 3, device=device)
    torque_b = torch.zeros(articulation.num_instances, len(body_ids), 3, device=device)
    force_b[..., 0], force_b[..., 1], force_b[..., 2] = 3.0, -5.0, 7.0
    torque_b[..., 0], torque_b[..., 1], torque_b[..., 2] = 0.5, -1.5, 2.5
    articulation.permanent_wrench_composer.set_forces_and_torques_index(
        forces=force_b, torques=torque_b, body_ids=body_ids
    )

    # Step once so the link poses are non-trivial (rotated), giving the quaternion rotation teeth.
    articulation.set_joint_position_target_index(target=articulation.data.default_joint_pos.torch.clone())
    articulation.write_data_to_sim()
    sim.step()
    articulation.update(sim.cfg.dt)

    # write_data_to_sim must NOT advance the public body_link_pose_w shadow timestamp: the wrench
    # path now reads the backend-order pose buffer instead of refreshing the public shadow.
    shadow_ts_before = articulation.data._body_link_pose_w.timestamp
    articulation.write_data_to_sim()
    assert articulation.data._body_link_pose_w.timestamp == shadow_ts_before

    # The wrench buffer is in backend order; the world-frame wrench must match the one composed
    # from the SAME physical body's pose read via the public (user-order) shadow.
    wrench_buf = wp.to_torch(articulation._wrench_buf).to(device)
    pose = articulation.data.body_link_pose_w.torch[0, public_body_id]  # user order, [pos(3), quat_xyzw(4)]
    quat_xyzw = pose[3:7]
    expected_force_w = math_utils.quat_apply(quat_xyzw, force_b[0, 0])
    expected_torque_w = math_utils.quat_apply(quat_xyzw, torque_b[0, 0])
    torch.testing.assert_close(wrench_buf[0, backend_body_id, 0:3], expected_force_w, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(wrench_buf[0, backend_body_id, 3:6], expected_torque_w, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(wrench_buf[0, backend_body_id, 6:9], pose[0:3], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_reversed_joint_ordering_joint_acc_matches_canonicalized_finite_difference(sim, num_articulations, device):
    """Reversed joint ordering: the fused ``joint_acc`` finite difference reads the backend-order
    velocity source and equals the identity-order acceleration permuted into public order.
    """
    articulation_cfg = generate_articulation_cfg("anymal").replace(
        joint_ordering=tuple(reversed(ANYMAL_C_PHYSX_JOINT_NAMES))
    )
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    data = articulation.data

    joint_ordering = articulation.joint_ordering
    assert joint_ordering is not None
    num_joints = articulation.num_joints

    user_to_backend = torch.as_tensor(
        [int(joint_ordering.user_to_backend_indices[u]) for u in range(num_joints)],
        dtype=torch.long,
        device=device,
    )

    # Controlled, non-uniform finite-difference scenario in backend order (so a wrong permutation
    # in the kernel would produce a different result -- i.e. the test has teeth).
    cur_vel_backend = (torch.arange(1, num_joints + 1, dtype=torch.float32, device=device) * 0.1).reshape(1, -1)
    prev_vel_backend = (torch.arange(1, num_joints + 1, dtype=torch.float32, device=device) * -0.03).reshape(1, -1)

    # Push the current velocity into the backend DOF_VELOCITY binding (backend order).
    articulation.root_view.set_attribute(TT.DOF_VELOCITY, wp.from_torch(cur_vel_backend.contiguous()))

    # ``_previous_joint_vel`` is stored in PUBLIC order: prev_user[u] = prev_backend[map[u]].
    prev_vel_user = prev_vel_backend[:, user_to_backend].contiguous()
    data._previous_joint_vel.assign(wp.from_torch(prev_vel_user))

    # Force a stale finite-difference state with a known dt so the ordered branch recomputes, and a
    # stale backend velocity staging so it re-reads the value we just set.
    dt = 0.02
    data._joint_acc.timestamp = data._sim_timestamp - dt
    data._joint_vel_backend.timestamp = -1.0

    joint_acc_user = data.joint_acc.torch.clone()

    # Expected identity-order acceleration, then permuted into public order.
    acc_backend = (cur_vel_backend - prev_vel_backend) / dt
    expected_user = acc_backend[:, user_to_backend]
    torch.testing.assert_close(joint_acc_user, expected_user, rtol=1e-5, atol=1e-6)


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
def test_branching_fixture_mjwarp_ordering_reorders_ovphysx_to_dfs(sim, device):
    """Resolve depth-first MJWarp order cross-backend for ``joint_ordering="mjwarp"`` on OVPhysX.

    Live coverage for the cross-backend symbolic-convention path: OVPhysX is a
    PhysX-family backend (native breadth-first order), so requesting ``mjwarp`` triggers a temporary
    Newton USD discovery of the depth-first order and reorders the public joint/body axes to it. The
    MJWarp/DFS ground truth is the same tuple isaaclab_newton's
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
    assert articulation.joint_ordering is not None
    assert articulation.body_ordering is not None


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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_articulation_dynamics_refresh_after_same_timestamp_model_writes(sim, device):
    """Refresh computed dynamics after model-property writes without advancing simulation time."""
    articulation, _ = generate_articulation(generate_articulation_cfg("panda"), 1, device=device)
    sim.reset()

    data = articulation.data
    num_jacobian_bodies = articulation.num_bodies - 1
    num_generalized_dofs = articulation.num_joints

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
    """Gather computed dynamics into MJWarp body and joint order."""
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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_articulation_dynamics_preserve_floating_base_columns_during_joint_reordering(sim, device):
    """Keep floating-base columns leading while gathering actuated-joint axes."""
    articulation_cfg = generate_articulation_cfg("anymal").replace(
        joint_ordering=tuple(reversed(ANYMAL_C_PHYSX_JOINT_NAMES))
    )
    articulation, _ = generate_articulation(articulation_cfg, 1, device=device)
    sim.reset()

    joint_ordering = articulation.joint_ordering
    assert joint_ordering is not None
    joint_user_to_backend = torch.as_tensor(joint_ordering.user_to_backend_indices, dtype=torch.long, device=device)
    generalized_user_to_backend = torch.cat(
        (
            torch.arange(6, device=device),
            6 + joint_user_to_backend,
        )
    )
    raw_jacobian = _read_binding_to_torch(articulation, TT.JACOBIAN, device).reshape(
        1, articulation.num_bodies, 6, articulation.num_joints + 6
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


@pytest.mark.parametrize("num_articulations", [1, 2])
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


@pytest.mark.parametrize("num_articulations", [1, 2])
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


@pytest.mark.parametrize("num_articulations", [1, 2])
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


@pytest.mark.parametrize("num_articulations", [1, 2])
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


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
def test_initialization_hand_with_tendons(sim, num_articulations, device):
    """Test initialization for fixed base articulated hand with tendons.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is fixed base
    3. All buffers have correct shapes
    4. The articulation can be simulated

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="shadow_hand")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)

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
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 24)
    assert articulation.data.body_mass.torch.shape == (num_articulations, articulation.num_bodies)
    assert articulation.data.body_inertia.torch.shape == (num_articulations, articulation.num_bodies, 9)

    # Cross-check binding shapes against cached counts.  See the equivalent
    # block in test_initialization_fixed_base_single_joint for why the verbatim
    # PhysX ``root_view.max_dofs == shared_metatype.dof_count`` identity is
    # replaced with binding-shape checks on OVPhysX.
    for tt in (TT.DOF_POSITION, TT.DOF_VELOCITY, TT.DOF_STIFFNESS):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_joints
    for tt in (TT.BODY_MASS, TT.BODY_COM_POSE):
        binding = articulation.root_view.try_binding_for(tt)
        if binding is not None:
            assert binding.shape[1] == articulation.num_bodies
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


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.xfail(reason=_OMNI_PHYSX_SCHEMAS_GAP_REASON, strict=False)
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
    articulation_cfg.spawn.articulation_props.fix_root_link = True
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

    assert sim_utils.apply_articulation_root_properties(asset_path, [], stage, fix_root_link=True)
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


@pytest.mark.parametrize("num_articulations", [1, 2])
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
    articulation_cfg = generate_articulation_cfg(articulation_type="panda")
    # Unfix root link by making it non-kinematic
    articulation_cfg.spawn.articulation_props.fix_root_link = False
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


@pytest.mark.parametrize("num_articulations", [1, 2])
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


@pytest.mark.parametrize("num_articulations", [1, 2])
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


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("add_ground_plane", [True])
def test_joint_effort_limits(sim, num_articulations, device, add_ground_plane):
    """Validate joint effort limits via joint_effort_out_of_limit()."""
    # Create articulation
    articulation_cfg = generate_articulation_cfg(articulation_type="panda")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)

    # Minimal env wrapper exposing scene["robot"]
    class _Env:
        def __init__(self, art):
            self.scene = {"robot": art}

    env = _Env(articulation)
    robot_all = SceneEntityCfg(name="robot")

    sim.reset()
    assert articulation.is_initialized

    # Case A: no clipping → should NOT terminate
    articulation._data.computed_torque.torch.zero_()
    articulation._data.applied_torque.torch.zero_()
    out = joint_effort_out_of_limit(env, robot_all)  # [N]
    assert torch.all(~out)

    # Case B: simulate clipping → should terminate
    articulation._data.computed_torque.torch.fill_(100.0)  # pretend controller commanded 100
    articulation._data.applied_torque.torch.fill_(50.0)  # pretend actuator clipped to 50
    out = joint_effort_out_of_limit(env, robot_all)  # [N]
    assert torch.all(out)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
def test_external_force_buffer(sim, num_articulations, device):
    """Test if external force buffer correctly updates in the force value is zero case.

    This test verifies that:
    1. External forces can be applied correctly
    2. Force buffers are updated properly
    3. Zero forces are handled correctly

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # play the simulator
    sim.reset()

    # find bodies to apply the force
    body_ids, _ = articulation.find_bodies("base")

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

    # perform simulation
    for step in range(5):
        # initiate force tensor
        external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)

        if step == 0 or step == 3:
            # set a non-zero force
            force = 1
        else:
            # set a zero force
            force = 0

        # set force value
        external_wrench_b[:, :, 0] = force
        external_wrench_b[:, :, 3] = force

        # apply force
        articulation.permanent_wrench_composer.set_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            body_ids=body_ids,
        )

        # check if the articulation's force and torque buffers are correctly updated
        for i in range(num_articulations):
            assert articulation.permanent_wrench_composer.composed_force.torch[i, 0, 0].item() == force
            assert articulation.permanent_wrench_composer.composed_torque.torch[i, 0, 0].item() == force

        # Check if the instantaneous wrench is correctly added to the permanent wrench
        articulation.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=external_wrench_b[..., :3],
            torques=external_wrench_b[..., 3:],
            body_ids=body_ids,
        )

        # apply action to the articulation
        articulation.set_joint_position_target_index(target=articulation.data.default_joint_pos.torch.clone())
        articulation.write_data_to_sim()

        # perform step
        sim.step()

        # update buffers
        articulation.update(sim.cfg.dt)


@pytest.mark.parametrize("num_articulations", [1, 2])
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


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
def test_external_force_on_single_body_at_position(sim, num_articulations, device):
    """Test application of external force on the base of the articulation at a given position.

    This test verifies that:
    1. External forces can be applied to specific bodies at a given position
    2. External forces can be applied to specific bodies in the global frame
    3. External forces are calculated and composed correctly
    4. The forces affect the articulation's motion correctly
    5. The articulation responds to the forces as expected

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


@pytest.mark.parametrize("num_articulations", [1, 2])
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


@pytest.mark.parametrize("num_articulations", [1, 2])
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
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

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
            # since there is a moment applied on the articulation, the articulation should rotate
            assert torch.abs(articulation.data.root_ang_vel_w.torch[i, 2]).item() > 0.1


@pytest.mark.parametrize("num_articulations", [1, 2])
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


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("add_ground_plane", [True])
def test_setting_gains_from_cfg(sim, num_articulations, device, add_ground_plane):
    """Test that gains are loaded from the configuration correctly.

    This test verifies that:
    1. Gains are loaded correctly from configuration
    2. The gains match the expected values
    3. The gains are applied correctly to the actuators

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=sim.device
    )

    # Play sim
    sim.reset()

    # Expected gains
    expected_stiffness = torch.full(
        (articulation.num_instances, articulation.num_joints), 10.0, device=articulation.device
    )
    expected_damping = torch.full_like(expected_stiffness, 2.0)

    # Check that gains are loaded from USD file
    torch.testing.assert_close(articulation.actuators["body"].stiffness, expected_stiffness)
    torch.testing.assert_close(articulation.actuators["body"].damping, expected_damping)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
def test_setting_gains_from_cfg_dict(sim, num_articulations, device):
    """Test that gains are loaded from the configuration dictionary correctly.

    This test verifies that:
    1. Gains are loaded correctly from configuration dictionary
    2. The gains match the expected values
    3. The gains are applied correctly to the actuators

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=sim.device
    )
    # Play sim
    sim.reset()

    # Expected gains
    expected_stiffness = torch.full(
        (articulation.num_instances, articulation.num_joints), 10.0, device=articulation.device
    )
    expected_damping = torch.full_like(expected_stiffness, 2.0)

    # Check that gains are loaded from USD file
    torch.testing.assert_close(articulation.actuators["body"].stiffness, expected_stiffness)
    torch.testing.assert_close(articulation.actuators["body"].damping, expected_damping)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("vel_limit_sim", [1e5, None])
@pytest.mark.parametrize("vel_limit", [1e2, None])
@pytest.mark.parametrize("add_ground_plane", [False])
def test_setting_velocity_limit_implicit(sim, num_articulations, device, vel_limit_sim, vel_limit, add_ground_plane):
    """Test setting of velocity limit for implicit actuators.

    This test verifies that:
    1. The solver clamp ``velocity_limit_sim`` is applied to the simulation; when unset, the
       USD-authored value is kept
    2. The joint velocity limit ``velocity_limit`` is never pushed to the solver and keeps its
       configured value; when unset, it falls back to the solver clamp

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
        vel_limit_sim: The velocity limit to set in simulation
        vel_limit: The velocity limit to set in actuator
    """
    # create simulation
    articulation_cfg = generate_articulation_cfg(
        articulation_type="single_joint_implicit",
        velocity_limit_sim=vel_limit_sim,
        velocity_limit=vel_limit,
    )
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg,
        num_articulations=num_articulations,
        device=device,
    )
    # Play sim
    sim.reset()

    # read the values set into the simulation
    physx_vel_limit = _read_binding_to_torch(articulation, TT.DOF_MAX_VELOCITY, device)
    # check data buffer
    torch.testing.assert_close(articulation.data.joint_velocity_limits.torch, physx_vel_limit)
    # check actuator has simulation velocity limit
    torch.testing.assert_close(articulation.actuators["joint"].velocity_limit_sim, physx_vel_limit)

    # the solver clamp comes from velocity_limit_sim when set, otherwise the USD-authored value
    if vel_limit_sim is None:
        sim_limit = articulation_cfg.spawn.joint_drive_props.max_joint_velocity
    else:
        sim_limit = vel_limit_sim
    expected_velocity_limit = torch.full_like(physx_vel_limit, sim_limit)
    torch.testing.assert_close(physx_vel_limit, expected_velocity_limit)

    # the joint velocity limit keeps its configured value and is not pushed to the solver;
    # when unset it falls back to the solver clamp
    joint_limit = vel_limit if vel_limit is not None else sim_limit
    expected_joint_limit = torch.full_like(physx_vel_limit, joint_limit)
    torch.testing.assert_close(articulation.actuators["joint"].velocity_limit, expected_joint_limit)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("vel_limit_sim", [1e5, None])
@pytest.mark.parametrize("vel_limit", [1e2, None])
def test_setting_velocity_limit_explicit(sim, num_articulations, device, vel_limit_sim, vel_limit):
    """Test setting of velocity limit for explicit actuators."""
    articulation_cfg = generate_articulation_cfg(
        articulation_type="single_joint_explicit",
        velocity_limit_sim=vel_limit_sim,
        velocity_limit=vel_limit,
    )
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg,
        num_articulations=num_articulations,
        device=device,
    )
    # Play sim
    sim.reset()

    # collect limit init values
    physx_vel_limit = _read_binding_to_torch(articulation, TT.DOF_MAX_VELOCITY, device)
    actuator_vel_limit = articulation.actuators["joint"].velocity_limit
    actuator_vel_limit_sim = articulation.actuators["joint"].velocity_limit_sim

    # check data buffer for joint_velocity_limits_sim
    torch.testing.assert_close(articulation.data.joint_velocity_limits.torch, physx_vel_limit)
    # check actuator velocity_limit_sim is set to physx
    torch.testing.assert_close(actuator_vel_limit_sim, physx_vel_limit)

    if vel_limit is not None:
        expected_actuator_vel_limit = torch.full(
            (articulation.num_instances, articulation.num_joints),
            vel_limit,
            device=articulation.device,
        )
        # check actuator is set
        torch.testing.assert_close(actuator_vel_limit, expected_actuator_vel_limit)
        # check physx is not velocity_limit
        assert not torch.allclose(actuator_vel_limit, physx_vel_limit)
    else:
        # check actuator velocity_limit is the same as the PhysX default
        torch.testing.assert_close(actuator_vel_limit, physx_vel_limit)

    # simulation velocity limit is set to USD value unless user overrides
    if vel_limit_sim is not None:
        limit = vel_limit_sim
    else:
        limit = articulation_cfg.spawn.joint_drive_props.max_joint_velocity
    # check physx is set to expected value
    expected_vel_limit = torch.full_like(physx_vel_limit, limit)
    torch.testing.assert_close(physx_vel_limit, expected_vel_limit)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("effort_limit_sim", [1e5, None])
@pytest.mark.parametrize("effort_limit", [1e2, 80.0, None])
def test_setting_effort_limit_implicit(sim, num_articulations, device, effort_limit_sim, effort_limit):
    """Test setting of effort limit for implicit actuators.

    This test verifies the effort limit resolution logic for actuator models implemented in :class:`ActuatorBase`:
    - Case 1: If USD value == actuator config value: values match correctly
    - Case 2: If USD value != actuator config value: actuator config value is used
    - Case 3: If actuator config value is None: USD value is used as default
    """
    articulation_cfg = generate_articulation_cfg(
        articulation_type="single_joint_implicit",
        effort_limit_sim=effort_limit_sim,
        effort_limit=effort_limit,
    )
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg,
        num_articulations=num_articulations,
        device=device,
    )
    # Play sim
    if effort_limit_sim is not None and effort_limit is not None:
        with pytest.raises(ValueError):
            sim.reset()
        return
    sim.reset()

    # obtain the physx effort limits
    physx_effort_limit = _read_binding_to_torch(articulation, TT.DOF_MAX_FORCE, device)

    # check that the two are equivalent
    torch.testing.assert_close(
        articulation.actuators["joint"].effort_limit_sim,
        articulation.actuators["joint"].effort_limit,
    )
    torch.testing.assert_close(articulation.actuators["joint"].effort_limit_sim, physx_effort_limit)

    # decide the limit based on what is set
    if effort_limit_sim is None and effort_limit is None:
        limit = articulation_cfg.spawn.joint_drive_props.max_force
    elif effort_limit_sim is not None and effort_limit is None:
        limit = effort_limit_sim
    elif effort_limit_sim is None and effort_limit is not None:
        limit = effort_limit

    # check that the max force is what we set
    expected_effort_limit = torch.full_like(physx_effort_limit, limit)
    torch.testing.assert_close(physx_effort_limit, expected_effort_limit)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("effort_limit_sim", [1e5, None])
@pytest.mark.parametrize("effort_limit", [80.0, 1e2, None])
def test_setting_effort_limit_explicit(sim, num_articulations, device, effort_limit_sim, effort_limit):
    """Test setting of effort limit for explicit actuators.

    This test verifies the effort limit resolution logic for actuator models implemented in :class:`ActuatorBase`:
    - Case 1: If USD value == actuator config value: values match correctly
    - Case 2: If USD value != actuator config value: actuator config value is used
    - Case 3: If actuator config value is None: USD value is used as default

    """

    articulation_cfg = generate_articulation_cfg(
        articulation_type="single_joint_explicit",
        effort_limit_sim=effort_limit_sim,
        effort_limit=effort_limit,
    )
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg,
        num_articulations=num_articulations,
        device=device,
    )
    # Play sim
    sim.reset()

    # usd default effort limit is set to 80
    usd_default_effort_limit = 80.0

    # collect limit init values
    physx_effort_limit = _read_binding_to_torch(articulation, TT.DOF_MAX_FORCE, device)
    actuator_effort_limit = articulation.actuators["joint"].effort_limit
    actuator_effort_limit_sim = articulation.actuators["joint"].effort_limit_sim

    # check actuator effort_limit_sim is set to physx
    torch.testing.assert_close(actuator_effort_limit_sim, physx_effort_limit)

    if effort_limit is not None:
        expected_actuator_effort_limit = torch.full_like(actuator_effort_limit, effort_limit)
        # check actuator is set
        torch.testing.assert_close(actuator_effort_limit, expected_actuator_effort_limit)

        # check physx effort limit does not match the one explicit actuator has
        assert not (torch.allclose(actuator_effort_limit, physx_effort_limit))
    else:
        # When effort_limit is None, actuator should use USD default values
        expected_actuator_effort_limit = torch.full_like(physx_effort_limit, usd_default_effort_limit)
        torch.testing.assert_close(actuator_effort_limit, expected_actuator_effort_limit)

    # when using explicit actuators, the limits are set to high unless user overrides
    if effort_limit_sim is not None:
        limit = effort_limit_sim
    else:
        limit = ActuatorBase._DEFAULT_MAX_EFFORT_SIM  # type: ignore
    # check physx internal value matches the expected sim value
    expected_effort_limit = torch.full_like(physx_effort_limit, limit)
    torch.testing.assert_close(actuator_effort_limit_sim, expected_effort_limit)
    torch.testing.assert_close(physx_effort_limit, expected_effort_limit)


@pytest.mark.parametrize("num_articulations", [1, 2])
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

    if num_articulations > 1:
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


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
def test_write_root_velocity_invalidates_body_frame_cache(sim, num_articulations, device):
    """Writing root velocity refreshes cached body-frame root velocities before a step."""
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)

    sim.reset()
    for _ in range(3):
        sim.step()
        articulation.update(sim.cfg.dt)
    ang_before = articulation.data.root_ang_vel_b.torch.clone()

    new_vel = torch.zeros(num_articulations, 6, device=device)
    new_vel[:, :3] = torch.tensor([3.0, 0.0, 0.0], device=device)
    new_vel[:, 3:] = torch.tensor([0.0, 0.0, 5.0], device=device)
    articulation.write_root_velocity_to_sim_index(
        root_velocity=wp.from_torch(new_vel.contiguous(), dtype=wp.spatial_vectorf)
    )

    ang_after = articulation.data.root_ang_vel_b.torch
    torch.testing.assert_close(
        ang_after.norm(dim=-1),
        torch.full((num_articulations,), 5.0, device=device),
        atol=1e-3,
        rtol=1e-3,
    )
    assert not torch.allclose(ang_after, ang_before)


@pytest.mark.parametrize("num_articulations", [1, 2])
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

    # Check that current joint position is not the same as default joint position, meaning
    # the articulation moved. We can't check that it reached its desired joint position as the gains
    # are not properly tuned
    assert not torch.allclose(articulation.data.joint_pos.torch, joint_pos)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("with_offset", [True, False])
def test_body_root_state(sim, num_articulations, device, with_offset):
    """Test for reading the `body_state_w` property.

    This test verifies that:
    1. Body states can be read correctly
    2. States are correct with and without offsets
    3. States are consistent across different devices

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
        with_offset: Whether to test with offset
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
    if with_offset:
        offset = [0.5, 0.0, 0.0]
    else:
        offset = [0.0, 0.0, 0.0]

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
        root_link_pose_w = articulation.data.root_link_pose_w.torch
        root_link_vel_w = articulation.data.root_link_vel_w.torch
        root_com_pose_w = articulation.data.root_com_pose_w.torch
        root_com_vel_w = articulation.data.root_com_vel_w.torch
        body_link_pose_w = articulation.data.body_link_pose_w.torch
        body_link_vel_w = articulation.data.body_link_vel_w.torch
        body_com_pose_w = articulation.data.body_com_pose_w.torch
        body_com_vel_w = articulation.data.body_com_vel_w.torch

        if with_offset:
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
        else:
            # single joint center of masses are at link frames so they will be the same
            torch.testing.assert_close(root_link_pose_w, root_com_pose_w)
            torch.testing.assert_close(root_com_vel_w, root_link_vel_w)
            torch.testing.assert_close(body_link_pose_w, body_com_pose_w)
            torch.testing.assert_close(body_com_vel_w, body_link_vel_w)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.parametrize("state_location", ["com", "link"])
@pytest.mark.parametrize("gravity_enabled", [False])
def test_write_root_state(sim, num_articulations, device, with_offset, state_location, gravity_enabled):
    """Test the setters for root_state using both the link frame and center of mass as reference frame.

    This test verifies that:
    1. Root states can be written correctly
    2. States are correct with and without offsets
    3. States can be written for both COM and link frames
    4. States are consistent across different devices

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
        with_offset: Whether to test with offset
        state_location: Whether to test COM or link frame
    """
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)
    env_idx = torch.tensor([x for x in range(num_articulations)], device=device, dtype=torch.int32)

    # Play sim
    sim.reset()

    # change center of mass offset from link frame
    if with_offset:
        offset = torch.tensor([1.0, 0.0, 0.0]).repeat(num_articulations, 1, 1)
    else:
        offset = torch.tensor([0.0, 0.0, 0.0]).repeat(num_articulations, 1, 1)

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
        elif state_location == "link":
            torch.testing.assert_close(rand_state[..., :7], articulation.data.root_link_pose_w.torch)
            torch.testing.assert_close(rand_state[..., 7:], articulation.data.root_link_vel_w.torch)


@pytest.mark.parametrize("device", ["cpu"])
def test_body_com_pose_b_cache_and_set_coms_invalidation(sim, device):
    """Body-frame COM offsets stay cached and invalidate derived buffers after writes."""
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type="single_joint_implicit")
    articulation, _ = generate_articulation(articulation_cfg, 2, device=device)

    sim.reset()
    articulation.update(sim.cfg.dt)

    articulation.data.body_com_pose_b
    first_timestamp = articulation.data._body_com_pose_b.timestamp
    articulation.update(sim.cfg.dt)
    articulation.data.body_com_pose_b
    assert articulation.data._body_com_pose_b.timestamp == first_timestamp

    dependent_buffers = [
        ("root_com_pose_w", articulation.data._root_com_pose_w),
        ("root_com_vel_w", articulation.data._root_com_vel_w),
        ("root_link_vel_w", articulation.data._root_link_vel_w),
        ("body_com_pose_w", articulation.data._body_com_pose_w),
        ("body_com_vel_w", articulation.data._body_com_vel_w),
        ("body_link_vel_w", articulation.data._body_link_vel_w),
        ("root_link_lin_vel_b", articulation.data._root_link_lin_vel_b),
        ("root_link_ang_vel_b", articulation.data._root_link_ang_vel_b),
        ("root_com_lin_vel_b", articulation.data._root_com_lin_vel_b),
        ("root_com_ang_vel_b", articulation.data._root_com_ang_vel_b),
        ("root_state_w", articulation.data._root_state_w_buf),
        ("root_link_state_w", articulation.data._root_link_state_w_buf),
        ("root_com_state_w", articulation.data._root_com_state_w_buf),
        ("body_state_w", articulation.data._body_state_w_buf),
        ("body_link_state_w", articulation.data._body_link_state_w_buf),
        ("body_com_state_w", articulation.data._body_com_state_w_buf),
        ("body_com_jacobian_w", articulation.data._body_com_jacobian_w),
        ("mass_matrix", articulation.data._mass_matrix),
        ("gravity_compensation_forces", articulation.data._gravity_compensation_forces),
    ]
    for _, buffer in dependent_buffers:
        buffer.timestamp = articulation.data._sim_timestamp

    coms = wp.zeros((articulation.num_instances, articulation.num_bodies), dtype=wp.transformf, device=device)
    articulation.set_coms_index(coms=coms)

    assert articulation.data._body_com_pose_b.timestamp >= 0.0
    for name, buffer in dependent_buffers:
        assert buffer.timestamp < articulation.data._sim_timestamp, name


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


@pytest.mark.parametrize("device", ["cpu"])
def test_deprecated_joint_state_writer_delegates_to_index_writers(sim, device, mocker):
    """Keep the deprecated combined API as a thin composition of public writers."""
    articulation_cfg = generate_articulation_cfg(articulation_type="anymal")
    articulation, _ = generate_articulation(articulation_cfg, 1, device)
    sim.reset()

    position = torch.tensor([[0.1]], device=device)
    velocity = torch.tensor([[0.2]], device=device)
    position_writer = mocker.patch.object(articulation, "write_joint_position_to_sim_index")
    velocity_writer = mocker.patch.object(articulation, "write_joint_velocity_to_sim_index")

    with pytest.warns(DeprecationWarning):
        articulation.write_joint_state_to_sim(
            position=position,
            velocity=velocity,
            joint_ids=[0],
            env_ids=[0],
        )

    position_writer.assert_called_once_with(position=position, joint_ids=[0], env_ids=[0])
    velocity_writer.assert_called_once_with(velocity=velocity, joint_ids=[0], env_ids=[0])


@pytest.mark.parametrize("device", test_devices())
def test_write_joint_state_to_sim_index_partial(sim, device):
    """Test fused joint-state writes with partial environment and joint indices."""
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
    articulation.write_joint_state_to_sim_index(
        position=position, velocity=velocity, env_ids=[1], joint_ids=[0, 2], skip_forward=True
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


@pytest.mark.parametrize("num_articulations", [1, 2])
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
    torch.testing.assert_close(body_link_pose_w, articulation.data.body_link_pose_w.torch)
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
    torch.testing.assert_close(body_link_pose_w_fresh, body_link_pose_w)
    torch.testing.assert_close(body_com_pose_w, articulation.data.body_com_pose_w.torch)
    torch.testing.assert_close(body_vel_w, body_com_vel_w_fresh)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.skip(reason=_SPATIAL_TENDON_OVSTAGE_GAP_REASON)
def test_spatial_tendons(sim, num_articulations, device):
    """Test spatial tendons apis.
    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation has spatial tendons
    3. All buffers have correct shapes
    4. The articulation can be simulated
    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    # skip test if Isaac Sim version is less than 5.0
    if has_kit() and get_isaac_sim_version().major < 5:
        pytest.skip("Spatial tendons are not supported in Isaac Sim < 5.0. Please update to Isaac Sim 5.0 or later.")
        return
    articulation_cfg = generate_articulation_cfg(articulation_type="spatial_tendon_test_asset")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)

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
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 3)
    assert articulation.data.body_mass.torch.shape == (num_articulations, articulation.num_bodies)
    assert articulation.data.body_inertia.torch.shape == (num_articulations, articulation.num_bodies, 9)
    assert articulation.num_spatial_tendons == 1

    articulation.set_spatial_tendon_stiffness_index(stiffness=10.0)
    articulation.set_spatial_tendon_limit_stiffness_index(limit_stiffness=10.0)
    articulation.set_spatial_tendon_damping_index(damping=10.0)
    articulation.set_spatial_tendon_offset_index(offset=10.0)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)


@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
def test_write_joint_frictions_to_sim(sim, num_articulations, device, add_ground_plane):
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

    for _ in range(100):
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

    friction_props_from_sim = _read_binding_to_torch(articulation, TT.DOF_FRICTION_PROPERTIES, "cpu")
    joint_friction_coeff_sim = friction_props_from_sim[:, :, 0]
    joint_dynamic_friction_coeff_sim = friction_props_from_sim[:, :, 1]
    joint_viscous_friction_coeff_sim = friction_props_from_sim[:, :, 2]
    assert torch.allclose(joint_dynamic_friction_coeff_sim, dynamic_friction.cpu())
    assert torch.allclose(joint_viscous_friction_coeff_sim, viscous_friction.cpu())
    assert torch.allclose(joint_friction_coeff_sim, friction.cpu())

    # For Isaac Sim >= 5.0: also test the combined API that can set dynamic and viscous via
    # write_joint_friction_coefficient_to_sim; reset the sim to isolate this path.
    if has_kit() and get_isaac_sim_version().major >= 5:
        # Reset simulator to ensure a clean state for the alternative API path
        sim.reset()

        # Warm up a few steps to populate buffers
        for _ in range(100):
            sim.step()
            articulation.update(sim.cfg.dt)

        # New random coefficients
        dynamic_friction_2 = torch.rand(num_articulations, articulation.num_joints, device=device)
        viscous_friction_2 = torch.rand(num_articulations, articulation.num_joints, device=device)
        friction_2 = torch.rand(num_articulations, articulation.num_joints, device=device)

        # Guarantee that the dynamic friction is not greater than the static friction
        dynamic_friction_2 = torch.min(dynamic_friction_2, friction_2)

        # Use the combined setter to write all three at once
        articulation.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=friction_2,
            joint_dynamic_friction_coeff=dynamic_friction_2,
            joint_viscous_friction_coeff=viscous_friction_2,
        )
        articulation.write_data_to_sim()

        # Step to let sim ingest new params and refresh data buffers
        for _ in range(100):
            sim.step()
            articulation.update(sim.cfg.dt)

        friction_props_from_sim_2 = _read_binding_to_torch(articulation, TT.DOF_FRICTION_PROPERTIES, "cpu")
        joint_friction_coeff_sim_2 = friction_props_from_sim_2[:, :, 0]
        friction_dynamic_coef_sim_2 = friction_props_from_sim_2[:, :, 1]
        friction_viscous_coeff_sim_2 = friction_props_from_sim_2[:, :, 2]

        # Validate values propagated
        assert torch.allclose(friction_viscous_coeff_sim_2, viscous_friction_2.cpu())
        assert torch.allclose(friction_dynamic_coef_sim_2, dynamic_friction_2.cpu())
        assert torch.allclose(joint_friction_coeff_sim_2, friction_2.cpu())


@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("articulation_type", ["panda"])
def test_set_material_properties(sim, num_articulations, device, add_ground_plane, articulation_type):
    """Test getting and setting per-shape material properties (friction/restitution).

    OVPhysX exposes per-collision-shape material as the
    ``articulation_shape_friction_and_restitution`` tensor binding (shape ``[N, S, 3]`` =
    static friction, dynamic friction, restitution), addressed through the
    :class:`~isaaclab_ovphysx.sim.views.OvPhysxView`. The binding is device-resident, so the
    buffer lives on the simulation device. (The PhysX backend instead uses a dedicated
    ``root_view.get_material_properties`` / ``set_material_properties`` view API.)
    """
    if not hasattr(TT, "SHAPE_FRICTION_AND_RESTITUTION"):
        pytest.skip("ovphysx wheel does not expose the shape material tensor type")

    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
    )

    # Play the simulator
    sim.reset()

    view = articulation.root_view
    # Number of collision shapes per articulation, from the material binding's shape [N, S, 3].
    num_shapes = view.binding_for(TT.SHAPE_FRICTION_AND_RESTITUTION).shape[1]

    # Random material per shape: (static_friction, dynamic_friction, restitution), on the sim device.
    materials = torch.empty(num_articulations, num_shapes, 3, device=device).uniform_(0.0, 1.0)
    materials[..., 1] = torch.min(materials[..., 0], materials[..., 1])  # dynamic <= static

    # Set material properties through the view, then simulate.
    view.set_attribute(TT.SHAPE_FRICTION_AND_RESTITUTION, wp.from_torch(materials, dtype=wp.float32))
    sim.step()
    articulation.update(sim.cfg.dt)

    # Read back from the simulation and verify the round-trip.
    materials_check = wp.to_torch(view.get_attribute(TT.SHAPE_FRICTION_AND_RESTITUTION))
    torch.testing.assert_close(materials_check, materials)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
