# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from isaaclab.test.utils import DeviceScope, resolve_test_sim_device, test_devices
from isaaclab.test.utils.articulation_ordering import (
    BRANCHING_MJWARP_BODY_NAMES,
    BRANCHING_MJWARP_JOINT_NAMES,
    BRANCHING_PHYSX_BODY_NAMES,
    BRANCHING_PHYSX_JOINT_NAMES,
    PANDA_BODY_NAMES,
    PANDA_JOINT_NAMES,
    PANDA_ROOT_PRESERVING_REVERSED_BODY_NAMES,
)

HEADLESS = True

# launch omniverse app
simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

"""Rest everything follows."""

import sys
from pathlib import Path

import pytest
import torch
import warp as wp
from isaaclab_physx.assets import Articulation
from isaaclab_physx.sim.schemas import PhysxJointCfg

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, get_articulation_name_ordering
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs.mdp.terminations import joint_effort_out_of_limit
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import compute_pose_error, matrix_from_quat, quat_inv, subtract_frame_transforms
from isaaclab.utils.version import get_isaac_sim_version, has_kit

##
# Pre-defined configs
##
from isaaclab_assets import (  # isort:skip
    ANYMAL_C_CFG,
    FRANKA_PANDA_CFG,
    FRANKA_PANDA_HIGH_PD_CFG,
)
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_PHYSX_CFG


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
                joint_drive_props=[sim_utils.UsdPhysicsDriveCfg(max_force=80.0), PhysxJointCfg(max_joint_velocity=5.0)],
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
                joint_drive_props=[sim_utils.UsdPhysicsDriveCfg(max_force=80.0), PhysxJointCfg(max_joint_velocity=5.0)],
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
    articulation = Articulation(articulation_cfg.replace(prim_path="/World/Env_[^/]*/Robot"))

    return articulation, translations


# ---------------------------------------------------------------------------
# Franka task-space tracking helpers for the IK test.
# Mirrors the helpers in ``isaaclab_newton/test/assets/test_articulation.py``.
# ---------------------------------------------------------------------------


def _setup_franka_at_home_pose(sim):
    """Build a Franka articulation at its configured home pose.

    See the Newton-side mirror for full docs. Standalone tests skip the
    env reset path that normally pushes ``default_joint_pos`` to sim,
    so we teleport explicitly to avoid the URDF-neutral
    near-singular pose where the Franka wrist axes nearly align.

    Args:
        sim: The simulation context to use.

    Returns:
        Tuple of ``(robot, ee_frame_idx, ee_jacobi_idx, arm_joint_ids)``.
    """
    cfg = FRANKA_PANDA_HIGH_PD_CFG.copy().replace(prim_path="/World/Env_[^/]*/Robot")
    sim_utils.create_prim("/World/Env_0", "Xform", translation=(0.0, 0.0, 0.0))
    robot = Articulation(cfg)
    sim.reset()
    assert robot.is_initialized

    ee_frame_idx = robot.find_bodies("panda_hand")[0][0]
    ee_jacobi_idx = ee_frame_idx - 1
    arm_joint_ids = robot.find_joints(["panda_joint.*"])[0]

    robot.write_joint_state_to_sim(
        position=robot.data.default_joint_pos.torch[:, :].clone(),
        velocity=robot.data.default_joint_vel.torch[:, :].clone(),
    )
    return robot, ee_frame_idx, ee_jacobi_idx, arm_joint_ids


def _compute_ee_pose_root(robot, ee_frame_idx):
    """Return ``(ee_pos_b, ee_quat_b, root_pose_w)`` in the root frame."""
    ee_pose_w = robot.data.body_pose_w.torch[:, ee_frame_idx]
    root_pose_w = robot.data.root_pose_w.torch
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    return ee_pos_b, ee_quat_b, root_pose_w


def _compute_jacobian_root_frame(robot, ee_jacobi_idx, arm_joint_ids):
    """Return the EE Jacobian sliced to ``arm_joint_ids`` and rotated to the root frame."""
    jacobian = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, :][:, :, arm_joint_ids]
    base_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_pose_w.torch[:, 3:7]))
    jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
    return jacobian


def _build_relative_pose_target(robot, ee_frame_idx, delta_xyz, device):
    """Build a target pose = (current EE pose) + ``delta_xyz``, preserving orientation."""
    initial_ee_pos_b, initial_ee_quat_b, _ = _compute_ee_pose_root(robot, ee_frame_idx)
    target_pos_b = initial_ee_pos_b + torch.tensor([list(delta_xyz)], device=device, dtype=initial_ee_pos_b.dtype)
    return torch.cat([target_pos_b, initial_ee_quat_b], dim=-1)


def _summarize_history(history, tail: int = 200):
    """Return ``(min, mean)`` over the last ``tail`` samples."""
    tail_slice = history[-tail:]
    return min(tail_slice), sum(tail_slice) / len(tail_slice)


def _to_device_tensor(array: wp.array, device: str) -> torch.Tensor:
    """Convert a Warp array to a torch tensor on :paramref:`device`."""
    return wp.to_torch(array).to(device=device)


def _assert_backend_to_user(
    public_tensor: torch.Tensor, backend_tensor: torch.Tensor, user_to_backend: list[int]
) -> None:
    """Assert a public tensor equals a backend tensor reordered to user order."""
    torch.testing.assert_close(public_tensor, backend_tensor.to(device=public_tensor.device)[:, user_to_backend])


def _assert_user_write_reaches_backend(
    user_tensor: torch.Tensor, backend_tensor: torch.Tensor, backend_to_user: list[int]
) -> None:
    """Assert a user-order write reached backend storage in backend order."""
    torch.testing.assert_close(backend_tensor.to(device=user_tensor.device), user_tensor[:, backend_to_user])


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
    with build_simulation_context(
        device=device, auto_add_lighting=True, gravity_enabled=gravity_enabled, add_ground_plane=add_ground_plane
    ) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("gravity_enabled", [False])
def test_live_manual_root_preserving_ordering_reorders_backend_reads_and_writes(sim, device, gravity_enabled):
    """Smoke-test non-identity joint/body ordering through a live PhysX articulation."""
    articulation_cfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/Robot",
        joint_ordering=tuple(reversed(PANDA_JOINT_NAMES)),
        body_ordering=PANDA_ROOT_PRESERVING_REVERSED_BODY_NAMES,
    )
    articulation = Articulation(articulation_cfg)

    sim.reset()
    assert articulation.is_initialized
    assert articulation.backend_joint_names == list(PANDA_JOINT_NAMES)
    assert articulation.backend_body_names == list(PANDA_BODY_NAMES)
    assert articulation.joint_ordering is not None
    assert articulation.body_ordering is not None

    joint_user_to_backend = list(articulation.joint_ordering.user_to_backend_indices)
    joint_backend_to_user = list(articulation.joint_ordering.backend_to_user_indices)
    body_user_to_backend = list(articulation.body_ordering.user_to_backend_indices)

    joint_index = torch.arange(articulation.num_joints, device=device, dtype=torch.float32).unsqueeze(0)
    joint_pos = torch.linspace(-0.3, 0.3, articulation.num_joints, device=device).unsqueeze(0)
    joint_vel = torch.linspace(0.05, 0.13, articulation.num_joints, device=device).unsqueeze(0)
    joint_stiffness = 10.0 + joint_index

    articulation.write_joint_stiffness_to_sim_index(stiffness=joint_stiffness, full_data=True)
    articulation.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, full_data=True)
    articulation.write_data_to_sim()

    _assert_user_write_reaches_backend(
        joint_pos, _to_device_tensor(articulation.root_view.get_dof_positions(), device), joint_backend_to_user
    )
    _assert_user_write_reaches_backend(
        joint_vel, _to_device_tensor(articulation.root_view.get_dof_velocities(), device), joint_backend_to_user
    )
    _assert_user_write_reaches_backend(
        joint_stiffness,
        _to_device_tensor(articulation.root_view.get_dof_stiffnesses(), device),
        joint_backend_to_user,
    )

    sim.step()
    articulation.update(sim.cfg.dt)

    _assert_backend_to_user(
        articulation.data.joint_pos.torch,
        _to_device_tensor(articulation.root_view.get_dof_positions(), device),
        joint_user_to_backend,
    )
    _assert_backend_to_user(
        articulation.data.joint_vel.torch,
        _to_device_tensor(articulation.root_view.get_dof_velocities(), device),
        joint_user_to_backend,
    )
    _assert_backend_to_user(
        articulation.data.joint_stiffness.torch,
        _to_device_tensor(articulation.root_view.get_dof_stiffnesses(), device),
        joint_user_to_backend,
    )
    _assert_backend_to_user(
        articulation.data.body_link_pose_w.torch,
        _to_device_tensor(articulation.root_view.get_link_transforms(), device),
        body_user_to_backend,
    )
    _assert_backend_to_user(
        articulation.data.body_com_pose_b.torch,
        _to_device_tensor(articulation.root_view.get_coms(), device),
        body_user_to_backend,
    )

    # The split accessors must be sliced from the reordered public poses, not from a stale backend-order cache.
    torch.testing.assert_close(articulation.data.body_com_pos_b.torch, articulation.data.body_com_pose_b.torch[..., :3])
    torch.testing.assert_close(
        articulation.data.body_com_quat_b.torch, articulation.data.body_com_pose_b.torch[..., 3:]
    )
    torch.testing.assert_close(articulation.data.body_pos_w.torch, articulation.data.body_link_pose_w.torch[..., :3])
    torch.testing.assert_close(articulation.data.body_quat_w.torch, articulation.data.body_link_pose_w.torch[..., 3:])


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


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("gravity_enabled", [False])
def test_live_floating_root_writers_match_identity_after_body_reordering(sim, device, gravity_enabled):
    """Keep floating-base root writes invariant when public body order moves the root."""
    floating_spawn = FRANKA_PANDA_CFG.spawn.replace(fix_root_link=False)
    identity = Articulation(
        FRANKA_PANDA_CFG.replace(
            prim_path="/World/IdentityRobot",
            spawn=floating_spawn,
            body_ordering=None,
        )
    )
    ordered = Articulation(
        FRANKA_PANDA_CFG.replace(
            prim_path="/World/OrderedRobot",
            spawn=floating_spawn,
            body_ordering=tuple(reversed(PANDA_BODY_NAMES)),
        )
    )

    sim.reset()
    assert identity.is_initialized and ordered.is_initialized
    assert not identity.is_fixed_base and not ordered.is_fixed_base
    assert identity.body_ordering is None
    assert ordered.body_ordering is not None
    assert ordered.body_ordering.backend_to_user_indices[0] != 0

    backend_coms = torch.zeros((1, len(PANDA_BODY_NAMES), 7), device=device)
    body_index = torch.arange(len(PANDA_BODY_NAMES), device=device, dtype=torch.float32)
    backend_coms[0, :, 0] = 0.05 + 0.01 * body_index
    backend_coms[0, :, 1] = -0.03 - 0.02 * body_index
    backend_coms[0, :, 2] = 0.02 + 0.03 * body_index
    backend_coms[..., 6] = 1.0
    identity.set_coms_index(
        coms=wp.from_torch(backend_coms.contiguous(), dtype=wp.transformf),
        full_data=True,
    )
    ordered_user_to_backend = list(ordered.body_ordering.user_to_backend_indices)
    ordered.set_coms_index(
        coms=wp.from_torch(backend_coms[:, ordered_user_to_backend].contiguous(), dtype=wp.transformf),
        full_data=True,
    )
    torch.testing.assert_close(_to_device_tensor(identity.root_view.get_coms(), device), backend_coms)
    torch.testing.assert_close(_to_device_tensor(ordered.root_view.get_coms(), device), backend_coms)

    root_com_pose = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]], device=device)
    root_link_velocity = torch.tensor([[0.4, -0.3, 0.2, 1.1, -0.7, 0.9]], device=device)
    for articulation in (identity, ordered):
        articulation.write_root_com_pose_to_sim_index(root_pose=root_com_pose)
        articulation.write_root_link_velocity_to_sim_index(root_velocity=root_link_velocity)

    torch.testing.assert_close(
        _to_device_tensor(ordered.root_view.get_root_transforms(), device),
        _to_device_tensor(identity.root_view.get_root_transforms(), device),
    )
    torch.testing.assert_close(
        _to_device_tensor(ordered.root_view.get_root_velocities(), device),
        _to_device_tensor(identity.root_view.get_root_velocities(), device),
    )
    torch.testing.assert_close(ordered.data.root_com_vel_w.torch, identity.data.root_com_vel_w.torch)
    for articulation in (identity, ordered):
        torch.testing.assert_close(articulation.data.root_com_pose_w.torch, root_com_pose)
        torch.testing.assert_close(articulation.data.root_link_vel_w.torch, root_link_velocity)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.parametrize("body_ordering", ["identity", "reversed"])
def test_live_direct_view_mass_inertia_writes_become_visible(sim, device, gravity_enabled, body_ordering):
    """Direct tensor-view writes to body masses/inertias become visible on the lazy read.

    Develop reads these properties directly from the tensor view on every access. The timestamp-lazy
    implementation could instead hide direct ``root_view.set_masses`` / ``set_inertias`` writes after
    its first read. This regression requires those writes to become visible through ``data.body_mass`` /
    ``data.body_inertia`` when the lazy buffer is next eligible to refresh:

    - Case A: after a primed read and a subsequent simulation update (the lazy gate opens once per
      step).
    - Case B: on the very first read of a cold buffer.

    Case B runs first so that the buffers are still cold. Both identity and non-identity (reversed) body
    ordering are covered; under ordering the public buffers must equal the backend-order view gathered through
    ``user_to_backend``.
    """
    body_ordering_arg = None if body_ordering == "identity" else PANDA_ROOT_PRESERVING_REVERSED_BODY_NAMES
    articulation = Articulation(FRANKA_PANDA_CFG.replace(prim_path="/World/Robot", body_ordering=body_ordering_arg))
    sim.reset()
    assert articulation.is_initialized

    if body_ordering == "identity":
        assert articulation.body_ordering is None
        body_user_to_backend = list(range(articulation.num_bodies))
    else:
        assert articulation.body_ordering is not None
        body_user_to_backend = list(articulation.body_ordering.user_to_backend_indices)

    cpu_env_ids = wp.array(list(range(articulation.num_instances)), dtype=wp.int32, device="cpu")

    def write_backend_mass_inertia(delta_mass: float, delta_inertia: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Write distinct backend-order masses/inertias straight through the tensor view."""
        backend_masses = wp.to_torch(articulation.root_view.get_masses()).clone() + delta_mass
        backend_inertias = wp.to_torch(articulation.root_view.get_inertias()).clone() + delta_inertia
        articulation.root_view.set_masses(
            wp.from_torch(backend_masses.contiguous(), dtype=wp.float32), indices=cpu_env_ids
        )
        articulation.root_view.set_inertias(
            wp.from_torch(backend_inertias.contiguous(), dtype=wp.float32), indices=cpu_env_ids
        )
        return backend_masses, backend_inertias

    # Case B: the buffers have not been read since reset, so the first read must reflect the write.
    backend_masses, backend_inertias = write_backend_mass_inertia(0.293, 0.023)
    _assert_backend_to_user(articulation.data.body_mass.torch, backend_masses, body_user_to_backend)
    _assert_backend_to_user(articulation.data.body_inertia.torch, backend_inertias, body_user_to_backend)

    # Case A: the buffers are now primed; a new write becomes visible after the sim update opens the lazy gate.
    backend_masses, backend_inertias = write_backend_mass_inertia(0.137, 0.011)
    articulation.update(sim.cfg.dt)
    _assert_backend_to_user(articulation.data.body_mass.torch, backend_masses, body_user_to_backend)
    _assert_backend_to_user(articulation.data.body_inertia.torch, backend_inertias, body_user_to_backend)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("gravity_enabled", [False])
def test_branching_fixture_resolves_distinct_conventions(sim, device, gravity_enabled):
    """Resolve concrete breadth-first PhysX and depth-first MJWarp name orders."""
    fixture_path = Path(__file__).parent / "data" / "articulation_ordering_branching.usda"
    articulation = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(fixture_path)),
            actuators={},
            joint_ordering="mjwarp",
            body_ordering="mjwarp",
        )
    )
    sim.reset()
    assert articulation.is_initialized

    assert tuple(articulation.backend_joint_names) == BRANCHING_PHYSX_JOINT_NAMES
    assert tuple(articulation.backend_body_names) == BRANCHING_PHYSX_BODY_NAMES
    assert get_articulation_name_ordering(articulation, "mjwarp", "joint") == BRANCHING_MJWARP_JOINT_NAMES
    assert get_articulation_name_ordering(articulation, "mjwarp", "body") == BRANCHING_MJWARP_BODY_NAMES
    assert tuple(articulation.joint_names) == BRANCHING_MJWARP_JOINT_NAMES
    assert tuple(articulation.body_names) == BRANCHING_MJWARP_BODY_NAMES
    assert articulation.joint_ordering is not None
    assert articulation.body_ordering is not None


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("add_ground_plane", [True])
def test_initialization_floating_base(sim, num_articulations, device, add_ground_plane):
    """Test initialization for a floating-base with articulation root on provided prim path.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is not fixed base
    3. All buffers have correct shapes

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

    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert getattr(actuator, "is_implicit_model", False) == is_implicit_model_cfg


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

    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names
    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert getattr(actuator, "is_implicit_model", False) == is_implicit_model_cfg

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
    """A tendon length target lands in the simulation as ``rest_length - target`` on the selected cells only.

    The index form commands every tendon of environment 0; the mask form commands tendon 0 of
    environment 1. Every other cell must keep its initial offset.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="shadow_hand")
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)

    sim.reset()
    assert articulation.is_initialized
    assert articulation.is_fixed_base
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 24)
    assert articulation.data.body_mass.torch.shape == (num_articulations, articulation.num_bodies)
    assert articulation.data.body_inertia.torch.shape == (num_articulations, articulation.num_bodies, 9)
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert getattr(actuator, "is_implicit_model", False) == is_implicit_model_cfg
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

    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names

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
def test_initialization_fixed_base_made_floating_base(sim, num_articulations, device, add_ground_plane):
    """Test initialization for fixed base made floating-base using schema properties.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is floating base after modification
    3. All buffers have correct shapes

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
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

    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
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


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
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
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
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


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("add_ground_plane", [True])
def test_external_force_on_single_body(sim, num_articulations, device, add_ground_plane):
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
    articulation, translations = generate_articulation(articulation_cfg, num_articulations, device=sim.device)
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
        # the lateral force along the base y-axis must push the standing robot sideways
        lateral_displacement = articulation.data.root_pos_w.torch[:, 1] - translations[:, 1]
        assert torch.all(lateral_displacement > 0.5), f"lateral displacement {lateral_displacement.tolist()}"


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("add_ground_plane", [True])
def test_external_force_on_single_body_at_position(sim, num_articulations, device, add_ground_plane):
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
    articulation, translations = generate_articulation(articulation_cfg, num_articulations, device=sim.device)
    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies("base")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 2] = 500.0
    external_wrench_positions_b = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    external_wrench_positions_b[..., 1] = 1.0

    # Now we are ready!
    for i in range(5):
        # reset root state
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
        # apply force
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
        # the upward force applied 1 m along the base y-axis must roll the robot about its x-axis
        roll_rate = articulation.data.root_ang_vel_b.torch[:, 0]
        assert torch.all(roll_rate > 0.1), f"roll rate {roll_rate.tolist()} (is_global={is_global})"


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
            # the response axis depends on the link frames, so check that the articulation rotates
            assert torch.linalg.vector_norm(articulation.data.root_ang_vel_w.torch[i]).item() > 0.1


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
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


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize(("stiffness", "damping"), [(10.0, 2.0), ({".*": 10.0}, {".*": 2.0})])
def test_setting_gains_from_cfg(sim, num_articulations, device, add_ground_plane, stiffness, damping):
    """Test that gains are loaded from the configuration correctly.

    This test verifies that:
    1. Gains are loaded correctly from a scalar or a per-joint dictionary configuration
    2. The gains match the expected values
    3. The gains are applied correctly to the actuators

    The humanoid has its articulation root on a rigid body below the spawned prim, so this test also checks
    that the floating-base root is discovered there.

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid", stiffness=stiffness, damping=damping)
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=sim.device
    )

    # Play sim
    sim.reset()

    # Floating-base root discovered on a rigid body below the spawned prim
    assert articulation.is_initialized
    assert not articulation.is_fixed_base
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 21)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names

    # Expected gains
    expected_stiffness = torch.full(
        (articulation.num_instances, articulation.num_joints), 10.0, device=articulation.device
    )
    expected_damping = torch.full_like(expected_stiffness, 2.0)

    # Check that gains are loaded from USD file
    torch.testing.assert_close(articulation.actuators["body"].stiffness, expected_stiffness)
    torch.testing.assert_close(articulation.actuators["body"].damping, expected_damping)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("joint_velocity_limit", [1e5, None])
def test_setting_velocity_limit_writes_to_solver(sim, device, joint_velocity_limit):
    """Test that the resolved joint velocity limit reaches the PhysX solver.

    The full limit-resolution matrix (config override vs. USD default, implicit and explicit
    actuators, actuator-limit soft fallback) is covered on the Newton backend and at unit
    level. This smoke test only verifies the PhysX write path: the configured limit (or the
    USD-authored default when unset) lands in the native solver buffers and matches
    ``data.joint_vel_limits``.
    """
    articulation_cfg = generate_articulation_cfg(
        articulation_type="single_joint_implicit",
        joint_velocity_limit=joint_velocity_limit,
    )
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg,
        num_articulations=1,
        device=device,
    )
    # Play sim
    sim.reset()

    # read the values set into the simulation
    physx_vel_limit = wp.to_torch(articulation.root_view.get_dof_max_velocities()).to(device)
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


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("joint_effort_limit", [1e5, None])
@pytest.mark.parametrize("gravity_enabled", [False])
def test_setting_effort_limit_writes_to_solver(sim, device, joint_effort_limit, gravity_enabled):
    """Test that the resolved joint effort limit and a commanded effort reach the PhysX solver.

    The full limit-resolution matrix (config override vs. USD default, implicit and explicit
    actuators, actuator-limit soft fallback) is covered on the Newton backend and at unit
    level. This smoke test verifies the PhysX write path: the configured limit (or the USD-authored
    default when unset) lands in the native solver buffers, and a commanded effort produces motion.
    """
    articulation_cfg = generate_articulation_cfg(
        articulation_type="single_joint_implicit",
        joint_effort_limit=joint_effort_limit,
    )
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg,
        num_articulations=1,
        device=device,
    )
    # Play sim
    sim.reset()

    # obtain the physx effort limits
    physx_effort_limit = wp.to_torch(articulation.root_view.get_dof_max_forces()).to(device=device)
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

    # Exercise the command path as well as the property readback. The Isaac Sim 6.0 tensor backend
    # accepted this write and updated its staging buffer without applying the effort to the joint.
    initial_position = articulation.data.default_joint_pos.torch.clone()
    articulation.write_joint_state_to_sim_index(
        position=initial_position,
        velocity=torch.zeros_like(initial_position),
        full_data=True,
    )
    effort_target = torch.full_like(initial_position, 10.0)
    articulation.actuators.target_command.set_position_index(value=initial_position, full_data=True)
    articulation.actuators.target_command.set_velocity_index(value=torch.zeros_like(initial_position), full_data=True)
    articulation.actuators.target_command.set_effort_index(value=effort_target, full_data=True)
    articulation.write_data_to_sim()
    sim.step()
    articulation.update(sim.cfg.dt)
    assert torch.all(articulation.data.joint_vel.torch > 1e-3), (
        "a positive effort target must accelerate the commanded joint"
    )


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_reset(sim, num_articulations, device, monkeypatch):
    """Test that reset method works properly."""
    articulation_cfg = generate_articulation_cfg(articulation_type="humanoid")
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
    )

    # Play the simulator
    sim.reset()

    # Now we are ready!
    actuator = next(iter(articulation.actuators.values()))
    actuator_reset = actuator.reset
    reset_env_ids = []

    def record_actuator_reset(env_ids=None):
        reset_env_ids.append(env_ids)
        actuator_reset(env_ids)

    monkeypatch.setattr(actuator, "reset", record_actuator_reset)

    num_bodies = articulation.num_bodies
    composers = (articulation.instantaneous_wrench_composer, articulation.permanent_wrench_composer)
    ones = torch.ones((num_articulations, num_bodies, 3), device=device)
    articulation.permanent_wrench_composer.set_forces_and_torques_index(forces=ones, torques=ones)
    articulation.instantaneous_wrench_composer.add_forces_and_torques_index(forces=ones, torques=ones)

    # A partial reset clears only the selected environment
    articulation.reset(env_ids=torch.tensor([0], device=device))
    for composer in composers:
        assert composer.active
        for buffer in (composer.out_force_b.torch, composer.out_torque_b.torch):
            assert torch.count_nonzero(buffer[0]) == 0
            assert torch.count_nonzero(buffer[1:]) == buffer[1:].numel()

    # A full reset resets every actuator environment and clears all external forces and torques
    reset_env_ids.clear()
    articulation.reset()
    assert reset_env_ids == [None]
    for composer in composers:
        assert not composer.active
        assert torch.count_nonzero(composer.out_force_b.torch) == 0
        assert torch.count_nonzero(composer.out_torque_b.torch) == 0


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

    # The commanded joint must move toward its target. We can't check that it reached the target as the gains
    # are not properly tuned
    default_joint_pos = articulation.data.default_joint_pos.torch
    assert torch.all(articulation.data.joint_pos.torch[:, 3].abs() < default_joint_pos[:, 3].abs() - 0.5)


@pytest.mark.parametrize("num_articulations", [2])
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
    articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)
    env_idx = torch.tensor([x for x in range(num_articulations)], device=device, dtype=torch.int32)
    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10, "Possible reference leak for articulation"
    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized, "Articulation is not initialized"
    # Check that fixed base
    assert articulation.is_fixed_base, "Articulation is not a fixed base"
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 1)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names

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
    com = wp.to_torch(articulation.root_view.get_coms())
    link_offset = [1.0, 0.0, 0.0]  # the offset from CenterPivot to Arm frames
    new_com = torch.tensor(offset, device=device).repeat(num_articulations, 1, 1)
    com[:, arm_idx, :3] = new_com.squeeze(-2)
    articulation.set_coms_index(
        coms=wp.from_torch(com.to(device).contiguous(), dtype=wp.transformf),
        env_ids=wp.from_torch(env_idx, dtype=wp.int32),
    )

    # check they are set
    torch.testing.assert_close(wp.to_torch(articulation.root_view.get_coms()), com.cpu())

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

        # the fixed root stays at its default state
        default_root_pose = articulation.data.default_root_pose.torch.clone()
        default_root_pose[:, :3] += env_pos
        torch.testing.assert_close(root_link_pose_w, default_root_pose)
        torch.testing.assert_close(root_com_vel_w, articulation.data.default_root_vel.torch)

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
            pos_gt = torch.zeros(num_articulations, num_bodies, 3, device=device)
            px = (link_offset[0] + offset[0]) * torch.cos(joint_pos)
            py = torch.zeros(num_articulations, 1, 1, device=device)
            pz = (link_offset[0] + offset[0]) * torch.sin(joint_pos)
            pos_gt[:, arm_idx, :] = torch.cat([px, py, pz], dim=-1).squeeze(-2)
            pos_gt += env_pos.unsqueeze(-2).repeat(1, num_bodies, 1)
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


@pytest.mark.parametrize("num_articulations", [2])
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
    com = wp.to_torch(articulation.root_view.get_coms())
    new_com = offset
    com[:, 0, :3] = new_com.squeeze(-2)
    articulation.set_coms_index(
        coms=wp.from_torch(com.to(device).contiguous(), dtype=wp.transformf),
        env_ids=wp.from_torch(env_idx, dtype=wp.int32),
    )

    # check they are set
    torch.testing.assert_close(wp.to_torch(articulation.root_view.get_coms()), com)

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


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
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


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
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

    articulation.write_joint_position_to_sim_index(position=rand_joint_pos)
    articulation.write_joint_velocity_to_sim_index(velocity=rand_joint_vel)
    # make sure valued updated
    body_link_pose_w = articulation.data.body_link_pose_w.torch
    body_com_vel_w = articulation.data.body_com_vel_w.torch
    original_body_states = torch.cat([original_body_link_pose_w, original_body_com_vel_w], dim=-1)
    body_state_w = torch.cat([body_link_pose_w, body_com_vel_w], dim=-1)
    # every non-root body must move with the joint write
    assert torch.all((original_body_states[:, 1:] != body_state_w[:, 1:]).any(dim=-1))
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


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_spatial_tendons(sim, num_articulations, device):
    """Test spatial tendons apis.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation has spatial tendons
    3. All buffers have correct shapes
    4. Spatial tendon properties reach the simulation for the right environments
    5. Joint state writes accept int64 environment and joint selectors

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

    # Distinct per-environment values so that an environment mix-up cannot pass
    env_offset = torch.arange(num_articulations, dtype=torch.float32, device=device).unsqueeze(1)
    stiffness, limit_stiffness, damping, offset = (10.0 + env_offset, 20.0 + env_offset, 3.0 + env_offset, env_offset)
    articulation.set_spatial_tendon_stiffness_index(stiffness=stiffness)
    articulation.set_spatial_tendon_limit_stiffness_index(limit_stiffness=limit_stiffness)
    articulation.set_spatial_tendon_damping_index(damping=damping)
    articulation.set_spatial_tendon_offset_index(offset=offset)
    articulation.write_spatial_tendon_properties_to_sim_index()

    root_view = articulation.root_view
    torch.testing.assert_close(_to_device_tensor(root_view.get_spatial_tendon_stiffnesses(), device), stiffness)
    torch.testing.assert_close(
        _to_device_tensor(root_view.get_spatial_tendon_limit_stiffnesses(), device), limit_stiffness
    )
    torch.testing.assert_close(_to_device_tensor(root_view.get_spatial_tendon_dampings(), device), damping)
    torch.testing.assert_close(_to_device_tensor(root_view.get_spatial_tendon_offsets(), device), offset)

    # Joint state writes accept int64 selectors (regression)
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


@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
def test_write_joint_frictions_to_sim(sim, num_articulations, device, add_ground_plane):
    """Test that joint friction coefficients written through the combined API reach the simulation."""
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

    for _ in range(100):
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

    friction_props_from_sim = wp.to_torch(articulation.root_view.get_dof_friction_properties())
    joint_friction_coeff_sim = friction_props_from_sim[:, :, 0]
    joint_dynamic_friction_coeff_sim = friction_props_from_sim[:, :, 1]
    joint_viscous_friction_coeff_sim = friction_props_from_sim[:, :, 2]
    assert torch.allclose(joint_dynamic_friction_coeff_sim, dynamic_friction.cpu())
    assert torch.allclose(joint_viscous_friction_coeff_sim, viscous_friction.cpu())
    assert torch.allclose(joint_friction_coeff_sim, friction.cpu())


##
# Shape-contract regression tests for the new BaseArticulation accessors.
# Mirror the Newton-side tests so both backends can be diffed against the
# same documented contract. These are PhysX's reference shapes — when the
# Newton-side tests pass with the same expected_shape formulas, the
# cross-backend contract holds.
##


@pytest.mark.parametrize("num_articulations", [4])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["panda", "anymal"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_get_jacobians_link_origin_contract(sim, num_articulations, device, articulation_type, gravity_enabled):
    """PhysX reference: ``J · q_dot`` matches ``[body_link_lin_vel_w; body_link_ang_vel_w]``.

    The cross-backend contract on
    :attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w` says
    the Jacobian's linear rows reference each body's link origin. PhysX's
    raw ``_root_view.get_jacobians()`` returns COM-referenced linear rows;
    the IsaacLab wrapper applies the COM→origin shift kernel so the contract
    holds. This test pins the identity from the PhysX side and parametrizes
    on Anymal so the (non-trivial) shift surfaces if it ever regresses.

    Scene gravity is disabled (``gravity_enabled=False``) so the only source
    of a J · q_dot ↔ body_*_w mismatch is the reference-point contract (or a
    regression). The tolerance ``5e-2`` is loose enough to absorb the small
    PhysX state-propagation lag between the Jacobian and the velocity
    buffers (~2% on max angular speed) but well below the
    COM-vs-link-origin bug magnitude (panda hand COM offset ≈ 3 cm × ω at
    typical motion ≈ several rad/s gives a 0.1+ m/s linear-row residual,
    2× the tolerance).
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    assert articulation.is_initialized

    torch.manual_seed(0)
    qdot = torch.randn(num_articulations, articulation.num_joints, device=device) * 0.5
    articulation.write_joint_velocity_to_sim(velocity=qdot)
    sim.step()
    articulation.update(sim.cfg.dt)

    # body_link_jacobian_w prepends ``num_base_dofs`` floating-base columns; slice past
    # them so the joint axis aligns with joint_vel (actuated-only).
    J = articulation.data.body_link_jacobian_w.torch[..., articulation.num_base_dofs :]
    qdot_view = articulation.data.joint_vel.torch
    v_pred = torch.einsum("nbij,nj->nbi", J, qdot_view)

    body_lin_w = articulation.data.body_link_lin_vel_w.torch
    body_ang_w = articulation.data.body_link_ang_vel_w.torch
    if articulation.is_fixed_base:
        body_lin_w = body_lin_w[:, 1:]
        body_ang_w = body_ang_w[:, 1:]

    torch.testing.assert_close(v_pred[..., 3:6], body_ang_w, atol=1.5e-1, rtol=5e-2)
    torch.testing.assert_close(v_pred[..., 0:3], body_lin_w, atol=1.5e-1, rtol=5e-2)

    # Shape contract: fixed-base Jacobians omit the root body; floating-base ones prepend the base DoFs.
    num_dofs = articulation.num_joints + articulation.num_base_dofs
    num_jacobian_bodies = articulation.num_bodies - 1 if articulation.is_fixed_base else articulation.num_bodies
    J_full = articulation.data.body_link_jacobian_w.torch
    assert J_full.shape == (num_articulations, num_jacobian_bodies, 6, num_dofs)

    # The joint-space mass matrix is square over all DoFs, symmetric, and positive definite.
    M = articulation.data.mass_matrix.torch
    assert M.shape == (num_articulations, num_dofs, num_dofs)
    assert (M.diagonal(dim1=-2, dim2=-1) > 1e-6).all()
    asym = (M - M.transpose(-1, -2)).abs().max().item()
    assert asym < 1e-4, f"|M - M^T|_max = {asym:.3e} — mass matrix is not symmetric"
    eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype).expand_as(M)
    torch.linalg.cholesky(M + 1e-6 * eye)


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["panda", "anymal"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_dynamics_refresh_after_manual_joint_write(sim, num_articulations, device, articulation_type, gravity_enabled):
    """Each Jacobian and mass-matrix read independently reflects a manual joint write without stepping."""
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    sim.step()
    articulation.update(sim.cfg.dt)

    q_initial = articulation.data.joint_pos.torch.clone()
    env_ids = wp.array([0], dtype=wp.int32, device=device)
    for property_name in ("body_link_jacobian_w", "body_com_jacobian_w", "mass_matrix"):
        articulation.write_joint_position_to_sim_index(position=q_initial, env_ids=env_ids)
        before = getattr(articulation.data, property_name).torch.clone()
        # A separate write for each getter prevents another getter from refreshing FK on its behalf.
        articulation.write_joint_position_to_sim_index(position=q_initial + 0.5, env_ids=env_ids)
        after = getattr(articulation.data, property_name).torch.clone()
        assert not torch.allclose(before, after, atol=1e-3), f"{property_name} stayed stale after a joint write"


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.isaacsim_ci
def test_get_gravity_compensation_forces_static_equilibrium(sim, num_articulations, device, articulation_type):
    """PhysX accuracy: ``τ_gc`` must hold the manipulator in static equilibrium.

    The contract is the EOM identity ``M(q) q̈ + C(q,q̇) q̇ + g(q) = τ_input``.
    Setting ``τ_input = g(q)`` at ``q̇ = 0`` gives ``q̈ = 0`` — the arm should
    not move. This pins
    :attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces`
    in isolation: sign errors, frame errors, and DoF-ordering errors all
    surface as joint drift, while a controller-level test would have those
    bugs averaged out by PD damping.

    Newton-side variant of the same name lives in
    ``isaaclab_newton/test/assets/test_articulation.py`` (backend parity).
    """
    base_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    # Replace default Franka actuators with a passthrough implicit actuator
    # (stiffness = 0, damping = 0). With both gains zero the effort target
    # we set IS the joint torque applied — no PD spring-damper masks the
    # gravity-comp signal. Default Franka cfg has stiffness=80 / damping=4
    # which would absorb gravity through PD bias and hide accessor bugs.
    cfg = base_cfg.replace(
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    # FRANKA_PANDA_CFG has rigid_props.disable_gravity=False already, but be
    # defensive — gravity must be ON for τ_gc to have anything to cancel.
    cfg = cfg.replace(
        spawn=cfg.spawn.replace(
            rigid_props=cfg.spawn.rigid_props.replace(disable_gravity=False),
        ),
    )

    articulation, _ = generate_articulation(cfg, num_articulations, device=device)
    sim.reset()
    assert articulation.is_initialized

    # Force a clean static state: default joint positions, zero velocities.
    # ``sim.reset`` may leave residual ``q_dot`` from solver settling under
    # gravity, so we pin it explicitly here.
    default_q = articulation.data.default_joint_pos.torch.clone()
    default_qd = torch.zeros_like(default_q)
    articulation.write_joint_state_to_sim(default_q, default_qd)
    articulation.update(sim.cfg.dt)

    # Default joint pose from FRANKA_PANDA_CFG bends the elbow
    # (joint2=-0.569, joint4=-2.81, joint6=3.04) so several links carry a
    # gravity load — τ_gc is non-trivial in this configuration. A natural-
    # hang pose (all zeros) would produce near-zero τ_gc and make this
    # test uninformative.
    init_q = articulation.data.joint_pos.torch.clone()

    # Step 100 times applying only τ_gc as joint efforts.
    for _ in range(100):
        # ``gravity_compensation_forces`` shape is ``(N, num_joints + num_base_dofs)``
        # — leading ``num_base_dofs`` floating-base entries (0 on fixed-base) followed
        # by the actuated-joint entries. Slice past the floating-base entries so the
        # remaining tensor aligns with ``set_joint_effort_target`` (actuated only).
        tau_gc = articulation.data.gravity_compensation_forces.torch[:, articulation.num_base_dofs :]
        articulation.set_joint_effort_target(tau_gc)
        articulation.write_data_to_sim()
        sim.step()
        articulation.update(sim.cfg.dt)

    final_q = articulation.data.joint_pos.torch
    drift = (final_q - init_q).abs().max()
    # Tight bound: 5e-3 rad ≈ 0.3°. Numerical integration over 100 steps will
    # accumulate some floor (sub-millirad on Franka), but a sign or frame bug
    # in τ_gc produces drift of at least a degree per step on bent-elbow
    # poses. This bound separates "correct" from "broken" cleanly.
    assert drift < 5e-3, (
        f"max joint drift {drift:.5f} rad after 100 gravity-comp-only steps —"
        " τ_gc did not hold static equilibrium. Check sign, DoF ordering, and"
        " whether gravity_compensation_forces returns g(q) (positive) or"
        " its negation."
    )


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_franka_ik_tracking_accuracy(sim, device, articulation_type, gravity_enabled):
    """PhysX-side IK convergence sentinel — backend parity with the Newton test.

    Mirrors :func:`isaaclab_newton.test.assets.test_articulation.test_franka_ik_tracking_accuracy`
    so both backends are pinned by the same IK trajectory. With the
    robot teleported to its configured init_state home pose and scene
    gravity off, PhysX's IK converges to ~mm precision on this 5 cm
    Cartesian step. A bridge regression (wrong J shape, wrong DoF
    ordering) would push the steady-state error well past the
    threshold.
    """
    robot, ee_frame_idx, ee_jacobi_idx, arm_joint_ids = _setup_franka_at_home_pose(sim)

    sim.step()
    robot.update(sim.cfg.dt)
    target_pose_b = _build_relative_pose_target(robot, ee_frame_idx, (0.05, 0.0, 0.0), device)

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1,
        device=device,
    )
    ik.set_command(target_pose_b)

    pos_history: list[float] = []
    rot_history: list[float] = []
    for _ in range(800):
        jacobian = _compute_jacobian_root_frame(robot, ee_jacobi_idx, arm_joint_ids)
        ee_pos_b, ee_quat_b, _ = _compute_ee_pose_root(robot, ee_frame_idx)
        joint_pos = robot.data.joint_pos.torch[:, arm_joint_ids]

        joint_pos_des = ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        robot.set_joint_position_target(joint_pos_des, joint_ids=arm_joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        pos_error, rot_error = compute_pose_error(ee_pos_b, ee_quat_b, target_pose_b[:, 0:3], target_pose_b[:, 3:7])
        pos_history.append(pos_error.norm(dim=-1).max().item())
        rot_history.append(rot_error.norm(dim=-1).max().item())

    pos_min, pos_mean = _summarize_history(pos_history)
    rot_min, rot_mean = _summarize_history(rot_history)

    print(f"IK_METRIC pos_min={pos_min:.5f} pos_mean={pos_mean:.5f} rot_min={rot_min:.5f} rot_mean={rot_mean:.5f}")

    # Assert on tail mean (not min) so an oscillating envelope can't
    # squeeze through. Threshold matched to the Newton-side test
    # (5 mm / 0.05 rad).
    assert pos_mean < 5e-3, f"IK pos_mean {pos_mean:.5f} > 5 mm — bridge regression?"
    assert rot_mean < 5e-2, f"IK rot_mean {rot_mean:.5f} > 0.05 rad — bridge regression?"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
