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
from isaaclab_physx.assets.articulation.articulation_data import ArticulationData

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ActuatorBase, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, get_articulation_name_ordering
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
    OperationalSpaceController,
    OperationalSpaceControllerCfg,
)
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
    SHADOW_HAND_CFG,
)


def test_cached_read_launch_reuses_command_and_resets(monkeypatch):
    """The PhysX read path should record once, replay, and discard the command on reset."""

    class FakeCommand:
        launch_count = 0

        def launch(self):
            self.launch_count += 1

    class FakeDevice:
        is_cuda = True
        is_capturing = False

    device = FakeDevice()
    command = FakeCommand()
    launch_calls = []

    def fake_launch(*args, **kwargs):
        launch_calls.append((args, kwargs))
        return command

    module = sys.modules[ArticulationData.__module__]
    monkeypatch.setattr(module.wp, "get_device", lambda device_name: device)
    monkeypatch.setattr(module.wp, "launch", fake_launch)
    data = ArticulationData.__new__(ArticulationData)
    data.device = "cuda:0"
    data._cached_read_launches = {}

    for _ in range(2):
        data._launch_cached_read("joint_pos", object(), dim=1, inputs=[], outputs=[])

    assert len(launch_calls) == 1
    assert launch_calls[0][1]["record_cmd"] is True
    assert command.launch_count == 2
    data._reset_cached_read_launches()
    device.is_capturing = True
    data._launch_cached_read("joint_pos", object(), dim=1, inputs=[], outputs=[])
    assert "record_cmd" not in launch_calls[-1][1]
    assert data._cached_read_launches == {}


def test_cached_read_launch_ignores_empty_recording(monkeypatch):
    """PhysX should treat an empty recorded launch as a completed no-op."""

    class FakeDevice:
        is_cuda = True
        is_capturing = False

    launch_calls = []

    def fake_launch(*args, **kwargs):
        launch_calls.append((args, kwargs))
        return

    module = sys.modules[ArticulationData.__module__]
    monkeypatch.setattr(module.wp, "get_device", lambda device_name: FakeDevice())
    monkeypatch.setattr(module.wp, "launch", fake_launch)
    data = ArticulationData.__new__(ArticulationData)
    data.device = "cuda:0"
    data._cached_read_launches = {}

    data._launch_cached_read("joint_pos", object(), dim=0, inputs=[], outputs=[])

    assert len(launch_calls) == 1
    assert launch_calls[0][1]["record_cmd"] is True
    assert data._cached_read_launches == {}


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
                joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_force=80.0, max_joint_velocity=5.0),
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
                joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_force=80.0, max_joint_velocity=5.0),
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


# ---------------------------------------------------------------------------
# Franka task-space tracking helpers (shared between IK and OSC tests).
# Mirrors the helpers in ``isaaclab_newton/test/assets/test_articulation.py``.
# ---------------------------------------------------------------------------


def _setup_franka_at_home_pose(sim, *, zero_actuator_pd: bool = False, enable_rigid_body_gravity: bool = False):
    """Build a Franka articulation at its configured home pose.

    See the Newton-side mirror for full docs. Standalone tests skip the
    env reset path that normally pushes ``default_joint_pos`` to sim,
    so we teleport explicitly to avoid the URDF-neutral
    near-singular pose where the Franka wrist axes nearly align.

    Args:
        sim: The simulation context to use.
        zero_actuator_pd: If True, sets the panda_shoulder/panda_forearm
            actuator stiffness and damping to zero.
        enable_rigid_body_gravity: If True, override
            ``FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity``
            (which defaults to True) so gravity actually loads the arm. Required
            for any test that wants to exercise gravity-related dynamics
            (e.g. gravity-compensation accuracy tests).

    Returns:
        Tuple of ``(robot, ee_frame_idx, ee_jacobi_idx, arm_joint_ids)``.
    """
    cfg = FRANKA_PANDA_HIGH_PD_CFG.copy().replace(prim_path="/World/Env_.*/Robot")
    if zero_actuator_pd:
        cfg.actuators["panda_shoulder"].stiffness = 0.0
        cfg.actuators["panda_shoulder"].damping = 0.0
        cfg.actuators["panda_forearm"].stiffness = 0.0
        cfg.actuators["panda_forearm"].damping = 0.0
    if enable_rigid_body_gravity:
        cfg = cfg.replace(
            spawn=cfg.spawn.replace(
                rigid_props=cfg.spawn.rigid_props.replace(disable_gravity=False),
            ),
        )
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


def _compute_ee_vel_root(jacobian_b, joint_vel):
    """Return the EE 6D velocity in the root frame as ``J · q_dot``.

    Required to make OSC's ``kd * ee_vel_b`` damping term meaningful.
    Passing zero EE velocity (the convenient hack) leaves the impedance
    undamped and the EE oscillates around the target. ``J · q_dot``
    avoids relying on ``data.body_vel_w`` (Newton's lazy velocity
    buffers can return stale/zero values until forced materialization),
    keeping the helper backend-symmetric. ``J`` correctness is pinned
    independently by ``test_get_jacobians_link_origin_contract``.
    """
    return torch.bmm(jacobian_b, joint_vel.unsqueeze(-1)).squeeze(-1)


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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
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
    if device == "cuda:0":
        assert articulation.data._cached_read_launches
    _assert_backend_to_user(
        articulation.data.body_com_pose_b.torch,
        _to_device_tensor(articulation.root_view.get_coms(), device),
        body_user_to_backend,
    )

    torch.testing.assert_close(articulation.data.body_com_pos_b.torch, articulation.data.body_com_pose_b.torch[..., :3])
    torch.testing.assert_close(
        articulation.data.body_com_quat_b.torch, articulation.data.body_com_pose_b.torch[..., 3:]
    )
    torch.testing.assert_close(articulation.data.body_pos_w.torch, articulation.data.body_link_pose_w.torch[..., :3])
    torch.testing.assert_close(articulation.data.body_quat_w.torch, articulation.data.body_link_pose_w.torch[..., 3:])


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("gravity_enabled", [False])
def test_live_floating_root_writers_match_identity_after_body_reordering(sim, device, gravity_enabled):
    """Keep floating-base root writes invariant when public body order moves the root."""
    floating_spawn = FRANKA_PANDA_CFG.spawn.replace(
        articulation_props=FRANKA_PANDA_CFG.spawn.articulation_props.replace(fix_root_link=False)
    )
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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
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
    - Case B: on the very first read of a cold buffer (initial timestamp -1.0).

    Both identity and non-identity (reversed) body ordering are covered; under ordering the public
    buffers must equal the backend-order view gathered through ``user_to_backend``.
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

    # Case A: prime the buffers, write through the view, then advance the sim so the lazy gate opens.
    _ = articulation.data.body_mass.torch
    _ = articulation.data.body_inertia.torch
    backend_masses, backend_inertias = write_backend_mass_inertia(0.137, 0.011)
    articulation.update(sim.cfg.dt)
    _assert_backend_to_user(articulation.data.body_mass.torch, backend_masses, body_user_to_backend)
    _assert_backend_to_user(articulation.data.body_inertia.torch, backend_inertias, body_user_to_backend)

    # Case B: a cold buffer (initial timestamp -1.0) must reflect a write on its first read.
    articulation.data._body_mass.timestamp = -1.0
    articulation.data._body_inertia.timestamp = -1.0
    backend_masses, backend_inertias = write_backend_mass_inertia(0.293, 0.023)
    _assert_backend_to_user(articulation.data.body_mass.torch, backend_masses, body_user_to_backend)
    _assert_backend_to_user(articulation.data.body_inertia.torch, backend_inertias, body_user_to_backend)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.parametrize("ordering_axis", ["joint", "body"])
def test_unused_jacobian_ordering_map_is_none(sim, device, gravity_enabled, ordering_axis):
    """Keep the inactive Jacobian-axis map unset under single-axis ordering."""
    fixture_path = Path(__file__).parent / "data" / "articulation_ordering_branching.usda"
    articulation = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(fixture_path)),
            actuators={},
            joint_ordering="mjwarp" if ordering_axis == "joint" else None,
            body_ordering="mjwarp" if ordering_axis == "body" else None,
        )
    )
    sim.reset()
    assert articulation.is_initialized

    if ordering_axis == "joint":
        assert articulation.joint_ordering is not None
        assert articulation.body_ordering is None
        assert articulation.data._jacobian_joint_user_to_backend is not None
        assert articulation.data._jacobian_body_user_to_backend is None
    else:
        assert articulation.joint_ordering is None
        assert articulation.body_ordering is not None
        assert articulation.data._jacobian_joint_user_to_backend is None
        assert articulation.data._jacobian_body_user_to_backend is not None


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

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.root_view.max_dofs == articulation.root_view.shared_metatype.dof_count
    # -- link related
    assert articulation.root_view.max_links == articulation.root_view.shared_metatype.link_count
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names
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

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.root_view.max_dofs == articulation.root_view.shared_metatype.dof_count
    # -- link related
    assert articulation.root_view.max_links == articulation.root_view.shared_metatype.link_count
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names
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

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.root_view.max_dofs == articulation.root_view.shared_metatype.dof_count
    # -- link related
    assert articulation.root_view.max_links == articulation.root_view.shared_metatype.link_count
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names
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

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.root_view.max_dofs == articulation.root_view.shared_metatype.dof_count
    # -- link related
    assert articulation.root_view.max_links == articulation.root_view.shared_metatype.link_count
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names
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

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.root_view.max_dofs == articulation.root_view.shared_metatype.dof_count
    # -- link related
    assert articulation.root_view.max_links == articulation.root_view.shared_metatype.link_count
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

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.root_view.max_dofs == articulation.root_view.shared_metatype.dof_count
    # -- link related
    assert articulation.root_view.max_links == articulation.root_view.shared_metatype.link_count
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
    articulation_cfg = generate_articulation_cfg(articulation_type="panda").copy()
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

    # Check some internal physx data for debugging
    # -- joint related
    assert articulation.root_view.max_dofs == articulation.root_view.shared_metatype.dof_count
    # -- link related
    assert articulation.root_view.max_links == articulation.root_view.shared_metatype.link_count
    # -- link names (check within articulation ordering is correct)
    prim_path_body_names = [path.split("/")[-1] for path in articulation.root_view.link_paths[0]]
    assert prim_path_body_names == articulation.body_names

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
            assert articulation.permanent_wrench_composer.out_force_b.torch[i, 0, 0].item() == force
            assert articulation.permanent_wrench_composer.out_torque_b.torch[i, 0, 0].item() == force

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
    physx_vel_limit = wp.to_torch(articulation.root_view.get_dof_max_velocities()).to(device)
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
    physx_vel_limit = wp.to_torch(articulation.root_view.get_dof_max_velocities()).to(device)
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
    physx_effort_limit = wp.to_torch(articulation.root_view.get_dof_max_forces()).to(device=device)

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
    physx_effort_limit = wp.to_torch(articulation.root_view.get_dof_max_forces()).to(device)
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
    articulation.reset()
    assert reset_env_ids == [None]

    # Reset should zero external forces and torques
    assert not articulation._instantaneous_wrench_composer.active
    assert not articulation._permanent_wrench_composer.active
    assert torch.count_nonzero(articulation._instantaneous_wrench_composer.out_force_b.torch) == 0
    assert torch.count_nonzero(articulation._instantaneous_wrench_composer.out_torque_b.torch) == 0
    assert torch.count_nonzero(articulation._permanent_wrench_composer.out_force_b.torch) == 0
    assert torch.count_nonzero(articulation._permanent_wrench_composer.out_torque_b.torch) == 0

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
        assert torch.count_nonzero(articulation._instantaneous_wrench_composer.out_force_b.torch) == num_bodies * 3
        assert torch.count_nonzero(articulation._instantaneous_wrench_composer.out_torque_b.torch) == num_bodies * 3
        assert torch.count_nonzero(articulation._permanent_wrench_composer.out_force_b.torch) == num_bodies * 3
        assert torch.count_nonzero(articulation._permanent_wrench_composer.out_torque_b.torch) == num_bodies * 3


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

    articulation.write_joint_position_to_sim_index(position=rand_joint_pos)
    articulation.write_joint_velocity_to_sim_index(velocity=rand_joint_vel)
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

    friction_props_from_sim = wp.to_torch(articulation.root_view.get_dof_friction_properties())
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

        friction_props_from_sim_2 = wp.to_torch(articulation.root_view.get_dof_friction_properties())
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
    """Test getting and setting material properties (friction/restitution) of articulation shapes."""
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
    )

    # Play the simulator
    sim.reset()

    # Get number of shapes from the articulation
    max_shapes = articulation.root_view.max_shapes

    # Generate random material properties: (static_friction, dynamic_friction, restitution)
    materials = torch.empty(num_articulations, max_shapes, 3, device="cpu").uniform_(0.0, 1.0)
    # Ensure dynamic friction <= static friction
    materials[..., 1] = torch.min(materials[..., 0], materials[..., 1])

    # Set material properties via the PhysX view-level API
    env_ids = torch.arange(num_articulations, dtype=torch.int32)
    articulation.root_view.set_material_properties(
        wp.from_torch(materials, dtype=wp.float32), wp.from_torch(env_ids, dtype=wp.int32)
    )

    # Simulate physics
    sim.step()
    articulation.update(sim.cfg.dt)

    # Get material properties from simulation
    materials_check = wp.to_torch(articulation.root_view.get_material_properties())

    # Check if material properties are set correctly
    torch.testing.assert_close(materials_check, materials)


##
# Shape-contract regression tests for the new BaseArticulation accessors.
# Mirror the Newton-side tests so both backends can be diffed against the
# same documented contract. These are PhysX's reference shapes — when the
# Newton-side tests pass with the same expected_shape formulas, the
# cross-backend contract holds.
##


@pytest.mark.parametrize("num_articulations", [1, 4])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.isaacsim_ci
def test_get_jacobians_shape_fixed_base(sim, num_articulations, device, articulation_type):
    """PhysX reference: fixed-base ``body_link_jacobian_w`` is ``(N, num_bodies-1, 6, num_joints)``."""
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    assert articulation.is_initialized
    assert articulation.is_fixed_base

    J = articulation.data.body_link_jacobian_w.torch
    expected = (num_articulations, articulation.num_bodies - 1, 6, articulation.num_joints)
    assert J.shape == torch.Size(expected), f"expected {expected}, got {tuple(J.shape)}"


@pytest.mark.parametrize("num_articulations", [1, 4])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.isaacsim_ci
def test_get_mass_matrix_shape_and_nonsingular_fixed_base(sim, num_articulations, device, articulation_type):
    """PhysX reference: fixed-base ``mass_matrix`` shape + non-singular."""
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    assert articulation.is_initialized

    sim.step()
    articulation.update(sim.cfg.dt)

    M = articulation.data.mass_matrix.torch
    expected = (num_articulations, articulation.num_joints, articulation.num_joints)
    assert M.shape == torch.Size(expected), f"expected {expected}, got {tuple(M.shape)}"

    # Each diagonal entry is the joint's effective inertia and must be positive
    # for any physical articulation. Padded zero rows/cols (the bug) would show
    # up here as zero diagonal entries — much more sensitive than checking the
    # determinant, which can be small for a well-conditioned 9x9 just from
    # numerical cancellation.
    diag = M.diagonal(dim1=-2, dim2=-1)
    assert (diag > 1e-6).all(), f"mass matrix has non-positive diagonal entries: min={diag.min()}"


@pytest.mark.parametrize("num_articulations", [1, 4])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["anymal"])
@pytest.mark.isaacsim_ci
def test_get_jacobians_shape_floating_base(sim, num_articulations, device, add_ground_plane, articulation_type):
    """PhysX reference: floating-base ``body_link_jacobian_w``.

    Floating-base articulations include the 6 floating-base spatial-velocity columns
    at the front of the DoF axis, so the shape is
    ``(N, num_bodies, 6, num_joints + num_base_dofs)`` — matching Newton and the
    cross-library industry convention (Pinocchio, Drake, MuJoCo, RBDL, OCS2,
    iDynTree).
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    assert articulation.is_initialized
    assert not articulation.is_fixed_base

    J = articulation.data.body_link_jacobian_w.torch
    expected = (num_articulations, articulation.num_bodies, 6, articulation.num_joints + articulation.num_base_dofs)
    assert J.shape == torch.Size(expected), f"expected {expected}, got {tuple(J.shape)}"


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


@pytest.mark.parametrize("num_articulations", [4])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["panda", "anymal"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_get_mass_matrix_symmetry_pd(sim, num_articulations, device, articulation_type, gravity_enabled):
    """The joint-space mass matrix ``M(q)`` must be square, symmetric, and positive-definite.

    Mirrors the Newton-side test in
    ``source/isaaclab_newton/test/assets/test_articulation.py``. Pins
    three structural properties of :attr:`~isaaclab.assets.BaseArticulationData.mass_matrix`
    that every backend must satisfy. Both backends include the 6 floating-base
    rows/cols on floating-base assets (matching the cross-library industry
    convention); this test cares about square + symmetric + PD across both
    fixed- and floating-base, not the absolute column count.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    assert articulation.is_initialized

    sim.step()
    articulation.update(sim.cfg.dt)

    M = articulation.data.mass_matrix.torch  # (N, J, J)
    assert M.dim() == 3, f"expected 3-D mass matrix, got shape {tuple(M.shape)}"
    assert M.shape[0] == num_articulations
    assert M.shape[1] == M.shape[2], f"mass matrix is not square: {tuple(M.shape)}"

    asym = (M - M.transpose(-1, -2)).abs().max().item()
    assert asym < 1e-4, f"|M - M^T|_max = {asym:.3e} — mass matrix is not symmetric"

    eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype).expand_as(M)
    torch.linalg.cholesky(M + 1e-6 * eye)


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["panda", "anymal"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_jacobian_refreshes_after_manual_joint_write(
    sim, num_articulations, device, articulation_type, gravity_enabled
):
    """After ``write_joint_position_to_sim_index`` (no sim step), the Jacobian read
    must reflect the new joint state — not the previous one.

    PhysX-side counterpart to the Newton test of the same name. PhysX's
    :attr:`body_link_jacobian_w` triggers FK indirectly through
    :attr:`body_link_pose_w` (used by the shift kernel); :attr:`body_com_jacobian_w` is
    a passthrough to ``_root_view.get_jacobians()``. This test confirms that PhysX's
    tensor view returns up-to-date Jacobians after a manual joint write — i.e., that
    PhysX internally refreshes FK on ``get_jacobians`` (or that our property does).
    Failure means we need to add ``update_articulations_kinematic()`` before the
    passthrough.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    sim.step()
    articulation.update(sim.cfg.dt)

    # Read J at the baseline joint state.
    J_link_0 = articulation.data.body_link_jacobian_w.torch.clone()
    J_com_0 = articulation.data.body_com_jacobian_w.torch.clone()

    # Manually write a different joint state — large delta to make the change visible.
    # No sim.step / update — FK becomes stale.
    q_target = articulation.data.joint_pos.torch.clone() + 0.5
    env_ids = wp.array([0], dtype=wp.int32, device=device)
    articulation.write_joint_position_to_sim_index(position=q_target, env_ids=env_ids)

    # Read J again. With the FK trigger, J reflects q_target and differs from J at baseline.
    # Without the trigger, body_q stays at baseline, J unchanged.
    J_link_1 = articulation.data.body_link_jacobian_w.torch.clone()
    J_com_1 = articulation.data.body_com_jacobian_w.torch.clone()

    assert not torch.allclose(J_link_0, J_link_1, atol=1e-3), (
        "body_link_jacobian_w did not change after manual joint write — "
        "FK trigger likely missing (eval_jacobian / shift kernel reading stale state.body_q)."
    )
    assert not torch.allclose(J_com_0, J_com_1, atol=1e-3), (
        "body_com_jacobian_w did not change after manual joint write — "
        "PhysX get_jacobians may not auto-refresh FK; consider adding update_articulations_kinematic()."
    )


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["panda", "anymal"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_mass_matrix_refreshes_after_manual_joint_write(
    sim, num_articulations, device, articulation_type, gravity_enabled
):
    """After ``write_joint_position_to_sim_index`` (no sim step), the mass matrix read
    must reflect the new joint state.

    PhysX-side counterpart. :attr:`mass_matrix` is a passthrough to
    ``_root_view.get_generalized_mass_matrices()``. Failure means PhysX's tensor view
    does not auto-refresh FK on this getter, and we need to add
    ``update_articulations_kinematic()`` before the passthrough.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    sim.step()
    articulation.update(sim.cfg.dt)

    M_0 = articulation.data.mass_matrix.torch.clone()
    q_target = articulation.data.joint_pos.torch.clone() + 0.5
    env_ids = wp.array([0], dtype=wp.int32, device=device)
    articulation.write_joint_position_to_sim_index(position=q_target, env_ids=env_ids)
    M_1 = articulation.data.mass_matrix.torch.clone()

    assert not torch.allclose(M_0, M_1, atol=1e-3), (
        "mass_matrix did not change after manual joint write — "
        "PhysX get_generalized_mass_matrices may not auto-refresh FK."
    )


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


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_franka_osc_tracking_accuracy(sim, device, articulation_type, gravity_enabled):
    """PhysX-side OSC pose tracking sentinel — backend parity with Newton.

    Mirrors :func:`isaaclab_newton.test.assets.test_articulation.test_franka_osc_tracking_accuracy`.
    Zero out the actuator's PD gains so OSC's joint-effort output is
    not opposed by the implicit-PD term, matching the Newton test setup.
    """
    robot, ee_frame_idx, ee_jacobi_idx, arm_joint_ids = _setup_franka_at_home_pose(sim, zero_actuator_pd=True)

    osc = OperationalSpaceController(
        OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_stiffness_task=500.0,
            motion_damping_ratio_task=1.0,
        ),
        num_envs=1,
        device=device,
    )

    sim.step()
    robot.update(sim.cfg.dt)
    target_pose_b = _build_relative_pose_target(robot, ee_frame_idx, (0.05, 0.0, 0.0), device)

    pos_history: list[float] = []
    rot_history: list[float] = []
    for _ in range(800):
        jacobian_b = _compute_jacobian_root_frame(robot, ee_jacobi_idx, arm_joint_ids)
        mass_matrix = robot.data.mass_matrix.torch[:, arm_joint_ids, :][:, :, arm_joint_ids]
        ee_pos_b, ee_quat_b, _ = _compute_ee_pose_root(robot, ee_frame_idx)
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
        joint_vel = robot.data.joint_vel.torch[:, arm_joint_ids]
        ee_vel_b = _compute_ee_vel_root(jacobian_b, joint_vel)

        osc.set_command(target_pose_b, current_ee_pose_b=ee_pose_b)
        joint_efforts = osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            mass_matrix=mass_matrix,
            gravity=None,
        )

        robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        pos_error, rot_error = compute_pose_error(ee_pos_b, ee_quat_b, target_pose_b[:, 0:3], target_pose_b[:, 3:7])
        pos_history.append(pos_error.norm(dim=-1).max().item())
        rot_history.append(rot_error.norm(dim=-1).max().item())

    pos_min, pos_mean = _summarize_history(pos_history)
    rot_min, rot_mean = _summarize_history(rot_history)

    print(f"OSC_METRIC pos_min={pos_min:.5f} pos_mean={pos_mean:.5f} rot_min={rot_min:.5f} rot_mean={rot_mean:.5f}")

    # Assert on tail mean. Threshold matched to the Newton-side test
    # (5 mm / 0.05 rad). Both backends converge to machine precision
    # with proper ee-velocity feedback (``J · q_dot``).
    assert pos_mean < 5e-3, f"OSC pos_mean {pos_mean:.5f} > 5 mm — bridge regression?"
    assert rot_mean < 5e-2, f"OSC rot_mean {rot_mean:.5f} > 0.05 rad — bridge regression?"


def _run_osc_stay_still_under_gravity(
    sim,
    device: str,
    *,
    gravity_compensation_enabled: bool,
    num_steps: int = 100,
):
    """Run OSC with a stay-still target on Franka under gravity, return EE drift summary.

    Shared helper for the gravity-comp tests. Setup mirrors
    :func:`test_franka_osc_tracking_accuracy` (zero actuator PD so OSC's joint-effort
    output is not opposed by an implicit-PD spring), but with scene gravity ON and the
    target = the EE pose captured after the first sim step (which already includes a
    fraction-of-a-mm of gravity-induced motion; that's the baseline drift starts from).

    Args:
        gravity_compensation_enabled: If ``True``, the OSC controller cfg has
            ``gravity_compensation=True`` and ``osc.compute(gravity=g(q))`` receives
            the data-layer ``gravity_compensation_forces`` slice. If ``False``,
            ``gravity_compensation=False`` and ``gravity=None``.

    Returns:
        Tuple ``((pos_min, pos_mean), (rot_min, rot_mean))`` over the last 20% of
        steps (per :func:`_summarize_history`), where ``pos`` is in meters and
        ``rot`` in radians.
    """
    # Enable rigid-body gravity so the arm actually feels weight.
    # ``FRANKA_PANDA_HIGH_PD_CFG`` defaults ``disable_gravity=True`` for IK/OSC tests.
    robot, ee_frame_idx, ee_jacobi_idx, arm_joint_ids = _setup_franka_at_home_pose(
        sim, zero_actuator_pd=True, enable_rigid_body_gravity=True
    )

    osc = OperationalSpaceController(
        OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=gravity_compensation_enabled,
            motion_stiffness_task=500.0,
            motion_damping_ratio_task=1.0,
        ),
        num_envs=1,
        device=device,
    )

    sim.step()
    robot.update(sim.cfg.dt)

    # Stay-still target = current EE pose in root frame, captured right after the
    # first step. The OSC loop must hold this pose under gravity.
    initial_ee_pos_b, initial_ee_quat_b, _ = _compute_ee_pose_root(robot, ee_frame_idx)
    target_pose_b = torch.cat([initial_ee_pos_b, initial_ee_quat_b], dim=-1)

    pos_history: list[float] = []
    rot_history: list[float] = []
    for _ in range(num_steps):
        jacobian_b = _compute_jacobian_root_frame(robot, ee_jacobi_idx, arm_joint_ids)
        mass_matrix = robot.data.mass_matrix.torch[:, arm_joint_ids, :][:, :, arm_joint_ids]
        ee_pos_b, ee_quat_b, _ = _compute_ee_pose_root(robot, ee_frame_idx)
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
        joint_vel = robot.data.joint_vel.torch[:, arm_joint_ids]
        ee_vel_b = _compute_ee_vel_root(jacobian_b, joint_vel)

        # ``gravity_compensation_forces`` shape is ``(N, num_joints + num_base_dofs)``;
        # slice past the leading floating-base columns (0 for fixed-base Franka, so a
        # no-op here, but the pattern matches the action-term convention).
        gravity = (
            robot.data.gravity_compensation_forces.torch[:, [j + robot.num_base_dofs for j in arm_joint_ids]]
            if gravity_compensation_enabled
            else None
        )

        osc.set_command(target_pose_b, current_ee_pose_b=ee_pose_b)
        joint_efforts = osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            mass_matrix=mass_matrix,
            gravity=gravity,
        )
        robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

        pos_error, rot_error = compute_pose_error(ee_pos_b, ee_quat_b, target_pose_b[:, 0:3], target_pose_b[:, 3:7])
        pos_history.append(pos_error.norm(dim=-1).max().item())
        rot_history.append(rot_error.norm(dim=-1).max().item())

    return _summarize_history(pos_history), _summarize_history(rot_history)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.parametrize("gravity_enabled", [True])
@pytest.mark.isaacsim_ci
def test_franka_osc_gravity_compensation_holds_under_gravity(sim, device, articulation_type, gravity_enabled):
    """OSC with ``gravity_compensation=True`` must hold the EE pose under gravity.

    With scene gravity ON and zero actuator PD (so OSC torques are not opposed by an
    implicit-PD spring), passing
    :attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces` through
    ``osc.compute(gravity=...)`` should keep the arm at the initial pose.

    Pins three things that the existing direct-primitive
    :func:`test_get_gravity_compensation_forces_static_equilibrium` does not:
      1. OSC's ``_jacobi_joint_idx`` indexing — the ``+ num_base_dofs`` shift.
      2. OSC's :meth:`OperationalSpaceController.compute` correctly adds ``g(q)`` to
         its torque output.
      3. The data-property ``gravity_compensation_forces`` is reachable from the OSC
         pipeline (catches gating regressions in
         :meth:`OperationalSpaceControllerAction._compute_dynamic_quantities`).

    Companion test :func:`test_franka_osc_no_gravity_compensation_sags_under_gravity`
    runs the same setup with ``gravity_compensation=False`` and reports the
    uncompensated drift magnitude — a sanity check that gravity is loading the arm.
    """
    (pos_min, pos_mean), (rot_min, rot_mean) = _run_osc_stay_still_under_gravity(
        sim, device, gravity_compensation_enabled=True
    )
    print(f"OSC_GC_ON pos_min={pos_min:.5f} pos_mean={pos_mean:.5f} rot_min={rot_min:.5f} rot_mean={rot_mean:.5f}")

    assert pos_mean < 5e-3, f"OSC + gravity_compensation pos_mean {pos_mean:.5f} > 5 mm — regression?"
    assert rot_mean < 5e-2, f"OSC + gravity_compensation rot_mean {rot_mean:.5f} > 0.05 rad — regression?"


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.parametrize("gravity_enabled", [True])
@pytest.mark.isaacsim_ci
def test_franka_osc_no_gravity_compensation_sags_under_gravity(sim, device, articulation_type, gravity_enabled):
    """OSC without ``gravity_compensation`` under gravity: sanity check that the arm sags.

    Companion to :func:`test_franka_osc_gravity_compensation_holds_under_gravity`.
    Same setup, but ``gravity_compensation=False`` and ``osc.compute(gravity=None)``.
    With zero actuator PD, OSC's task-space impedance is the only restoring force —
    the steady-state solution is whatever pose error the impedance produces enough
    joint torque to balance ``g(q)``.

    Asserts the drift is **non-trivially larger** than the with-comp threshold (5 mm).
    Without this check, a regression that broke ``gravity_compensation_forces`` by
    returning zeros (or a no-op `g(q)`) would pass the with-comp test silently. The
    bound here proves gravity is actually loading the arm and the with-comp pass is
    meaningful.
    """
    (pos_min, pos_mean), (rot_min, rot_mean) = _run_osc_stay_still_under_gravity(
        sim, device, gravity_compensation_enabled=False
    )
    print(f"OSC_GC_OFF pos_min={pos_min:.5f} pos_mean={pos_mean:.5f} rot_min={rot_min:.5f} rot_mean={rot_mean:.5f}")

    # Sanity: with gravity on and no comp, OSC's task-space spring vs gravity-load
    # equilibrium produces a non-zero pose error. If this asserts fails, the test
    # setup itself is broken (e.g., gravity is not on, or the home pose has no
    # gravity load), which would invalidate the with-comp test as well.
    assert pos_mean > 5e-3, (
        f"OSC + no gravity_compensation pos_mean {pos_mean:.5f} ≤ 5 mm — gravity not loading the arm?"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
