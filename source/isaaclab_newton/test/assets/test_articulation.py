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
    ANYMAL_C_PHYSX_JOINT_NAMES,
    BRANCHING_MJWARP_BODY_NAMES,
    BRANCHING_MJWARP_JOINT_NAMES,
    BRANCHING_PHYSX_BODY_NAMES,
    BRANCHING_PHYSX_JOINT_NAMES,
    PANDA_JOINT_NAMES,
    PANDA_ROOT_PRESERVING_REVERSED_BODY_NAMES,
)

HEADLESS = True

# launch omniverse app
simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

"""Rest everything follows."""

import sys
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace

import newton
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.assets.articulation.actuator_control import NewtonActuatorControl
from isaaclab_newton.assets.articulation.articulation import _configure_builder_joint_target_modes
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager
from isaaclab_physx.sim.schemas import PhysxJointCfg
from newton import JointTargetMode, JointType, ModelBuilder, ModelFlags, ShapeFlags
from newton.selection import ArticulationView
from newton.solvers import SolverMuJoCo

from pxr import UsdPhysics

import isaaclab.assets.articulation.ordering_resolvers as ordering_resolvers
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import (
    IdealPDActuatorCfg,
    ImplicitActuator,
    ImplicitActuatorCfg,
)
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets.articulation.ordering_resolvers import get_articulation_name_ordering
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.envs.mdp.events import randomize_rigid_body_collider_offsets, randomize_rigid_body_material
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import compute_pose_error, matrix_from_quat, quat_inv, subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import ANYMAL_C_CFG, FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort:skip
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_NEWTON_CFG

SIM_CFGs = {
    "humanoid": SimulationCfg(
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=80,
                nconmax=25,
                ls_iterations=20,
                cone="pyramidal",
                update_data_interval=2,
                integrator="implicitfast",
                impratio=1,
            ),
            num_substeps=2,
            debug_mode=False,
        ),
    ),
    "anymal": SimulationCfg(
        dt=1 / 200,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=70,
                nconmax=70,
                ls_iterations=40,
                cone="elliptic",
                impratio=100,
                integrator="implicitfast",
            ),
            num_substeps=2,
            debug_mode=True,
        ),
    ),
    "panda": SimulationCfg(
        dt=1 / 120,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=20,
                nconmax=20,
                ls_iterations=20,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
            ),
            num_substeps=1,
            debug_mode=False,
        ),
    ),
    # "panda" with 4 solver substeps: at a single 1/120 substep, MJWarp implicitfast
    # carries a ~0.4-1.0 mm integration limit cycle under a gravity load that never
    # settles; 4 substeps integrate the same frozen per-control-step torques to a
    # dead-still ~1 um hold. Used by the gravity-compensation precision test.
    "panda_fine": SimulationCfg(
        dt=1 / 120,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=20,
                nconmax=20,
                ls_iterations=20,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
            ),
            num_substeps=4,
            debug_mode=False,
        ),
    ),
    "single_joint_implicit": SimulationCfg(
        dt=1 / 120,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=20,
                nconmax=20,
                ls_iterations=20,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
            ),
            num_substeps=1,
            debug_mode=False,
        ),
    ),
    "single_joint_explicit": SimulationCfg(
        dt=1 / 120,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=20,
                nconmax=20,
                ls_iterations=20,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
            ),
            num_substeps=1,
            debug_mode=False,
        ),
    ),
    "shadow_hand": SimulationCfg(
        dt=1 / 120,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=70,
                nconmax=70,
                ls_iterations=40,
                cone="elliptic",
                impratio=100,
                integrator="implicitfast",
            ),
            num_substeps=2,
            debug_mode=True,
        ),
    ),
}


class CustomDrive(ImplicitActuator):
    """Implicit actuator with a class name that does not encode its execution type."""


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
            "single_joint_explicit" or "spatial_tendon_test_asset".
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
        # The MuJoCo variant, not the PhysX one: only that variant authors the hand's tendons.
        articulation_cfg = SHADOW_HAND_NEWTON_CFG
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


def fix_reversed_joints(stage):
    """Fix reversed joints on the USD stage.

    Some USD assets have joints where physics:body0 is the child and physics:body1 is the parent,
    which is the opposite of what Newton expects. This function detects reversed joints by building
    a graph of body connections and identifying the root body (the one attached to world via a joint
    with a missing body target). Any joint where body0 is closer to the root than body1 is swapped.
    """
    from pxr import UsdPhysics

    # First pass: find root bodies (bodies with a joint that has only one target, i.e. attached to world)
    root_bodies: set[str] = set()
    joints_to_check = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        body0_targets = prim.GetRelationship("physics:body0").GetTargets()
        body1_targets = prim.GetRelationship("physics:body1").GetTargets()
        if body0_targets and not body1_targets:
            root_bodies.add(str(body0_targets[0]))
        elif body1_targets and not body0_targets:
            root_bodies.add(str(body1_targets[0]))
        elif body0_targets and body1_targets:
            joints_to_check.append(prim)

    if not root_bodies:
        return

    # Second pass: for each joint with two bodies, ensure body0 is the parent (closer to root)
    for prim in joints_to_check:
        body0_rel = prim.GetRelationship("physics:body0")
        body1_rel = prim.GetRelationship("physics:body1")
        body0_path = str(body0_rel.GetTargets()[0])
        body1_path = str(body1_rel.GetTargets()[0])

        # Determine if we need to swap: body1 is root or ancestor → need to swap
        body1_is_parent = body1_path in root_bodies or body0_path.startswith(body1_path + "/")
        body0_is_parent = body0_path in root_bodies or body1_path.startswith(body0_path + "/")

        if body0_is_parent or not body1_is_parent:
            continue  # already correct or ambiguous

        # Swap body0 and body1
        body0_rel.SetTargets(body1_rel.GetTargets())
        body1_rel.SetTargets([body0_path])

        # Swap local transforms
        for attr_suffix in ("localPos", "localRot"):
            attr0 = prim.GetAttribute(f"physics:{attr_suffix}0")
            attr1 = prim.GetAttribute(f"physics:{attr_suffix}1")
            val0, val1 = attr0.Get(), attr1.Get()
            if val0 is not None and val1 is not None:
                attr0.Set(val1)
                attr1.Set(val0)


_REVERSED_JOINT_USD_FILES = {"revolute_articulation.usd"}
"""USD filenames with known reversed joint body0/body1 ordering."""


_ANYMAL_C_BODY_NAMES = (
    "base",
    "LF_HIP",
    "LF_THIGH",
    "LF_SHANK",
    "LF_FOOT",
    "LH_HIP",
    "LH_THIGH",
    "LH_SHANK",
    "LH_FOOT",
    "RF_HIP",
    "RF_THIGH",
    "RF_SHANK",
    "RF_FOOT",
    "RH_HIP",
    "RH_THIGH",
    "RH_SHANK",
    "RH_FOOT",
)
_ANYMAL_C_ROOT_PRESERVING_REVERSED_BODY_NAMES = (_ANYMAL_C_BODY_NAMES[0], *reversed(_ANYMAL_C_BODY_NAMES[1:]))


_NEWTON_USER_ORDER_STATE_CACHES = (
    "_joint_pos_user",
    "_joint_vel_user",
    "_body_link_pose_w_user",
    "_body_com_vel_w_user",
)


def generate_articulation(
    articulation_cfg: ArticulationCfg, num_articulations: int, device: str, add_ground_plane: bool = False
) -> tuple[Articulation, torch.tensor]:
    """Generate an articulation from a configuration.

    Handles the creation of the articulation, the environment prims and the articulation's environment
    translations

    Args:
        articulation_cfg: Articulation configuration.
        num_articulations: Number of articulations to generate.
        device: Device to use for the tensors.
        add_ground_plane: Whether the simulation context authored a shared ground plane.

    Returns:
        The articulation and environment translations.

    """
    # Generate translations of 2.5 m in x for each articulation
    translations = np.zeros((num_articulations, 3), dtype=np.float32)
    translations[:, 0] = np.arange(num_articulations) * 2.5

    sim_utils.create_prim("/World/Env_0", "Xform", translation=translations[0])
    articulation_cfg = articulation_cfg.replace(prim_path="/World/Env_[^/]*/Robot")
    cfgs = [articulation_cfg]
    if add_ground_plane:
        cfgs.append(AssetBaseCfg(prim_path="/World/defaultGroundPlane"))
    clone_plan_from_env_0(
        CloneCfg(clone_template="/World/Env_{}"), cfgs, num_articulations, 2.5, positions=translations
    )
    articulation = Articulation(articulation_cfg)

    # Fix reversed joints for known-broken USD assets (body0/body1 swapped)
    usd_path = getattr(articulation_cfg.spawn, "usd_path", "")
    if any(name in usd_path for name in _REVERSED_JOINT_USD_FILES):
        import omni.usd

        fix_reversed_joints(omni.usd.get_context().get_stage())

    return articulation, torch.as_tensor(translations, device=device)


# ---------------------------------------------------------------------------
# Franka task-space tracking helpers (shared between the OSC tests).
# ---------------------------------------------------------------------------


def _setup_franka_at_home_pose(sim, *, zero_actuator_pd: bool = False, disable_gravity: bool = True):
    """Build a Franka articulation at its configured home pose.

    Constructs :data:`FRANKA_PANDA_HIGH_PD_CFG`, optionally zeroes the
    arm-actuator PD gains, resets the simulator, and teleports the
    arm joints to :attr:`default_joint_pos` (the env reset path that
    normally does this is not invoked for standalone tests, so the
    robot would otherwise sit at the URDF-neutral pose where the
    Franka wrist is near-singular).

    Args:
        sim: The simulation context to use.
        zero_actuator_pd: If True, sets the panda_shoulder/panda_forearm
            actuator stiffness and damping to zero. Used by the OSC test
            so OSC's joint-effort output is not opposed by the
            implicit-PD's residual ``kp·(target − q)``.
        disable_gravity: Per-body gravity flag written to the spawn config.
            :data:`FRANKA_PANDA_HIGH_PD_CFG` ships with gravity disabled;
            pass False for tests where the arm must feel scene gravity.

    Returns:
        Tuple of ``(robot, ee_frame_idx, ee_jacobi_idx, arm_joint_ids)``.
    """
    cfg = FRANKA_PANDA_HIGH_PD_CFG.copy().replace(prim_path="/World/Env_[^/]*/Robot")
    if zero_actuator_pd:
        cfg.actuators["panda_shoulder"].stiffness = 0.0
        cfg.actuators["panda_shoulder"].damping = 0.0
        cfg.actuators["panda_forearm"].stiffness = 0.0
        cfg.actuators["panda_forearm"].damping = 0.0
    cfg.spawn.rigid_props.disable_gravity = disable_gravity
    sim_utils.create_prim("/World/Env_0", "Xform", translation=(0.0, 0.0, 0.0))
    clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), (cfg,), 1, 0.0)
    robot = Articulation(cfg)
    replicate(sim.get_clone_plan())
    sim.reset()
    assert robot.is_initialized

    ee_frame_idx = robot.find_bodies("panda_hand")[0][0]
    ee_jacobi_idx = ee_frame_idx - 1
    arm_joint_ids = robot.find_joints(["panda_joint.*"])[0]

    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch[:, :].clone())
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch[:, :].clone())
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
    jacobian = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, arm_joint_ids]
    base_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_pose_w.torch[:, 3:7]))
    jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
    return jacobian


def _compute_ee_vel_root(jacobian_b, joint_vel):
    """Return the EE 6D velocity in the root frame as ``J · q_dot``.

    Required to make OSC's ``kd * ee_vel_b`` damping term meaningful.
    Passing zero EE velocity (the convenient hack) leaves the impedance
    undamped and the EE oscillates around the target. We use ``J · q_dot``
    rather than reading ``data.body_vel_w`` because Newton's lazy
    velocity buffers can return stale/zero values until forced
    materialization, while ``joint_vel`` and ``J`` are already pulled
    by the loop. ``J`` correctness is pinned independently by
    ``test_get_gravity_compensation_forces_matches_jacobian_gravity``.
    """
    return torch.bmm(jacobian_b, joint_vel.unsqueeze(-1)).squeeze(-1)


def _build_relative_pose_target(robot, ee_frame_idx, delta_xyz, device):
    """Build a target pose = (current EE pose) + ``delta_xyz``, preserving orientation."""
    initial_ee_pos_b, initial_ee_quat_b, _ = _compute_ee_pose_root(robot, ee_frame_idx)
    target_pos_b = initial_ee_pos_b + torch.tensor([list(delta_xyz)], device=device, dtype=initial_ee_pos_b.dtype)
    return torch.cat([target_pos_b, initial_ee_quat_b], dim=-1)


def _tail_mean(history, tail: int = 200):
    """Return the mean over the last ``tail`` samples."""
    tail_slice = history[-tail:]
    return sum(tail_slice) / len(tail_slice)


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
    articulation_type = request.getfixturevalue("articulation_type")
    sim_cfg = deepcopy(SIM_CFGs[articulation_type])
    sim_cfg.device = device
    if "use_newton_actuators" in request.fixturenames:
        sim_cfg.use_newton_actuators = request.getfixturevalue("use_newton_actuators")
    # ``gravity_enabled`` is silently ignored by ``build_simulation_context``
    # when an explicit ``sim_cfg`` is also passed; apply it here so the
    # fixture honors what its parameter advertises.
    if not gravity_enabled:
        sim_cfg.gravity = (0.0, 0.0, 0.0)
    with build_simulation_context(
        device=device,
        auto_add_lighting=True,
        gravity_enabled=gravity_enabled,
        add_ground_plane=add_ground_plane,
        sim_cfg=sim_cfg,
    ) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


def _make_target_mode_builder(
    joint_names: list[str], target_modes: list[JointTargetMode], stiffness: list[float], damping: list[float]
) -> ModelBuilder:
    """Build a zero-gain articulated model builder for target-mode tests."""
    builder = ModelBuilder()
    inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    parent = -1
    joint_ids = []
    for joint_name in joint_names:
        link = builder.add_link(mass=1.0, inertia=inertia, label=f"/World/Env_0/Robot/{joint_name}_link")
        joint_ids.append(
            builder.add_joint_revolute(
                parent,
                link,
                target_ke=0.0,
                target_kd=0.0,
                label=f"/World/Env_0/Robot/{joint_name}",
            )
        )
        parent = link
    builder.add_articulation(joint_ids, label="/World/Env_0/Robot")
    builder.articulation_label = ["/World/Env_0/Robot"]
    builder.joint_target_mode = [int(mode) for mode in target_modes]
    builder.joint_target_ke = stiffness
    builder.joint_target_kd = damping
    return builder


@pytest.mark.parametrize(
    ("actuator_cfg", "expected_native_groups"),
    [
        (ImplicitActuatorCfg(joint_names_expr=["joint"], stiffness=10.0, damping=1.0), set()),
        (IdealPDActuatorCfg(joint_names_expr=["joint"], stiffness=None, damping=None), {"explicit"}),
    ],
    ids=["implicit", "explicit"],
)
def test_prepare_native_actuators_activates_only_explicit_groups(monkeypatch, actuator_cfg, expected_native_groups):
    """Keep implicit-only articulations on the solver-drive path and leave solver gains untouched.

    Explicit groups activate the Newton-actuator path without writing gains; collection construction resolves
    the actuator defaults later.
    """
    activation_calls = []
    gain_writes = []
    articulation = SimpleNamespace(
        _sim_cfg=SimpleNamespace(use_newton_actuators=True),
        device="cpu",
        find_joints=lambda _: ([0], ["joint"]),
        write_joint_stiffness_to_sim_index=lambda **_: gain_writes.append("stiffness"),
        write_joint_damping_to_sim_index=lambda **_: gain_writes.append("damping"),
    )
    monkeypatch.setattr(SimulationManager, "activate_newton_actuator_path", lambda: activation_calls.append(True))

    control = NewtonActuatorControl(articulation)
    group_name = "explicit" if expected_native_groups else "implicit"
    native_groups = control.prepare_native_actuators(collection=None, actuator_cfgs={group_name: actuator_cfg})

    assert native_groups == expected_native_groups
    assert gain_writes == []
    if expected_native_groups:
        assert activation_calls == [True]
    else:
        assert not control.native_actuator_path_active
        assert not articulation._has_newton_actuators
        assert activation_calls == []


@pytest.mark.parametrize(
    ("actuator_cfg", "expected_mode", "expected_actuator_indices"),
    [
        (
            ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=0.0),
            JointTargetMode.POSITION,
            [0, 1],
        ),
        (
            ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=2.0),
            JointTargetMode.VELOCITY,
            [-2, -3],
        ),
        (
            ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=2.0),
            JointTargetMode.POSITION_VELOCITY,
            [0, -2, 1, -3],
        ),
        (
            ImplicitActuatorCfg(
                class_type=f"{__name__}:CustomDrive", joint_names_expr=[".*"], stiffness=10.0, damping=2.0
            ),
            JointTargetMode.POSITION_VELOCITY,
            [0, -2, 1, -3],
        ),
        (
            ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0),
            JointTargetMode.EFFORT,
            None,
        ),
        (
            IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=2.0),
            JointTargetMode.EFFORT,
            None,
        ),
    ],
)
def test_actuator_cfg_sets_newton_target_mode_before_solver_init(
    actuator_cfg, expected_mode, expected_actuator_indices
):
    """Resolve configured modes before finalization constructs MuJoCo actuators."""
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        articulation_root_prim_path="",
        actuators={"joint": actuator_cfg},
    )
    builder = _make_target_mode_builder(
        ["left_joint", "right_joint"], [JointTargetMode.NONE, JointTargetMode.NONE], [0.0, 0.0], [0.0, 0.0]
    )
    _configure_builder_joint_target_modes(builder, articulation_cfg)
    model = builder.finalize(device="cpu")
    solver = SolverMuJoCo(model, use_mujoco_cpu=True)
    assert model.joint_target_mode.numpy().tolist() == [int(expected_mode), int(expected_mode)]
    assert (
        solver.mjc_actuator_to_newton_idx.numpy().tolist() if solver.mjc_actuator_to_newton_idx is not None else None
    ) == expected_actuator_indices


def test_actuator_cfg_matches_explicit_descendant_articulation_root():
    """Match target modes against an explicitly configured descendant articulation root."""
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        articulation_root_prim_path="/base",
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=0.0)},
    )
    builder = _make_target_mode_builder(["joint"], [JointTargetMode.NONE], [0.0], [0.0])
    builder.articulation_label = ["/World/Env_0/Robot/base"]
    _configure_builder_joint_target_modes(builder, articulation_cfg)
    assert builder.joint_target_mode == [int(JointTargetMode.POSITION)]


def test_actuator_cfg_matches_clone_plan_root_expr(monkeypatch):
    """Match builder labels against the clone slot spelling clone-plan root resolution returns."""
    articulation_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=0.0)},
    )
    monkeypatch.setattr(
        "isaaclab_newton.assets.articulation.articulation.resolve_matching_prims_from_source",
        lambda *_args, **_kwargs: [(None, "/World/envs/env_[^/]+/Robot/base")],
    )
    builder = _make_target_mode_builder(["joint"], [JointTargetMode.NONE], [0.0], [0.0])
    builder.articulation_label = ["/World/envs/env_0/Robot/base"]
    _configure_builder_joint_target_modes(builder, articulation_cfg)
    assert builder.joint_target_mode == [int(JointTargetMode.POSITION)]


@pytest.mark.parametrize("joint_type", [JointType.FREE, JointType.FIXED])
def test_actuator_cfg_leaves_excluded_joint_types_imported(joint_type):
    """Leave target modes for free and fixed joints unchanged."""
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        articulation_root_prim_path="",
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=0.0)},
    )
    builder = _make_target_mode_builder(["joint"], [JointTargetMode.NONE], [0.0], [0.0])
    builder.joint_type[0] = joint_type
    _configure_builder_joint_target_modes(builder, articulation_cfg)
    assert builder.joint_target_mode == [int(JointTargetMode.NONE)]


def test_actuator_cfg_uses_imported_gain_for_none_stiffness():
    """Retain the imported stiffness when an implicit actuator config leaves it unset."""
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        articulation_root_prim_path="",
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=0.0)},
    )
    builder = _make_target_mode_builder(["joint"], [JointTargetMode.EFFORT], [10.0], [0.0])
    _configure_builder_joint_target_modes(builder, articulation_cfg)
    assert builder.joint_target_mode == [int(JointTargetMode.POSITION)]


def test_actuator_cfg_leaves_unconfigured_newton_target_modes_imported():
    """Leave target modes for DOFs outside an actuator group unchanged."""
    subset_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        articulation_root_prim_path="",
        actuators={
            "shoulder": ImplicitActuatorCfg(joint_names_expr=["left_shoulder"], stiffness=10.0, damping=0.0),
        },
    )
    builder = _make_target_mode_builder(
        ["left_shoulder", "right_shoulder"],
        [JointTargetMode.NONE, JointTargetMode.VELOCITY],
        [0.0, 0.0],
        [0.0, 2.0],
    )
    _configure_builder_joint_target_modes(builder, subset_cfg)
    assert builder.joint_target_mode == [int(JointTargetMode.POSITION), int(JointTargetMode.VELOCITY)]


@pytest.mark.parametrize(
    ("stiffness", "damping", "expected_modes"),
    [
        ({"left_joint": 10.0}, 0.0, [JointTargetMode.POSITION, JointTargetMode.EFFORT]),
        ({"left_joint": 10.0}, {"right_joint": 2.0}, [JointTargetMode.POSITION, JointTargetMode.VELOCITY]),
    ],
)
def test_actuator_cfg_aligns_partial_dictionary_gains_by_joint_name(stiffness, damping, expected_modes):
    """Resolve sparse stiffness and damping dictionaries independently by joint name."""
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        articulation_root_prim_path="",
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=stiffness, damping=damping)},
    )
    builder = _make_target_mode_builder(
        ["left_joint", "right_joint"], [JointTargetMode.NONE, JointTargetMode.NONE], [0.0, 0.0], [0.0, 0.0]
    )
    _configure_builder_joint_target_modes(builder, articulation_cfg)
    assert builder.joint_target_mode == [int(mode) for mode in expected_modes]


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.parametrize("articulation_type", ["single_joint_explicit"])
def test_branching_fixture_physx_ordering_reorders_newton_to_bfs(sim, device, gravity_enabled, articulation_type):
    """Resolve the documented Newton ``joint_ordering="physx"`` sim-to-sim workflow on a branching asset.

    Mirrors :func:`isaaclab_physx.test.assets.test_articulation.test_branching_fixture_resolves_distinct_conventions`
    with the backend roles swapped: here the live backend is Newton (depth-first, so its native view is
    the MJWarp order), and the request is ``physx``/``body_ordering="physx"``. Cross-backend discovery must
    resolve the breadth-first PhysX order and reorder the public joint/body axes to it. This is the headline
    workflow documented in
    ``docs/source/overview/core-concepts/physical-backends/joint_and_body_ordering.rst``.

    The branching fixture is shared between both backends; a copy lives in this package's
    test data directory so the two backends assert against the same ground-truth asset.

    The same articulation also checks the MJWarp-order emulation used for cross-backend discovery and that
    selected inertial-property writes keep Newton's inverse arrays current under the body ordering.
    """
    fixture_path = Path(__file__).parent / "data" / "articulation_ordering_branching.usda"
    sim_utils.create_prim("/World/Env_0", "Xform")
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_0/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(fixture_path)),
        actuators={},
        joint_ordering="physx",
        body_ordering="physx",
    )
    clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), (articulation_cfg,), 1, 0.0)
    articulation = Articulation(articulation_cfg)

    replicate(sim.get_clone_plan())
    sim.reset()
    assert articulation.is_initialized

    # Newton's native traversal is depth-first, so the live backend view already reflects MJWarp order.
    assert tuple(articulation.backend_joint_names) == BRANCHING_MJWARP_JOINT_NAMES
    assert tuple(articulation.backend_body_names) == BRANCHING_MJWARP_BODY_NAMES

    # The same-backend "mjwarp" request takes an identity fast path, so also force the cross-backend
    # emulation (the temporary Newton USD builder a PhysX-backed articulation would use) and compare its
    # independently rebuilt view against the live backend view. BFS and DFS differ on this branching fixture.
    emulated_names = ordering_resolvers._get_mjwarp_names_from_newton_usd_builder(articulation)
    assert emulated_names is not None
    assert emulated_names["joint"] == tuple(articulation.backend_joint_names)
    assert emulated_names["body"] == tuple(articulation.backend_body_names)
    assert get_articulation_name_ordering(articulation, "mjwarp", kind="joint") == tuple(
        articulation.backend_joint_names
    )
    assert get_articulation_name_ordering(articulation, "mjwarp", kind="body") == tuple(articulation.backend_body_names)

    # Cross-backend discovery (bypassing the same-backend fast path) resolves the breadth-first PhysX order.
    assert get_articulation_name_ordering(articulation, "physx", kind="joint") == BRANCHING_PHYSX_JOINT_NAMES
    assert get_articulation_name_ordering(articulation, "physx", kind="body") == BRANCHING_PHYSX_BODY_NAMES

    # The requested PhysX ordering reorders the public joint/body axes to the BFS convention.
    assert tuple(articulation.joint_names) == BRANCHING_PHYSX_JOINT_NAMES
    assert tuple(articulation.body_names) == BRANCHING_PHYSX_BODY_NAMES
    assert articulation.joint_ordering is not None
    assert articulation.body_ordering is not None

    # Selected mass and inertia writes with int64 selectors update Newton's inverse arrays in backend order.
    env_ids = torch.tensor([0], dtype=torch.int64, device=device)
    body_ids = torch.tensor([2, articulation.num_bodies - 1], dtype=torch.int64, device=device)
    backend_body_ids = torch.tensor(
        [articulation.data.body_ordering.user_to_backend_indices[index] for index in body_ids.tolist()],
        dtype=torch.int64,
        device=device,
    )
    assert backend_body_ids[0] != body_ids[0]
    model = SimulationManager.get_model()

    masses = articulation.data.body_mass.torch[env_ids][:, body_ids].clone() + torch.tensor([[1.0, 2.0]], device=device)
    articulation.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
    model_inv_mass = wp.to_torch(articulation.root_view.get_attribute("body_inv_mass", model)[:, 0])
    torch.testing.assert_close(model_inv_mass[env_ids][:, backend_body_ids], masses.reciprocal())

    inertia_matrices = torch.diag_embed(torch.tensor([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]], device=device))
    articulation.set_inertias_index(inertias=inertia_matrices.reshape(1, 2, 9), env_ids=env_ids, body_ids=body_ids)
    model_inv_inertia = wp.to_torch(articulation.root_view.get_attribute("body_inv_inertia", model)[:, 0])
    torch.testing.assert_close(model_inv_inertia[env_ids][:, backend_body_ids], torch.linalg.inv(inertia_matrices))


def test_num_shapes_per_body_follows_public_body_order() -> None:
    """Align Newton shape counts with the public body-name axis."""

    class _ShapeCountSurface:
        backend_num_shapes_per_body = Articulation.backend_num_shapes_per_body
        num_shapes_per_body = Articulation.num_shapes_per_body

    articulation = _ShapeCountSurface()
    articulation._num_shapes_per_body_backend = None
    articulation._root_view = SimpleNamespace(
        body_shapes=((), (object(), object()), (object(), object(), object())),
    )
    articulation.body_ordering = SimpleNamespace(
        user_to_backend_indices=(2, 0, 1),
    )

    assert articulation.num_shapes_per_body == [3, 0, 2]


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["anymal"])  # consumed by the sim fixture
@pytest.mark.parametrize("use_newton_actuators", [True])  # consumed by the sim fixture
def test_newton_native_actuator_gain_write_maps_public_joint_subset_to_backend(
    sim, articulation_type, use_newton_actuators, device
):
    """Map selected public joint IDs to Newton-controller columns."""
    articulation_cfg = generate_articulation_cfg("anymal").replace(
        actuators={
            "legs": IdealPDActuatorCfg(
                joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
                stiffness=40.0,
                damping=5.0,
                actuator_effort_limit=80.0,
            )
        },
        joint_ordering=tuple(reversed(ANYMAL_C_PHYSX_JOINT_NAMES)),
    )
    articulation, _ = generate_articulation(articulation_cfg, 2, device=sim.device)
    replicate(sim.get_clone_plan())
    sim.reset()
    assert articulation.joint_ordering is not None
    assert articulation.newton_actuator_adapter is not None

    def gather_stiffness() -> torch.Tensor:
        stiffness = torch.zeros(
            (articulation.num_instances, articulation.num_joints),
            device=articulation.device,
        )
        for actuator in articulation.newton_actuator_adapter.actuators:
            if hasattr(actuator.controller, "kp"):
                stiffness += wp.to_torch(
                    articulation.root_view.get_actuator_parameter(actuator, actuator.controller, "kp")
                )
        return stiffness

    stiffness_before = gather_stiffness()
    env_ids = torch.tensor([1], device=articulation.device, dtype=torch.long)
    joint_ids = torch.tensor([1, 6, 10], device=articulation.device, dtype=torch.long)
    stiffness = torch.tensor([[101.0, 106.0, 110.0]], device=articulation.device)

    with pytest.warns(DeprecationWarning, match="write_actuator_stiffness_to_sim"):
        articulation.write_actuator_stiffness_to_sim(
            stiffness=stiffness,
            env_ids=env_ids,
            joint_ids=joint_ids,
        )

    backend_joint_ids = torch.tensor(
        articulation.joint_ordering.user_to_backend_indices,
        device=articulation.device,
        dtype=torch.long,
    )[joint_ids]
    expected_stiffness = stiffness_before.clone()
    expected_stiffness[env_ids.unsqueeze(1), backend_joint_ids.unsqueeze(0)] = stiffness
    torch.testing.assert_close(gather_stiffness(), expected_stiffness)


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_newton_ordered_body_state_cache_invalidates_on_same_timestamp_root_write(
    sim, num_articulations, device, gravity_enabled, articulation_type
):
    """Refresh ordered body pose and velocity after root writes at the current simulation timestamp."""
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type).replace(
        body_ordering=_ANYMAL_C_ROOT_PRESERVING_REVERSED_BODY_NAMES
    )
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    replicate(sim.get_clone_plan())
    sim.reset()
    sim.step()
    articulation.update(sim.cfg.dt)

    data = articulation.data
    assert data.body_ordering is not None
    root_body_idx = articulation.find_bodies("base")[0][0]
    sim_timestamp = data._sim_timestamp

    cached_body_pose = data.body_link_pose_w.torch[:, root_body_idx].clone()
    written_root_pose = data.root_link_pose_w.torch.clone()
    written_root_pose[:, 0] += 0.25
    articulation.write_root_link_pose_to_sim_index(root_pose=written_root_pose)
    assert data._sim_timestamp == sim_timestamp
    torch.testing.assert_close(data.root_link_pose_w.torch, written_root_pose)
    SimulationManager.forward()  # Another consumer may resolve shared FK before this view reads.
    refreshed_body_pose = data.body_link_pose_w.torch[:, root_body_idx]
    torch.testing.assert_close(refreshed_body_pose, written_root_pose)
    assert not torch.equal(refreshed_body_pose, cached_body_pose)

    # Populate the velocity cache after the pose write so only the velocity write can invalidate it.
    cached_body_vel = data.body_com_vel_w.torch[:, root_body_idx].clone()
    written_root_vel = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], device=device, dtype=cached_body_vel.dtype)
    articulation.write_root_com_velocity_to_sim_index(root_velocity=written_root_vel)
    assert data._sim_timestamp == sim_timestamp
    torch.testing.assert_close(data.root_com_vel_w.torch, written_root_vel)
    refreshed_body_vel = data.body_com_vel_w.torch[:, root_body_idx]
    torch.testing.assert_close(refreshed_body_vel, written_root_vel)
    assert not torch.equal(refreshed_body_vel, cached_body_vel)


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.parametrize("ordering_mode", ["none", "reversed"])
@pytest.mark.parametrize("use_newton_actuators", [False])
def test_newton_ordered_state_caches_invalidate_on_rebind(
    sim, num_articulations, device, gravity_enabled, articulation_type, ordering_mode, use_newton_actuators
):
    """Rebind public state to recreated Newton arrays and invalidate ordered caches."""
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    if ordering_mode == "reversed":
        articulation_cfg = articulation_cfg.replace(
            joint_ordering=tuple(reversed(PANDA_JOINT_NAMES)),
            body_ordering=PANDA_ROOT_PRESERVING_REVERSED_BODY_NAMES,
        )
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    replicate(sim.get_clone_plan())
    sim.reset()
    assert articulation.is_initialized
    has_ordering = ordering_mode == "reversed"
    assert (articulation.data.joint_ordering is not None) is has_ordering
    assert (articulation.data.body_ordering is not None) is has_ordering

    sim.step()
    articulation.update(sim.cfg.dt)

    data = articulation.data
    primed_joint_vel_values = (
        np.arange(np.prod(data._sim_bind_joint_vel.shape), dtype=np.float32).reshape(data._sim_bind_joint_vel.shape)
        + 100.0
    )
    body_velocity_shape = (*data._sim_bind_body_com_vel_w.shape, 6)
    primed_body_com_vel_values = (
        np.arange(np.prod(body_velocity_shape), dtype=np.float32).reshape(body_velocity_shape) + 200.0
    )
    data._sim_bind_joint_vel.assign(
        wp.array(primed_joint_vel_values, dtype=wp.float32, device=data._sim_bind_joint_vel.device)
    )
    data._sim_bind_body_com_vel_w.assign(
        wp.array(
            primed_body_com_vel_values,
            dtype=wp.spatial_vectorf,
            device=data._sim_bind_body_com_vel_w.device,
        )
    )
    # The raw sim-bind writes above simulate the solver advancing state; in the
    # real pipeline the post-step callback republishes the passthrough shadows in
    # the same step. Mirror that here so ``joint_acc`` (which reads the passthrough
    # ``joint_vel`` shadow) observes the primed backend state. No-op under identity
    # ordering, where the getters alias the sim-bound arrays directly.
    data._refresh_user_order_state()
    data.update(sim.cfg.dt)
    primed_joint_acc = data.joint_acc.warp.numpy().copy()
    primed_body_com_acc_w = data.body_com_acc_w.warp.numpy().copy()
    assert data._joint_acc.timestamp == data._sim_timestamp
    assert data._body_com_acc_w.timestamp == data._sim_timestamp
    assert np.any(primed_joint_acc != 0.0)
    assert np.any(primed_body_com_acc_w != 0.0)

    public_to_binding = {
        "joint_pos": "_sim_bind_joint_pos",
        "joint_vel": "_sim_bind_joint_vel",
        "body_link_pose_w": "_sim_bind_body_link_pose_w",
        "body_com_vel_w": "_sim_bind_body_com_vel_w",
    }
    public_to_shadow = {
        "joint_pos": "_joint_pos_user",
        "joint_vel": "_joint_vel_user",
        "body_link_pose_w": "_body_link_pose_w_user",
        "body_com_vel_w": "_body_com_vel_w_user",
    }
    old_bindings = {name: getattr(data, name) for name in public_to_binding.values()}
    old_binding_ptrs = {name: int(array.ptr) for name, array in old_bindings.items()}
    old_public_proxies = {name: getattr(data, name) for name in public_to_binding}
    implicit_executor = articulation.actuators._implicit_executor
    assert implicit_executor is not None
    actuator_state_inputs = [implicit_executor.kernel_inputs]
    data.joint_pos_limits.torch.clone()
    # The Tier-1 state shadows are plain wp.arrays (no timestamp): they are
    # allocated for non-identity ordering and stay ``None`` for identity ordering.
    if has_ordering:
        for cache_name in _NEWTON_USER_ORDER_STATE_CACHES:
            assert getattr(data, cache_name) is not None
    else:
        for cache_name in _NEWTON_USER_ORDER_STATE_CACHES:
            assert getattr(data, cache_name) is None

    old_state = SimulationManager.get_state_0()
    old_model = SimulationManager.get_model()
    new_state = copy(old_state)
    new_model = copy(old_model)

    joint_q_values = np.arange(len(old_state.joint_q), dtype=np.float32) + 1000.0
    joint_qd_values = np.arange(len(old_state.joint_qd), dtype=np.float32) + 2000.0
    body_indices = np.arange(len(old_state.body_q), dtype=np.float32)[:, None]
    body_q_values = np.zeros((len(old_state.body_q), 7), dtype=np.float32)
    body_q_values[:, :3] = 3000.0 + 10.0 * body_indices + np.arange(3, dtype=np.float32)
    body_q_values[:, 6] = 1.0
    body_qd_values = 4000.0 + 10.0 * body_indices + np.arange(6, dtype=np.float32)
    limit_indices = np.arange(len(old_model.joint_limit_lower), dtype=np.float32)
    limit_lower_values = -5000.0 - limit_indices
    limit_upper_values = 5000.0 + limit_indices

    new_state.joint_q = wp.array(joint_q_values, dtype=wp.float32, device=old_state.joint_q.device)
    new_state.joint_qd = wp.array(joint_qd_values, dtype=wp.float32, device=old_state.joint_qd.device)
    new_state.body_q = wp.array(body_q_values, dtype=wp.transformf, device=old_state.body_q.device)
    new_state.body_qd = wp.array(body_qd_values, dtype=wp.spatial_vectorf, device=old_state.body_qd.device)
    new_model.joint_limit_lower = wp.array(
        limit_lower_values, dtype=wp.float32, device=old_model.joint_limit_lower.device
    )
    new_model.joint_limit_upper = wp.array(
        limit_upper_values, dtype=wp.float32, device=old_model.joint_limit_upper.device
    )
    SimulationManager.backend.state_0 = new_state
    SimulationManager.backend.model = new_model

    body_velocities = articulation.root_view.get_link_velocities(new_state)
    assert body_velocities is not None
    new_source_bindings = {
        "_sim_bind_joint_pos": articulation.root_view.get_dof_positions(new_state)[:, 0],
        "_sim_bind_joint_vel": articulation.root_view.get_dof_velocities(new_state)[:, 0],
        "_sim_bind_body_link_pose_w": articulation.root_view.get_link_transforms(new_state)[:, 0],
        "_sim_bind_body_com_vel_w": body_velocities[:, 0],
        "_sim_bind_joint_pos_limits_lower": articulation.root_view.get_attribute("joint_limit_lower", new_model)[:, 0],
        "_sim_bind_joint_pos_limits_upper": articulation.root_view.get_attribute("joint_limit_upper", new_model)[:, 0],
    }
    for binding_name, new_source in new_source_bindings.items():
        if binding_name in old_binding_ptrs:
            assert int(new_source.ptr) != old_binding_ptrs[binding_name]

    data._create_simulation_bindings()

    for binding_name, old_binding in old_bindings.items():
        rebound = getattr(data, binding_name)
        assert rebound is not old_binding
        assert int(rebound.ptr) != old_binding_ptrs[binding_name]
        assert int(rebound.ptr) == int(new_source_bindings[binding_name].ptr)

    for inputs in actuator_state_inputs:
        assert inputs[3].ptr == data.joint_pos.warp.ptr
        assert inputs[4].ptr == data.joint_vel.warp.ptr

    assert data._joint_acc.timestamp == -1.0
    assert data._body_com_acc_w.timestamp == -1.0

    joint_user_to_backend = (
        np.asarray(articulation.joint_ordering.user_to_backend_indices)
        if articulation.joint_ordering is not None
        else np.arange(articulation.num_joints)
    )
    body_user_to_backend = (
        np.asarray(articulation.body_ordering.user_to_backend_indices)
        if articulation.body_ordering is not None
        else np.arange(articulation.num_bodies)
    )
    expected_previous_joint_vel = new_source_bindings["_sim_bind_joint_vel"].numpy()[:, joint_user_to_backend]
    expected_previous_body_com_vel = new_source_bindings["_sim_bind_body_com_vel_w"].numpy()
    np.testing.assert_array_equal(data._previous_joint_vel.numpy(), expected_previous_joint_vel)
    np.testing.assert_array_equal(data._previous_body_com_vel.numpy(), expected_previous_body_com_vel)

    joint_acc = data.joint_acc.warp.numpy()
    body_com_acc_w = data.body_com_acc_w.warp.numpy()
    np.testing.assert_array_equal(joint_acc, np.zeros_like(expected_previous_joint_vel))
    np.testing.assert_array_equal(
        body_com_acc_w,
        np.zeros_like(expected_previous_body_com_vel[:, body_user_to_backend]),
    )
    assert data._joint_acc.timestamp == data._sim_timestamp
    assert data._body_com_acc_w.timestamp == data._sim_timestamp
    assert not np.array_equal(joint_acc, primed_joint_acc)
    assert not np.array_equal(body_com_acc_w, primed_body_com_acc_w)

    expected_public = {
        "joint_pos": new_source_bindings["_sim_bind_joint_pos"].numpy()[:, joint_user_to_backend],
        "joint_vel": new_source_bindings["_sim_bind_joint_vel"].numpy()[:, joint_user_to_backend],
        "body_link_pose_w": new_source_bindings["_sim_bind_body_link_pose_w"].numpy()[:, body_user_to_backend],
        "body_com_vel_w": new_source_bindings["_sim_bind_body_com_vel_w"].numpy()[:, body_user_to_backend],
    }
    for property_name, expected in expected_public.items():
        proxy = getattr(data, property_name)
        assert proxy is not old_public_proxies[property_name]
        binding_name = public_to_binding[property_name]
        assert int(proxy.warp.ptr) != old_binding_ptrs[binding_name]
        if has_ordering:
            shadow = getattr(data, public_to_shadow[property_name])
            assert int(proxy.warp.ptr) == int(shadow.ptr)
        else:
            assert int(proxy.warp.ptr) == int(getattr(data, binding_name).ptr)
        np.testing.assert_array_equal(proxy.warp.numpy(), expected)

    expected_limits = np.stack(
        (
            new_source_bindings["_sim_bind_joint_pos_limits_lower"].numpy()[:, joint_user_to_backend],
            new_source_bindings["_sim_bind_joint_pos_limits_upper"].numpy()[:, joint_user_to_backend],
        ),
        axis=-1,
    )
    np.testing.assert_array_equal(data.joint_pos_limits.warp.numpy(), expected_limits)


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("gravity_enabled", [True])
@pytest.mark.parametrize("articulation_type", ["anymal"])
@pytest.mark.parametrize("ordering_mode", ["none", "reversed"])
@pytest.mark.parametrize("use_newton_actuators", [False])
def test_newton_rebind_preserves_lab_owned_actuator_gains(
    sim, num_articulations, device, gravity_enabled, articulation_type, ordering_mode, use_newton_actuators
):
    """Keep Lab-owned actuator gains across a rebind that re-seeds the solver's sim gains.

    Part 2 (D3) regression: named actuator groups own their actuator kp/kd; the solver's
    sim gains are deliberately zeroed for explicit DOFs. A full sim reset recreates the solver arrays.
    Rebind must NOT resync the actuator-owned values from freshly rebuilt (here: sentinel) solver gains.
    ``none`` is the identity-ordering control that must pass with or without the fix.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type).replace(
        actuators={
            "legs": IdealPDActuatorCfg(
                joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
                stiffness=40.0,
                damping=5.0,
                actuator_effort_limit=80.0,
            )
        },
    )
    if ordering_mode == "reversed":
        articulation_cfg = articulation_cfg.replace(joint_ordering=tuple(reversed(ANYMAL_C_PHYSX_JOINT_NAMES)))
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)
    replicate(sim.get_clone_plan())
    sim.reset()
    assert articulation.is_initialized

    has_ordering = ordering_mode == "reversed"
    data = articulation.data
    assert (data.joint_ordering is not None) is has_ordering

    # Prime: explicit (IdealPD) actuators keep their PD in actuator-owned records,
    # while the solver's sim gains are zeroed so it applies no PD on these DOFs.
    np.testing.assert_allclose(articulation.actuators["legs"].stiffness.cpu().numpy(), 40.0)
    np.testing.assert_allclose(articulation.actuators["legs"].damping.cpu().numpy(), 5.0)
    np.testing.assert_allclose(data._sim_bind_joint_stiffness_sim.numpy(), 0.0)
    np.testing.assert_allclose(data._sim_bind_joint_damping_sim.numpy(), 0.0)

    # Simulate a full sim reset: shallow-copy the model, swap the joint gain arrays
    # for sentinel-filled arrays (standing in for whatever the solver rebuilds), and
    # rebind the data-side sim bindings.
    old_model = SimulationManager.get_model()
    new_model = copy(old_model)
    sentinel_ke = 12345.0
    sentinel_kd = 678.0
    new_model.joint_target_ke = wp.array(
        np.full(len(old_model.joint_target_ke), sentinel_ke, dtype=np.float32),
        dtype=wp.float32,
        device=old_model.joint_target_ke.device,
    )
    new_model.joint_target_kd = wp.array(
        np.full(len(old_model.joint_target_kd), sentinel_kd, dtype=np.float32),
        dtype=wp.float32,
        device=old_model.joint_target_kd.device,
    )
    SimulationManager.backend.model = new_model
    data._create_simulation_bindings()

    # The actuator-owned gains must survive the rebind unchanged...
    np.testing.assert_allclose(articulation.actuators["legs"].stiffness.cpu().numpy(), 40.0)
    np.testing.assert_allclose(articulation.actuators["legs"].damping.cpu().numpy(), 5.0)
    # ...while the sim-owned mirrors track the solver's freshly seeded (sentinel) gains.
    if has_ordering:
        np.testing.assert_allclose(data._joint_stiffness_user.numpy(), sentinel_ke)
        np.testing.assert_allclose(data._joint_damping_user.numpy(), sentinel_kd)
    else:
        np.testing.assert_allclose(data._sim_bind_joint_stiffness_sim.numpy(), sentinel_ke)
        np.testing.assert_allclose(data._sim_bind_joint_damping_sim.numpy(), sentinel_kd)


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("gravity_enabled", [True])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_newton_post_step_hook_publishes_ordered_state_and_deregisters(
    sim, num_articulations, device, gravity_enabled, articulation_type
):
    """Republish the user-order Tier-1 shadows inside the sim step, without any read.

    Part 1 (D1) regression: with non-identity ordering the passthrough state getters no
    longer reorder on read; the post-step callback republishes the shadows from live
    backend state inside the stepped region. We deliberately clobber the four shadows,
    step the simulation WITHOUT reading any state property, and assert the shadows again
    equal the reordered backend state -- which can only hold if the hook ran inside the
    step. Under the lazy design (no hook), only a property read would refresh them, so
    the clobbered shadows would stay stale and the assertions would fail.

    Ships the eager-mode invariant variant: CUDA-graph capture is not reliably reachable
    from this CPU test harness, and this invariant directly proves the in-step republish.

    Finally checks that ``_clear_callbacks`` deregisters the hook without touching other callbacks.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type).replace(
        actuators={"legs": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=40.0, damping=5.0)},
        joint_ordering=tuple(reversed(ANYMAL_C_PHYSX_JOINT_NAMES)),
        body_ordering=_ANYMAL_C_ROOT_PRESERVING_REVERSED_BODY_NAMES,
    )
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)
    replicate(sim.get_clone_plan())
    sim.reset()
    assert articulation.is_initialized

    data = articulation.data
    assert data.joint_ordering is not None
    assert data.body_ordering is not None
    joint_u2b = np.asarray(articulation.joint_ordering.user_to_backend_indices)
    body_u2b = np.asarray(articulation.body_ordering.user_to_backend_indices)

    # Clobber every Tier-1 shadow with a large sentinel so a stale (unrepublished) shadow
    # is unmistakably detectable -- the true backend joint/velocity state sits near zero,
    # so a plain zero-fill would coincide with it. Then step WITHOUT touching joint_pos /
    # joint_vel / body_link_pose_w / body_com_vel_w.
    data._joint_pos_user.fill_(1000.0)
    data._joint_vel_user.fill_(1000.0)
    data._body_link_pose_w_user.fill_(wp.transformf(1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0, 1.0))
    data._body_com_vel_w_user.fill_(wp.spatial_vectorf(1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0))
    sim.step()

    np.testing.assert_allclose(data._joint_pos_user.numpy(), data._sim_bind_joint_pos.numpy()[:, joint_u2b])
    np.testing.assert_allclose(data._joint_vel_user.numpy(), data._sim_bind_joint_vel.numpy()[:, joint_u2b])
    np.testing.assert_allclose(
        data._body_link_pose_w_user.numpy(), data._sim_bind_body_link_pose_w.numpy()[:, body_u2b]
    )
    np.testing.assert_allclose(data._body_com_vel_w_user.numpy(), data._sim_bind_body_com_vel_w.numpy()[:, body_u2b])

    # ``_clear_callbacks`` must deregister exactly this hook so it does not leak on the class-level list,
    # and leave an unrelated callback (standing in for another articulation's hook) untouched.
    registered_callback = articulation._post_step_callback
    assert registered_callback is not None
    assert registered_callback in SimulationManager._post_step_callbacks

    def _other_callback() -> None:
        return None

    SimulationManager.register_post_step_callback(_other_callback)
    articulation._clear_callbacks()

    assert articulation._post_step_callback is None
    assert registered_callback not in SimulationManager._post_step_callbacks
    assert _other_callback in SimulationManager._post_step_callbacks


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.parametrize("articulation_type", ["anymal"])
@pytest.mark.parametrize("use_newton_actuators", [False, True])
@pytest.mark.parametrize("ordering_mode", ["none", "reversed"])
def test_write_data_to_sim_writes_joint_targets_in_backend_order(
    sim, num_articulations, device, gravity_enabled, articulation_type, use_newton_actuators, ordering_mode
):
    """Write the published joint position targets into the backend buffer in backend joint order.

    Explicit actuators on the Newton-actuator path publish their raw targets; implicit actuators on the Lab
    path publish the processed targets. Both must reach the solver-bound buffer permuted to backend order.
    """
    actuator_cfg = (
        IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=40.0, damping=5.0)
        if use_newton_actuators
        else ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=40.0, damping=5.0)
    )
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type).replace(
        actuators={"legs": actuator_cfg},
    )
    if ordering_mode == "reversed":
        articulation_cfg = articulation_cfg.replace(joint_ordering=tuple(reversed(ANYMAL_C_PHYSX_JOINT_NAMES)))
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)
    replicate(sim.get_clone_plan())
    sim.reset()
    assert articulation.is_initialized
    assert (articulation.data.joint_ordering is not None) is (ordering_mode == "reversed")
    assert articulation._has_newton_actuators is use_newton_actuators

    # Distinct per-joint targets away from the defaults, so a skipped or unpermuted write is visible.
    target = articulation.data.default_joint_pos.torch.clone()
    target += 0.01 * torch.arange(1, articulation.num_joints + 1, device=device)
    articulation.set_joint_position_target_index(target=target)
    articulation.write_data_to_sim()

    source = (
        articulation.actuators.target_command.position.torch
        if use_newton_actuators
        else articulation.actuators.output_command.position.torch
    )
    torch.testing.assert_close(source, target)
    user_to_backend = (
        list(articulation.joint_ordering.user_to_backend_indices)
        if articulation.joint_ordering is not None
        else list(range(articulation.num_joints))
    )
    expected_backend_target = torch.empty_like(source)
    expected_backend_target[:, user_to_backend] = source
    torch.testing.assert_close(wp.to_torch(articulation.data._sim_bind_joint_position_target), expected_backend_target)


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("num_articulations", [3])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["humanoid"])
def test_gravity_vec_w_tracks_model_gravity(sim, num_articulations, device, add_ground_plane, articulation_type):
    """Per-env mutations to Newton's ``model.gravity`` reach ``GRAVITY_VEC_W`` and ``projected_gravity_b``.

    Regression for the pre-fix snapshot: ``GRAVITY_VEC_W`` used to be env 0's
    gravity broadcast to every env, hiding per-env gravity randomization (e.g.
    :class:`~isaaclab.envs.mdp.randomize_physics_scene_gravity`).

    The humanoid's articulation root sits on a rigid body, so this also checks floating-base initialization.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type, stiffness=0.0, damping=0.0)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device, add_ground_plane=True)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    replicate(sim.get_clone_plan())
    sim.reset()

    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that is floating base
    assert not articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert articulation.data.root_pos_w.torch.shape == (num_articulations, 3)
    assert articulation.data.root_quat_w.torch.shape == (num_articulations, 4)
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 21)

    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert getattr(actuator, "is_implicit_model", False) == is_implicit_model_cfg

    # GRAVITY_VEC_W must share storage with Newton's per-env gravity array.
    model = SimulationManager.get_model()
    model_gravity_arr = model.gravity[: model.world_count]
    global_gravity = wp.to_torch(model.gravity)[-1].clone()
    assert articulation.data.GRAVITY_VEC_W.warp.ptr == model_gravity_arr.ptr
    assert articulation.data.GRAVITY_VEC_W.shape == (num_articulations,)

    # Mutate model.gravity per-env in place, as randomize_physics_scene_gravity does.
    new_gravity = torch.tensor(
        [[0.1 * (i + 1), 0.2 * (i + 1), -3.0 - float(i)] for i in range(num_articulations)],
        device=device,
        dtype=torch.float32,
    )
    wp.to_torch(model_gravity_arr).copy_(new_gravity)
    SimulationManager.add_model_change(ModelFlags.MODEL_PROPERTIES)

    # Live view: new per-env values are visible immediately, no invalidation step.
    torch.testing.assert_close(articulation.data.GRAVITY_VEC_W.torch, new_gravity)
    torch.testing.assert_close(wp.to_torch(model.gravity)[-1], global_gravity)

    # Recompute the lazily-cached projected_gravity_b without sim.step (which would
    # drift root orientation from the reset state). Project against the same quat
    # buffer the kernel reads so the expectation holds for any default orientation.
    articulation.update(sim.cfg.dt)
    root_quat = articulation.data.root_link_quat_w.torch
    expected = math_utils.quat_apply_inverse(root_quat, torch.nn.functional.normalize(new_gravity, dim=-1))
    torch.testing.assert_close(articulation.data.projected_gravity_b.torch, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("articulation_type", ["panda"])
def test_fixed_base_reports_body_velocities(sim, num_articulations, device, articulation_type):
    """Test that fixed-base articulations report live body velocities while their joints move.

    Regression test: the fixed-base fallback in ``_create_buffers`` zeroed the body
    center-of-mass velocity binding together with the (genuinely unavailable) root velocity,
    so :attr:`body_lin_vel_w` and :attr:`body_ang_vel_w` read zeros for every fixed-base
    robot regardless of motion.

    This test verifies that:
    1. The articulation is initialized as fixed base with correctly shaped buffers
    2. Commanding a joint-space motion moves the bodies (finite difference of positions) while the root holds
       its default state
    3. The reported body velocities track the finite-difference ground truth

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, translations = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    # Play sim
    replicate(sim.get_clone_plan())
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

    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert getattr(actuator, "is_implicit_model", False) == is_implicit_model_cfg

    # the root holds its default state as it is fixed base
    default_root_pose = articulation.data.default_root_pose.torch.clone()
    default_root_pose[:, :3] = default_root_pose[:, :3] + translations
    default_root_vel = articulation.data.default_root_vel.torch.clone()

    # command a step away from the default pose so the distal bodies move
    joint_pos_target = articulation.data.default_joint_pos.torch.clone()
    joint_pos_target[:, 1] += 0.5
    prev_body_pos = articulation.data.body_link_pos_w.torch.clone()
    reported_max = []
    fin_diff_max = []
    for _ in range(20):
        articulation.set_joint_position_target(joint_pos_target)
        articulation.write_data_to_sim()
        sim.step()
        articulation.update(sim.cfg.dt)
        torch.testing.assert_close(articulation.data.root_link_pose_w.torch, default_root_pose)
        torch.testing.assert_close(articulation.data.root_com_vel_w.torch, default_root_vel)
        body_pos = articulation.data.body_link_pos_w.torch
        reported_max.append(articulation.data.body_lin_vel_w.torch.norm(dim=-1).amax())
        fin_diff_max.append(((body_pos - prev_body_pos) / sim.cfg.dt).norm(dim=-1).amax())
        prev_body_pos = body_pos.clone()
    reported_max = torch.stack(reported_max).amax()
    fin_diff_max = torch.stack(fin_diff_max).amax()

    # the commanded motion genuinely moves the bodies
    assert fin_diff_max > 0.1
    # and the reported body velocities track it (identically zero under the regression)
    assert reported_max > 0.5 * fin_diff_max


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("articulation_type", ["shadow_hand"])
def test_hand_with_tendons_initializes_and_targets_only_given_envs(sim, num_articulations, device, articulation_type):
    """Initialize a fixed-base hand with tendons; a tendon command for one environment must leave the others alone.

    ``set_fixed_tendon_position_target_index`` is declared backend-neutral and documented to accept
    partial data. Newton took ``env_ids`` and never forwarded it, so a partial command was sized
    against every instance and raised rather than commanding the environment asked for.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    replicate(sim.get_clone_plan())
    sim.reset()
    assert articulation.is_initialized
    assert articulation.is_fixed_base
    assert articulation.num_fixed_tendons > 0
    assert articulation.data.root_pos_w.torch.shape == (num_articulations, 3)
    assert articulation.data.root_quat_w.torch.shape == (num_articulations, 4)
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 24)
    assert articulation.data.body_mass.torch.shape == (num_articulations, articulation.num_bodies)
    assert articulation.data.body_inertia.torch.shape == (num_articulations, articulation.num_bodies, 9)
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert getattr(actuator, "is_implicit_model", False) == is_implicit_model_cfg

    target = torch.full((1, articulation.num_fixed_tendons), 1.0, dtype=torch.float32, device=device)
    articulation.set_fixed_tendon_position_target_index(target=target, env_ids=[0])

    for _ in range(30):
        articulation.write_data_to_sim()
        sim.step()
        articulation.update(sim.cfg.dt)

    # Both environments start from the same pose under the same gravity, so any divergence comes
    # from the command -- and identical poses would mean it reached both.
    commanded, untouched = articulation.data.joint_pos.torch[0], articulation.data.joint_pos.torch[1]
    assert not torch.allclose(commanded, untouched)
    # Each Shadow Hand tendon ``rh_XFJ0`` is the sum of joints ``rh_XFJ1`` and ``rh_XFJ2``, so the
    # commanded environment's tendon lengths must have moved toward the 1.0 target and away from
    # the uncommanded environment, which the actuator holds at its 0.0 control.
    for tendon_name in articulation.fixed_tendon_names:
        joint_ids, _ = articulation.find_joints([tendon_name[:-1] + "1", tendon_name[:-1] + "2"])
        assert commanded[joint_ids].sum() > untouched[joint_ids].sum()


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("articulation_type", ["shadow_hand"])
def test_fixed_tendon_properties_reach_solver(sim, num_articulations, device, articulation_type):
    """Written fixed tendon stiffness, damping, and position limits reach the MuJoCo solver.

    Covers both the index and the mask setters and writers.
    """
    articulation, _ = generate_articulation(
        generate_articulation_cfg(articulation_type=articulation_type), num_articulations, device
    )
    replicate(sim.get_clone_plan())
    sim.reset()
    shape = (num_articulations, articulation.num_fixed_tendons)
    limits = torch.tensor([-0.1, 0.2], device=device).expand(*shape, 2)

    articulation.set_fixed_tendon_stiffness_mask(stiffness=torch.full(shape, 12.0, device=device))
    articulation.set_fixed_tendon_damping_index(damping=torch.full(shape, 3.0, device=device))
    articulation.set_fixed_tendon_position_limit_index(limit=limits)
    articulation.write_fixed_tendon_properties_to_sim_mask()
    sim.step()

    solver_model = SimulationManager._solver.mjw_model
    np.testing.assert_allclose(solver_model.tendon_stiffness.numpy(), 12.0)
    np.testing.assert_allclose(solver_model.tendon_damping.numpy(), 3.0)
    np.testing.assert_allclose(solver_model.tendon_range.numpy(), limits.cpu().numpy(), rtol=1e-6)
    torch.testing.assert_close(articulation.data.fixed_tendon_pos_limits.torch, limits)

    articulation.set_fixed_tendon_stiffness_index(stiffness=42.0, env_ids=[1], fixed_tendon_ids=[1])
    articulation.set_fixed_tendon_damping_index(damping=6.0, env_ids=[1], fixed_tendon_ids=[1])
    articulation.write_fixed_tendon_properties_to_sim_index(env_ids=[1], fixed_tendon_ids=[1])
    sim.step()
    expected_stiffness = np.full(shape, 12.0)
    expected_damping = np.full(shape, 3.0)
    expected_stiffness[1, 1] = 42.0
    expected_damping[1, 1] = 6.0
    np.testing.assert_allclose(solver_model.tendon_stiffness.numpy(), expected_stiffness)
    np.testing.assert_allclose(solver_model.tendon_damping.numpy(), expected_damping)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_fragment_fix_root_link_uses_base_manager(sim, device, add_ground_plane, articulation_type):
    """Newton consumes the base manager's world joint without relocating the root API.

    The floating-base ANYmal made fixed-base must then hold its root at the default state.
    """
    articulation_cfg = deepcopy(generate_articulation_cfg(articulation_type=articulation_type))
    articulation_cfg.spawn.articulation_props = []
    articulation_cfg.spawn.fix_root_link = True
    articulation, translations = generate_articulation(
        articulation_cfg, num_articulations=1, device=device, add_ground_plane=True
    )

    root = sim_utils.get_first_matching_child_prim(
        "/World/Env_0/Robot",
        lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
        stage=sim.stage,
    )
    assert root is not None and root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert sim_utils.find_global_fixed_joint_prim("/World/Env_0/Robot", stage=sim.stage) is not None

    replicate(sim.get_clone_plan())
    sim.reset()
    assert articulation.is_initialized
    assert articulation.is_fixed_base
    assert articulation.data.joint_pos.torch.shape == (1, 12)

    for _ in range(10):
        sim.step()
        articulation.update(sim.cfg.dt)

        # check that the root is at the correct state - its default state as it is fixed base
        default_root_pose = articulation.data.default_root_pose.torch.clone()
        default_root_vel = articulation.data.default_root_vel.torch.clone()
        default_root_pose[:, :3] = default_root_pose[:, :3] + translations

        torch.testing.assert_close(articulation.data.root_link_pose_w.torch, default_root_pose)
        torch.testing.assert_close(articulation.data.root_com_vel_w.torch, default_root_vel)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["panda"])
def test_initialization_fixed_base_made_floating_base(
    sim, num_articulations, device, add_ground_plane, articulation_type
):
    """Test initialization for fixed base made floating-base using schema properties.

    This test verifies that:
    1. The articulation is properly initialized
    2. The articulation is floating base after modification
    3. All buffers have correct shapes

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type).copy()
    # Unfix root link by making it non-kinematic
    articulation_cfg.spawn.fix_root_link = False
    articulation, _ = generate_articulation(
        articulation_cfg, num_articulations, device=sim.device, add_ground_plane=True
    )

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    # Play sim
    replicate(sim.get_clone_plan())
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized
    # Check that is floating base
    assert not articulation.is_fixed_base
    # Check buffers that exists and have correct shapes
    assert articulation.data.root_pos_w.torch.shape == (num_articulations, 3)
    assert articulation.data.root_quat_w.torch.shape == (num_articulations, 4)
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 9)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.parametrize("state_field", ["joint_pos", "joint_vel"])
def test_out_of_range_default_joint_state(sim, device, articulation_type, state_field):
    """Initialization fails when the configured default joint position or velocity exceeds the joint limits."""
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type).copy()
    if state_field == "joint_pos":
        articulation_cfg.init_state.joint_pos = {"panda_joint1": 10.0, "panda_joint[2, 4]": -20.0}
    else:
        articulation_cfg.init_state.joint_vel = {"panda_joint1": 100.0, "panda_joint[2, 4]": -60.0}
    articulation, _ = generate_articulation(articulation_cfg, 1, device=device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    quantity = "positions" if state_field == "joint_pos" else "velocities"
    replicate(sim.get_clone_plan())
    with pytest.raises(ValueError, match=f"default {quantity} out of the limits"):
        replicate(sim.get_clone_plan())
        sim.reset()


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_external_force_on_multiple_bodies(sim, num_articulations, device, articulation_type):
    """Test application of external force on the legs of the articulation.

    This test verifies that:
    1. External forces can be applied to multiple bodies
    2. The forces affect the articulation's motion correctly
    3. The articulation responds to the forces as expected

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Play the simulator
    replicate(sim.get_clone_plan())
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies(".*_SHANK")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 1] = 200.0

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
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_external_force_on_multiple_bodies_at_position(sim, num_articulations, device, articulation_type):
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
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Play the simulator
    replicate(sim.get_clone_plan())
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies(".*_SHANK")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 2] = 50.0
    external_wrench_positions_b = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    external_wrench_positions_b[..., 1] = 1.0

    # Now we are ready!
    for i in range(2):
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
@pytest.mark.parametrize("articulation_type", ["humanoid"])
def test_loading_gains_from_usd(sim, num_articulations, device, articulation_type):
    """Test that gains are loaded from USD file if actuator model has them as None.

    This test verifies that:
    1. Gains are loaded correctly from USD file
    2. Default gains are applied when not specified
    3. The gains match the expected values

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type, stiffness=None, damping=None)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)

    # Play sim
    replicate(sim.get_clone_plan())
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
@pytest.mark.parametrize(
    ("joint_limit", "actuator_limit"),
    [(1e5, None), (None, 1e2), (1e5, 1e2)],
    ids=["joint_limit", "actuator_limit", "both_limits"],
)
@pytest.mark.parametrize("articulation_type", ["single_joint_implicit", "single_joint_explicit"])
@pytest.mark.parametrize("use_newton_actuators", [False])  # consumed by the sim fixture
def test_setting_joint_limits_from_cfg(
    sim, articulation_type, num_articulations, device, joint_limit, actuator_limit, use_newton_actuators
):
    """Test the velocity and effort limit resolution for implicit and explicit actuators.

    This test verifies that:
    1. The solver clamps ``joint_velocity_limit`` and ``joint_effort_limit`` are applied to the simulation;
       when unset, the USD-authored values are kept
    2. The actuator limits keep their configured values and are never pushed to the solver
    3. When unset, the actuator velocity limit falls back to the solver clamp, implicit actuators track the
       solver effort clamp, and an explicit actuator's effort limit falls back to the USD-authored value

    Args:
        sim: The simulation fixture
        articulation_type: The implicit or explicit single-joint articulation
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
        joint_limit: The velocity and effort limits to set in simulation
        actuator_limit: The velocity and effort limits to set in the actuator
    """
    articulation_cfg = generate_articulation_cfg(
        articulation_type=articulation_type,
        joint_velocity_limit=joint_limit,
        joint_effort_limit=joint_limit,
        actuator_velocity_limit=actuator_limit,
        actuator_effort_limit=actuator_limit,
    )
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg,
        num_articulations=num_articulations,
        device=device,
    )
    # Play sim
    replicate(sim.get_clone_plan())
    sim.reset()

    # read the values set into the simulation
    model = SimulationManager.get_model()
    newton_vel_limit = wp.to_torch(articulation.root_view.get_attribute("joint_velocity_limit", model)).to(device)[
        :, 0, :
    ]
    newton_effort_limit = wp.to_torch(articulation.root_view.get_attribute("joint_effort_limit", model)).to(device)[
        :, 0, :
    ]
    usd_vel_limit = next(
        p.max_joint_velocity for p in articulation_cfg.spawn.joint_drive_props if isinstance(p, PhysxJointCfg)
    )
    usd_effort_limit = next(
        p.max_force for p in articulation_cfg.spawn.joint_drive_props if isinstance(p, sim_utils.UsdPhysicsDriveCfg)
    )
    actuator = articulation.actuators["joint"]

    # check data buffers
    torch.testing.assert_close(articulation.data.joint_vel_limits.torch, newton_vel_limit)
    torch.testing.assert_close(articulation.data.joint_effort_limits.torch, newton_effort_limit)
    # the solver clamps come from the joint limits when set, otherwise the USD-authored values
    expected_vel_limit = usd_vel_limit if joint_limit is None else joint_limit
    expected_effort_limit = usd_effort_limit if joint_limit is None else joint_limit
    torch.testing.assert_close(newton_vel_limit, torch.full_like(newton_vel_limit, expected_vel_limit))
    torch.testing.assert_close(newton_effort_limit, torch.full_like(newton_effort_limit, expected_effort_limit))

    # the actuator velocity limit keeps its configured value and is not pushed to the solver;
    # when unset it falls back to the solver clamp
    if actuator_limit is not None:
        torch.testing.assert_close(actuator.actuator_velocity_limit, torch.full_like(newton_vel_limit, actuator_limit))
        assert not torch.allclose(actuator.actuator_velocity_limit, newton_vel_limit)
    else:
        torch.testing.assert_close(actuator.actuator_velocity_limit, newton_vel_limit)

    if articulation_type == "single_joint_implicit":
        # without a separately configured rated limit, the implicit actuator limits track the solver clamp
        torch.testing.assert_close(actuator.joint_effort_limit, newton_effort_limit)
        torch.testing.assert_close(actuator.actuator_effort_limit, newton_effort_limit)
    elif actuator_limit is not None:
        torch.testing.assert_close(actuator.actuator_effort_limit, torch.full_like(newton_effort_limit, actuator_limit))
    else:
        # an unset explicit actuator effort limit falls back to the USD-authored value, not the solver clamp
        torch.testing.assert_close(
            actuator.actuator_effort_limit, torch.full_like(newton_effort_limit, usd_effort_limit)
        )


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["humanoid"])
def test_reset(sim, num_articulations, device, articulation_type, monkeypatch):
    """Test that the actuator gains come from the configuration and that reset method works properly."""
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
    )

    # Play the simulator
    replicate(sim.get_clone_plan())
    sim.reset()

    # Check that gains are loaded from the configuration
    expected_stiffness = torch.full(
        (articulation.num_instances, articulation.num_joints), 10.0, device=articulation.device
    )
    expected_damping = torch.full_like(expected_stiffness, 2.0)
    torch.testing.assert_close(articulation.actuators["body"].stiffness, expected_stiffness)
    torch.testing.assert_close(articulation.actuators["body"].damping, expected_damping)

    # Now we are ready!
    # reset articulation
    actuator = next(iter(articulation.actuators.values()))
    actuator_reset = actuator.reset
    reset_env_ids = []

    def record_actuator_reset(env_ids=None):
        reset_env_ids.append(env_ids)
        actuator_reset(env_ids)

    monkeypatch.setattr(actuator, "reset", record_actuator_reset)
    articulation.reset()
    assert reset_env_ids == [None]

    instantaneous_composer = articulation.instantaneous_wrench_composer
    permanent_composer = articulation.permanent_wrench_composer

    # Reset should zero external forces and torques
    assert not instantaneous_composer.active
    assert not permanent_composer.active
    assert torch.count_nonzero(instantaneous_composer.out_force_b.torch) == 0
    assert torch.count_nonzero(instantaneous_composer.out_torque_b.torch) == 0
    assert torch.count_nonzero(permanent_composer.out_force_b.torch) == 0
    assert torch.count_nonzero(permanent_composer.out_torque_b.torch) == 0

    # A partial reset clears only the selected environment's wrenches
    num_bodies = articulation.num_bodies
    permanent_composer.set_forces_and_torques_index(
        forces=torch.ones((num_articulations, num_bodies, 3), device=device),
        torques=torch.ones((num_articulations, num_bodies, 3), device=device),
    )
    instantaneous_composer.add_forces_and_torques_index(
        forces=torch.ones((num_articulations, num_bodies, 3), device=device),
        torques=torch.ones((num_articulations, num_bodies, 3), device=device),
    )
    articulation.reset(env_ids=torch.tensor([0], device=device))
    assert instantaneous_composer.active
    assert permanent_composer.active
    assert torch.count_nonzero(instantaneous_composer.out_force_b.torch) == num_bodies * 3
    assert torch.count_nonzero(instantaneous_composer.out_torque_b.torch) == num_bodies * 3
    assert torch.count_nonzero(permanent_composer.out_force_b.torch) == num_bodies * 3
    assert torch.count_nonzero(permanent_composer.out_torque_b.torch) == num_bodies * 3


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["single_joint_implicit"])
def test_body_root_state(sim, num_articulations, device, articulation_type):
    """Test for reading the `body_state_w` property.

    This test verifies that:
    1. The single-joint articulation initializes as fixed base with correctly shaped buffers
    2. Body link and center-of-mass states match an analytic pendulum with a center-of-mass offset
    3. The fixed root holds its default state

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)
    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10, "Possible reference leak for articulation"
    # Play sim
    replicate(sim.get_clone_plan())
    sim.reset()
    # Check if articulation is initialized
    assert articulation.is_initialized, "Articulation is not initialized"
    # Check that fixed base
    assert articulation.is_fixed_base, "Articulation is not a fixed base"
    # Check buffers that exists and have correct shapes
    assert articulation.data.root_pos_w.torch.shape == (num_articulations, 3)
    assert articulation.data.root_quat_w.torch.shape == (num_articulations, 4)
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 1)
    assert articulation.data.body_mass.torch.shape == (num_articulations, articulation.num_bodies)
    assert articulation.data.body_inertia.torch.shape == (num_articulations, articulation.num_bodies, 9)

    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert getattr(actuator, "is_implicit_model", False) == is_implicit_model_cfg

    # Resolve body indices by name (ordering may differ across physics backends)
    root_idx = articulation.body_names.index("CenterPivot")
    arm_idx = articulation.body_names.index("Arm")

    # change center of mass offset from link frame
    offset = [0.5, 0.0, 0.0]

    # create com offsets — apply offset to the Arm body
    num_bodies = articulation.num_bodies
    com = wp.to_torch(articulation.root_view.get_attribute("body_com", SimulationManager.get_model()))
    link_offset = [1.0, 0.0, 0.0]  # the offset from CenterPivot to Arm frames
    new_com = torch.tensor(offset, device=device).repeat(num_articulations, 1, 1)
    com[:, 0, arm_idx, :] = new_com.squeeze(-2)
    articulation.root_view.set_attribute("body_com", SimulationManager.get_model(), wp.from_torch(com, dtype=wp.vec3f))
    with wp.ScopedDevice(device):
        SimulationManager._solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

    # check they are set
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_attribute("body_com", SimulationManager.get_model())), com
    )

    for i in range(50):
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

        # check that the root is at the correct state - its default state as it is fixed base
        default_root_pose = articulation.data.default_root_pose.torch.clone()
        default_root_vel = articulation.data.default_root_vel.torch.clone()
        default_root_pose[:, :3] = default_root_pose[:, :3] + env_pos
        torch.testing.assert_close(articulation.data.root_link_pose_w.torch, default_root_pose)
        torch.testing.assert_close(articulation.data.root_com_vel_w.torch, default_root_vel)

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


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("state_location", ["com", "link", "root"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_write_root_state_functions_data_consistency(
    sim, num_articulations, device, state_location, gravity_enabled, articulation_type
):
    """A root pose/velocity write must read back in the written frame and refresh the derived cross-frame caches.

    Regression coverage for the velocity invalidation cleanup: writing the root center-of-mass
    (or link) velocity must invalidate the derived root link (or com) velocity so the next read
    re-derives it. Linear velocity differs between the two frames, so - as in the rigid object
    test - we compare angular velocity, which is frame-independent and therefore only matches when
    the derived velocity was actually refreshed.

    The center-of-mass offset is set through ``set_coms_index``, which must update the body-frame CoM and
    the world-frame CoM derived from it without a sim step.
    """
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)

    # Play sim
    replicate(sim.get_clone_plan())
    sim.reset()

    # Resolve root body index by name (ordering may differ across physics backends)
    root_idx = articulation.find_bodies("base")[0][0]

    # change center of mass offset from link frame on the root body
    original_com = articulation.data.body_com_pos_b.torch.clone()
    # Populate the derived world-frame cache so a missing invalidation would surface as a stale read.
    original_com_w = articulation.data.body_com_pos_w.torch.clone()
    new_com = original_com.clone()
    new_com[:, root_idx] = torch.tensor([1.0, 0.0, 0.0], device=device)
    env_ids = torch.arange(num_articulations, device=device, dtype=torch.int32)
    # Full poses are accepted too; Newton uses the position and ignores the orientation.
    com_poses = articulation.data.body_com_pose_b.torch.clone()
    com_poses[..., :3] = new_com
    articulation.set_coms_index(coms=com_poses, env_ids=env_ids)
    torch.testing.assert_close(articulation.data.body_com_pos_b.torch, new_com, atol=1e-5, rtol=1e-5)
    articulation.set_coms_index(coms=original_com, env_ids=env_ids)
    torch.testing.assert_close(articulation.data.body_com_pos_b.torch, original_com, atol=1e-5, rtol=1e-5)
    _ = articulation.data.body_com_pos_w.torch
    articulation.set_coms_index(coms=new_com, env_ids=env_ids)

    torch.testing.assert_close(articulation.data.body_com_pos_b.torch, new_com, atol=1e-5, rtol=1e-5)
    # Without a sim step the links stay put, so the world-frame CoM is the link pose applied to the new offset.
    link_pos_w = articulation.data.body_link_pos_w.torch
    link_quat_w = articulation.data.body_link_quat_w.torch
    expected_com_w = link_pos_w + math_utils.quat_apply(link_quat_w, new_com)
    updated_com_w = articulation.data.body_com_pos_w.torch
    torch.testing.assert_close(updated_com_w, expected_com_w, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(updated_com_w, original_com_w)

    def random_root_state(num_envs: int) -> torch.Tensor:
        state = torch.rand(num_envs, 13, device=device)
        state[..., :3] += env_pos[:num_envs]
        # make quaternion a unit vector
        state[..., 3:7] = torch.nn.functional.normalize(state[..., 3:7], dim=-1)
        return state

    def write_root_state(state: torch.Tensor, env_ids: torch.Tensor | None = None):
        if state_location == "com":
            articulation.write_root_com_pose_to_sim_index(root_pose=state[..., :7], env_ids=env_ids)
            articulation.write_root_com_velocity_to_sim_index(root_velocity=state[..., 7:], env_ids=env_ids)
        elif state_location == "link":
            articulation.write_root_link_pose_to_sim_index(root_pose=state[..., :7], env_ids=env_ids)
            articulation.write_root_link_velocity_to_sim_index(root_velocity=state[..., 7:], env_ids=env_ids)
        elif state_location == "root":
            articulation.write_root_pose_to_sim_index(root_pose=state[..., :7], env_ids=env_ids)
            articulation.write_root_velocity_to_sim_index(root_velocity=state[..., 7:], env_ids=env_ids)

    def read_written_root_state() -> torch.Tensor:
        # the root pose is the link pose and the root velocity is the center-of-mass velocity
        pose_w = articulation.data.root_com_pose_w if state_location == "com" else articulation.data.root_link_pose_w
        vel_w = articulation.data.root_link_vel_w if state_location == "link" else articulation.data.root_com_vel_w
        return torch.cat((pose_w.torch, vel_w.torch), dim=-1)

    rand_state = random_root_state(num_articulations)

    # perform a step then update the buffers
    sim.step()
    articulation.update(sim.cfg.dt)

    # Prime the lazily-derived caches at the current sim timestamp. Without this they would
    # recompute on first access after the write regardless of invalidation; priming them makes a
    # missing reset_pose/reset_velocity observable as a stale read in the assertions below.
    _ = articulation.data.root_link_pose_w.torch
    _ = articulation.data.root_com_pose_w.torch
    _ = articulation.data.root_link_vel_w.torch
    _ = articulation.data.root_com_vel_w.torch

    write_root_state(rand_state)

    body_com_pose_b = articulation.data.body_com_pose_b.torch
    if state_location == "com":
        # the com pose/vel was written, so the derived link pose/vel must be refreshed
        root_com_pose_w = articulation.data.root_com_pose_w.torch
        root_com_vel_w = articulation.data.root_com_vel_w.torch
        expected_root_link_pos, expected_root_link_quat = math_utils.combine_frame_transforms(
            root_com_pose_w[:, :3],
            root_com_pose_w[:, 3:],
            math_utils.quat_rotate(
                math_utils.quat_inv(body_com_pose_b[:, root_idx, 3:7]), -body_com_pose_b[:, root_idx, :3]
            ),
            math_utils.quat_inv(body_com_pose_b[:, root_idx, 3:7]),
        )
        expected_root_link_pose = torch.cat((expected_root_link_pos, expected_root_link_quat), dim=1)
        root_link_pose_w = articulation.data.root_link_pose_w.torch
        root_link_vel_w = articulation.data.root_link_vel_w.torch
        torch.testing.assert_close(expected_root_link_pose, root_link_pose_w)
        # skip lin_vel because it differs from the link frame; angular velocity is frame-independent
        # and only matches when the derived velocity was actually refreshed after the write
        torch.testing.assert_close(root_com_vel_w[:, 3:], root_link_vel_w[:, 3:])
    else:
        # the link pose/vel was written, so the derived com pose/vel must be refreshed
        root_link_pose_w = articulation.data.root_link_pose_w.torch
        root_link_vel_w = articulation.data.root_link_vel_w.torch
        expected_com_pos, expected_com_quat = math_utils.combine_frame_transforms(
            root_link_pose_w[:, :3],
            root_link_pose_w[:, 3:],
            body_com_pose_b[:, root_idx, :3],
            body_com_pose_b[:, root_idx, 3:7],
        )
        expected_com_pose = torch.cat((expected_com_pos, expected_com_quat), dim=1)
        root_com_pose_w = articulation.data.root_com_pose_w.torch
        root_com_vel_w = articulation.data.root_com_vel_w.torch
        torch.testing.assert_close(expected_com_pose, root_com_pose_w)
        torch.testing.assert_close(root_link_vel_w[:, 3:], root_com_vel_w[:, 3:])

    # the written state reads back in the written frame
    torch.testing.assert_close(read_written_root_state(), rand_state)

    # a partial write only changes the selected environment
    partial_state = random_root_state(1)
    write_root_state(partial_state, env_ids=torch.tensor([0], device=device, dtype=torch.int32))
    expected_state = rand_state.clone()
    expected_state[0] = partial_state[0]
    torch.testing.assert_close(read_written_root_state(), expected_state)


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("articulation_type", ["humanoid"])
@pytest.mark.parametrize("root_prim_path", ["/torso", "/non_existing_prim_path"])
def test_setting_articulation_root_prim_path(sim, device, articulation_type, root_prim_path):
    """An explicit articulation root prim path initializes when it exists and fails clearly otherwise."""
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation_cfg.articulation_root_prim_path = root_prim_path
    articulation, _ = generate_articulation(articulation_cfg, 1, device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    if root_prim_path == "/torso":
        replicate(sim.get_clone_plan())
        sim.reset()
        assert articulation.is_initialized
    else:
        replicate(sim.get_clone_plan())
        with pytest.raises(KeyError, match="No articulations matching pattern"):
            sim.reset()


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_write_joint_state_data_consistency(sim, num_articulations, device, gravity_enabled, articulation_type):
    """Joint limit and joint state writes update the joint buffers and refresh the body state without a step.

    This test verifies that:
    1. Joint position limits are written and keep in-limit default joint positions
    2. A partial joint state write with unsorted int64 selectors updates only the selected entries
    3. A joint state write moves the bodies and refreshes the derived body poses and velocities
    4. Indexed joint limits that exclude a default joint position clamp it into the new limits

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
    """
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)
    env_idx = torch.tensor([x for x in range(num_articulations)])

    # Play sim
    replicate(sim.get_clone_plan())
    sim.reset()

    # Get current default joint pos
    default_joint_pos = articulation.data.default_joint_pos.torch.clone()

    limits = torch.zeros(num_articulations, articulation.num_joints, 2, device=device)
    limits[..., 0] = (torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0) * -1.0
    limits[..., 1] = torch.rand(num_articulations, articulation.num_joints, device=device) + 5.0
    articulation.write_joint_position_limit_to_sim_index(limits=limits)

    # Check new limits are in place and the in-limit defaults are preserved
    torch.testing.assert_close(articulation.data.joint_pos_limits.torch, limits)
    torch.testing.assert_close(articulation.data.default_joint_pos.torch, default_joint_pos)

    # Write joint state with unsorted int64 selectors
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
    # skip lin_vel because it differs from link frame, this should be fine because we are only checking
    # if velocity update is triggered, which can be determined by comparing angular velocity
    body_link_vel_w = articulation.data.body_link_vel_w.torch
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
    torch.testing.assert_close(expected_com_pos.view(len(env_idx), -1, 3), articulation.data.body_com_pos_w.torch)
    torch.testing.assert_close(expected_com_quat.view(len(env_idx), -1, 4), articulation.data.body_com_quat_w.torch)

    # Set new joint limits with indexing that invalidate the selected default joint positions
    env_ids = torch.arange(1, device=device, dtype=torch.int32)
    joint_ids = torch.nonzero(default_joint_pos[0].abs() > 0.1).squeeze(-1)[:2].to(torch.int32)
    assert len(joint_ids) == 2
    limits = torch.zeros(env_ids.shape[0], joint_ids.shape[0], 2, device=device)
    limits[..., 0] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) * -0.1
    limits[..., 1] = torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device) * 0.1
    articulation.write_joint_position_limit_to_sim_index(limits=limits, env_ids=env_ids, joint_ids=joint_ids)

    # Check new limits are in place and the defaults are clamped into them
    torch.testing.assert_close(articulation.data.joint_pos_limits.torch[env_ids][:, joint_ids], limits)
    default_joint_pos_torch = articulation.data.default_joint_pos.torch
    within_bounds = (default_joint_pos_torch[env_ids][:, joint_ids] >= limits[..., 0]) & (
        default_joint_pos_torch[env_ids][:, joint_ids] <= limits[..., 1]
    )
    assert torch.all(within_bounds)


@pytest.mark.parametrize("selector_kind", ["index", "mask"])
@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.parametrize("device", test_devices())
def test_write_joint_viscous_friction_to_sim(sim, num_articulations, device, articulation_type, selector_kind):
    """Test passive viscous joint damping is distinct from actuator derivative gains.

    Static joint friction writes also propagate directly to the Newton model.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type)
    articulation_cfg.actuators["panda_shoulder"].viscous_friction = 0.25
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device)
    replicate(sim.get_clone_plan())
    sim.reset()

    shoulder_joint_ids = articulation.actuators["panda_shoulder"].joint_indices
    expected_viscous_friction = torch.full((articulation.num_instances, 4), 0.25, device=device)
    torch.testing.assert_close(
        articulation.data.joint_viscous_friction_coeff.torch[:, shoulder_joint_ids], expected_viscous_friction
    )
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_attribute("joint_damping", SimulationManager.get_model()))[
            :, 0, shoulder_joint_ids
        ],
        expected_viscous_friction,
    )

    expected_pd_damping = torch.full_like(expected_viscous_friction, 4.0)
    torch.testing.assert_close(articulation.data.joint_damping.torch[:, shoulder_joint_ids], expected_pd_damping)
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_attribute("joint_target_kd", SimulationManager.get_model()))[
            :, 0, shoulder_joint_ids
        ],
        expected_pd_damping,
    )

    values = torch.full((articulation.num_instances, articulation.num_joints), 0.25, device=device)
    if selector_kind == "index":
        articulation.write_joint_viscous_friction_coefficient_to_sim_index(
            joint_viscous_friction_coeff=values,
        )
    else:
        articulation.write_joint_viscous_friction_coefficient_to_sim_mask(
            joint_viscous_friction_coeff=values,
        )

    torch.testing.assert_close(articulation.data.joint_viscous_friction_coeff.torch, values)
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_attribute("joint_damping", SimulationManager.get_model()))[:, 0],
        values,
    )

    # Distinct per-env rows catch writers that ignore the env index
    friction = torch.rand(articulation.num_instances, articulation.num_joints, device=device)
    assert not torch.allclose(friction[0], friction[1])
    articulation.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=friction)
    joint_friction_coeff_sim = wp.to_torch(
        articulation.root_view.get_attribute("joint_friction", SimulationManager.get_model())
    )[:, 0, :]
    torch.testing.assert_close(joint_friction_coeff_sim, friction)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_body_q_consistent_after_root_write(num_articulations, device, articulation_type):
    """Test that body_q is fresh when collide() runs after a root pose write.

    Regression test for a NaN bug where collide() used stale body_q after env
    reset because eval_fk was not called between write_root_pose and collide.

    Uses ``use_mujoco_contacts=False`` so the Newton collision pipeline is
    active, then patches ``_simulate_physics_only`` to capture body_q at
    the moment collide() is called and asserts it matches joint_q.
    """
    from unittest.mock import patch

    sim_cfg = SimulationCfg(
        dt=1 / 200,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=70,
                nconmax=70,
                integrator="implicitfast",
                use_mujoco_contacts=False,
            ),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )
    with build_simulation_context(sim_cfg=sim_cfg, device=device) as sim:
        sim._app_control_on_stop_handle = None
        articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
        articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)

        replicate(sim.get_clone_plan())
        sim.reset()

        model = SimulationManager.get_model()
        jc_starts = model.joint_coord_world_start.numpy()
        body_starts = model.body_world_start.numpy()

        for _ in range(5):
            sim.step()
            articulation.update(sim.cfg.dt)

        # Teleport env 0 by 10m (simulating a reset)
        new_pose = articulation.data.default_root_pose.torch.clone()
        new_pose[0, 0] += 10.0
        new_pose[0, 1] += 5.0
        articulation.write_root_pose_to_sim_index(
            root_pose=new_pose[0:1],
            env_ids=torch.tensor([0], device=device, dtype=torch.int32),
        )

        # Patch _simulate_physics_only to capture body_q before collide runs
        captured = {}
        original_simulate = SimulationManager._simulate_physics_only.__func__

        @classmethod  # type: ignore[misc]
        def _patched_simulate(cls):
            if cls._needs_collision_pipeline:
                bq = wp.to_torch(cls.backend.state_0.body_q)
                jq = wp.to_torch(cls.backend.state_0.joint_q)
                b0 = int(body_starts[0])
                jc0 = int(jc_starts[0])
                captured["bq_root"] = bq[b0, :3].clone()
                captured["jq_root"] = jq[jc0 : jc0 + 3].clone()
            original_simulate(cls)

        with patch.object(SimulationManager, "_simulate_physics_only", _patched_simulate):
            sim.step()
        articulation.update(sim.cfg.dt)

        assert captured, "collision pipeline did not run — _needs_collision_pipeline is False"

        bq_root = captured["bq_root"]
        jq_root = captured["jq_root"]
        diff = (jq_root - bq_root).abs().max().item()
        assert diff < 0.01, (
            f"body_q was stale when collide() ran: diff={diff:.4f}m, jq={jq_root.tolist()}, bq={bq_root.tolist()}"
        )


@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.parametrize("body_subset", [False, True])
def test_set_material_properties(sim, num_articulations, device, add_ground_plane, articulation_type, body_subset):
    """Material and collider-offset randomization write through to the robot's shapes in the Newton model.

    The event terms write the asset's view-level shape bindings; the assertions read the flat Newton model
    arrays at the collision shapes of the selected bodies and environments. Visual shapes are ignored because
    their materials and offsets have no physical effect.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device, add_ground_plane=True
    )

    # Play the simulator
    replicate(sim.get_clone_plan())
    sim.reset()

    # Resolve the robot's shapes per environment and body from the flat Newton model.
    model = SimulationManager.get_model()
    body_world = model.body_world.numpy()
    body_names = [label.rsplit("/", 1)[-1] for label in model.body_label]
    shape_body = model.shape_body.numpy()
    is_collision_shape = (model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES)) != 0

    def robot_shapes(env_index: int, selected_body_names: list[str] | None = None) -> np.ndarray:
        bodies = [
            body
            for body in np.flatnonzero(body_world == env_index)
            if selected_body_names is None or body_names[body] in selected_body_names
        ]
        return np.flatnonzero(np.isin(shape_body, bodies) & is_collision_shape)

    env = SimpleNamespace(scene={"robot": articulation}, sim=sim, device=device, num_envs=num_articulations)
    env_ids = torch.tensor([num_articulations - 1], device=device)

    # Randomize the materials in the last environment, with degenerate ranges.
    asset_cfg = SceneEntityCfg("robot")
    selected_body_names = None
    if body_subset:
        asset_cfg.body_ids, selected_body_names = articulation.find_bodies(["panda_link3", "panda_hand"])
    selected_shapes = robot_shapes(num_articulations - 1, selected_body_names)
    unselected_shapes = np.setdiff1d(robot_shapes(num_articulations - 1), selected_shapes)
    other_env_shapes = robot_shapes(0)
    assert len(selected_shapes) > 0 and len(other_env_shapes) > 0
    original_mu = model.shape_material_mu.numpy().copy()
    original_restitution = model.shape_material_restitution.numpy().copy()
    params = {
        "static_friction_range": (0.55, 0.55),
        "dynamic_friction_range": (0.55, 0.55),
        "restitution_range": (0.15, 0.15),
        "num_buckets": 1,
        "asset_cfg": asset_cfg,
    }
    material_term = randomize_rigid_body_material(
        EventTermCfg(func=randomize_rigid_body_material, mode="startup", params=params), env
    )
    material_term(env, env_ids, **params)

    # Randomize the collider offsets of every shape in the last environment.
    original_margin = model.shape_margin.numpy().copy()
    original_gap = model.shape_gap.numpy().copy()
    offset_params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "rest_offset_distribution_params": (0.01, 0.01),
        "contact_offset_distribution_params": (0.03, 0.03),
    }
    offset_term = randomize_rigid_body_collider_offsets(
        EventTermCfg(func=randomize_rigid_body_collider_offsets, mode="startup", params=offset_params), env
    )
    offset_term(env, env_ids, **offset_params)

    # Simulate physics
    sim.step()
    articulation.update(sim.cfg.dt)

    mu = model.shape_material_mu.numpy()
    restitution = model.shape_material_restitution.numpy()
    np.testing.assert_allclose(mu[selected_shapes], 0.55)
    np.testing.assert_allclose(restitution[selected_shapes], 0.15)
    for shapes in (unselected_shapes, other_env_shapes):
        np.testing.assert_array_equal(mu[shapes], original_mu[shapes])
        np.testing.assert_array_equal(restitution[shapes], original_restitution[shapes])

    # Newton maps the rest offset to the shape margin and the contact offset to margin + gap.
    randomized_shapes = robot_shapes(num_articulations - 1)
    np.testing.assert_allclose(model.shape_margin.numpy()[randomized_shapes], 0.01)
    np.testing.assert_allclose(model.shape_gap.numpy()[randomized_shapes], 0.02, atol=1e-6)
    np.testing.assert_array_equal(model.shape_margin.numpy()[other_env_shapes], original_margin[other_env_shapes])
    np.testing.assert_array_equal(model.shape_gap.numpy()[other_env_shapes], original_gap[other_env_shapes])


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize(
    "first_property",
    ["body_com_jacobian_w", "body_link_jacobian_w", "mass_matrix", "gravity_compensation_forces"],
)
def test_task_space_allocation_and_capture(monkeypatch, device, first_property):
    """Allocate only requested outputs and retain them across changed-state graph replay."""
    builder = ModelBuilder()
    builder.begin_world()
    base = builder.add_link(mass=4.0, inertia=wp.mat33(np.eye(3)), label="Robot/base")
    arm = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3)), com=(0.5, 0.0, 0.0), label="Robot/arm")
    slider = builder.add_link(mass=3.0, inertia=wp.mat33(np.eye(3)), com=(0.25, 0.0, 0.0), label="Robot/slider")
    builder.add_articulation(
        [
            builder.add_joint_fixed(-1, base),
            builder.add_joint_revolute(base, arm, axis=(0.0, 1.0, 0.0), label="hinge"),
            builder.add_joint_prismatic(arm, slider, axis=(1.0, 0.0, 0.0), label="slide"),
        ],
        label="Robot",
    )
    builder.end_world()
    model = builder.finalize(device=device)
    state, control = model.state(), model.control()
    monkeypatch.setattr(SimulationManager, "get_model", lambda: model)
    monkeypatch.setattr(SimulationManager, "get_state_0", lambda: state)
    monkeypatch.setattr(SimulationManager, "get_control", lambda: control)
    view = ArticulationView(model, "Robot", exclude_joint_types=[JointType.FIXED])
    eager = ArticulationData(view, device)
    eager._apply_ordering_maps_after_resolve()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    getattr(eager, first_property)  # Compile kernels before first-use capture on a fresh container.
    data = ArticulationData(view, device)
    data._apply_ordering_maps_after_resolve()
    properties = ("body_com_jacobian_w", "body_link_jacobian_w", "mass_matrix", "gravity_compensation_forces")
    assert all(getattr(data, f"_{name}_ta") is None for name in properties)
    assert data._jacobian_buf_flat is data._mass_matrix_full_buf is data._gravity_force_full_buf is None
    if wp.get_device(device).is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            output = getattr(data, first_property)
    else:
        output = getattr(data, first_property)
    required = {first_property}
    if first_property == "body_link_jacobian_w":
        required.add("body_com_jacobian_w")
    for name in properties:
        assert (getattr(data, f"_{name}_ta") is not None) == (name in required)
    assert (data._jacobian_buf_flat is not None) == (first_property != "gravity_compensation_forces")
    assert (data._mass_matrix_full_buf is not None) == (first_property == "mass_matrix")
    assert (data._gravity_force_full_buf is not None) == (first_property == "gravity_compensation_forces")

    for angle, displacement in ((0.0, 0.0), (0.7, 0.4), (-0.3, 0.2)):
        state.joint_q.assign(np.asarray([angle, displacement], dtype=np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        expected = getattr(eager, first_property).warp.numpy()
        if wp.get_device(device).is_cuda:
            wp.capture_launch(capture.graph)
        else:
            assert getattr(data, first_property) is output
        np.testing.assert_allclose(output.warp.numpy(), expected, atol=1e-5)
    data._create_simulation_bindings()
    data._apply_ordering_maps_after_resolve()
    assert getattr(data, first_property) is output


##
# Shape-contract regression tests for the new BaseArticulation accessors.
# These pin the public shape contract so future regressions (e.g., reverting
# to model-wide max sizing or to the wrong fixed-base row offset) fail fast.
##


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["anymal"])
@pytest.mark.isaacsim_ci
def test_heterogeneous_scene_per_view_shapes(sim, device, add_ground_plane, articulation_type):
    """Mixed-articulation scene: each view returns ITS OWN asset's shape.

    Direct regression test for the Codex round-2 finding. With Franka
    (9 DoFs) and Anymal-C (18 DoFs) co-resident in the model,
    ``model.max_dofs_per_articulation == 18`` and
    ``model.max_joints_per_articulation == anymal.num_bodies``. The Franka
    view's ``body_link_jacobian_w`` / ``mass_matrix`` outputs must use
    Franka's per-asset counts, NOT the model-wide maxima — otherwise
    Franka's mass matrix would carry zero-padded rows/cols and be
    singular.

    Uses the ``anymal`` ``SIM_CFGs`` entry (more capable solver settings)
    for the host sim; the ``articulation_type`` parametrize is only there
    so the ``sim`` fixture picks a config — the test itself constructs
    both Anymal and Franka articulations directly.
    """
    # ``num_per_type=1`` keeps the actuator-default replication path off —
    # Newton's USD default loader hits a (1, num_joints) vs (num_envs,
    # num_joints) shape mismatch with multi-instance multi-type scenes; one
    # of each is the minimum heterogeneous setup that still exercises the
    # per-articulation shape gate without that pre-existing quirk.
    num_per_type = 1

    franka_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Env_[^/]*/Franka")
    anymal_cfg = ANYMAL_C_CFG.replace(prim_path="/World/Env_[^/]*/Anymal")
    anymal_cfg.init_state.pos = (0.0, 5.0, anymal_cfg.init_state.pos[2])

    sim_utils.create_prim("/World/Env_0", "Xform")
    clone_plan_from_env_0(
        CloneCfg(clone_template="/World/Env_{}"),
        (franka_cfg, anymal_cfg, AssetBaseCfg(prim_path="/World/defaultGroundPlane")),
        num_per_type,
        2.5,
    )

    franka = Articulation(franka_cfg)
    anymal = Articulation(anymal_cfg)
    replicate(sim.get_clone_plan())
    sim.reset()
    assert franka.is_initialized and anymal.is_initialized
    assert franka.is_fixed_base and not anymal.is_fixed_base

    # Sanity: the model-wide maxima are larger than at least one view's
    # per-asset count, so a regression to model-wide sizing would manifest
    # as wrong shapes here. Assert that precondition explicitly so the test
    # fails clearly if the fixture stops being heterogeneous.
    model = SimulationManager.get_model()
    assert model.max_dofs_per_articulation > min(franka.num_joints, anymal.num_joints), (
        "scene is no longer heterogeneous; this test relies on model.max_dofs > one view's num_joints"
    )

    franka_J = franka.data.body_link_jacobian_w.torch
    anymal_J = anymal.data.body_link_jacobian_w.torch

    # Each view's output uses its OWN per-asset count, not the model-wide max.
    # Floating-base assets prepend ``num_base_dofs`` floating-base columns; fixed-base
    # assets have ``num_base_dofs == 0``.
    franka_dofs = franka.num_joints + franka.num_base_dofs
    anymal_dofs = anymal.num_joints + anymal.num_base_dofs
    assert franka_J.shape == torch.Size((num_per_type, franka.num_bodies - 1, 6, franka_dofs)), (
        f"Franka jacobian leaked model-wide shape: got {tuple(franka_J.shape)}"
    )
    assert anymal_J.shape == torch.Size((num_per_type, anymal.num_bodies, 6, anymal_dofs)), (
        f"Anymal jacobian leaked model-wide shape: got {tuple(anymal_J.shape)}"
    )

    sim.step()
    franka.update(sim.cfg.dt)
    anymal.update(sim.cfg.dt)

    franka_M = franka.data.mass_matrix.torch
    anymal_M = anymal.data.mass_matrix.torch

    assert franka_M.shape == torch.Size((num_per_type, franka_dofs, franka_dofs))
    assert anymal_M.shape == torch.Size((num_per_type, anymal_dofs, anymal_dofs))

    # Each view's mass matrix must have positive diagonals — padded zero
    # rows/cols (the round-2 bug) would surface as zero diagonals on the
    # smaller-DoF view. Using a per-diagonal check here instead of det()
    # because det of a real Franka mass matrix is naturally ~1e-13.
    assert (franka_M.diagonal(dim1=-2, dim2=-1) > 1e-6).all(), (
        "Franka mass matrix has non-positive diagonal under heterogeneous scene"
    )
    assert (anymal_M.diagonal(dim1=-2, dim2=-1) > 1e-6).all(), (
        "Anymal mass matrix has non-positive diagonal under heterogeneous scene"
    )

    # Gravity compensation gathers a FLAT model-wide DoF buffer, so a padded-layout
    # regression (indexing by ``art_id * max_dofs`` instead of
    # ``joint_qd_start[articulation_start[art_id]]``) is numerically invisible in
    # homogeneous scenes — this mixed scene is the only place it can surface.
    franka_g = franka.data.gravity_compensation_forces.torch
    anymal_g = anymal.data.gravity_compensation_forces.torch
    assert franka_g.shape == torch.Size((num_per_type, franka_dofs))
    assert anymal_g.shape == torch.Size((num_per_type, anymal_dofs))
    assert franka_g.abs().max() > 1e-3, "Franka gravity compensation is all-zero under heterogeneous scene"
    assert anymal_g.abs().max() > 1e-3, "Anymal gravity compensation is all-zero under heterogeneous scene"


@pytest.mark.parametrize("num_articulations", [4])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["panda", "anymal"])
@pytest.mark.parametrize("ordering_mode", ["none", "reversed"])
@pytest.mark.isaacsim_ci
def test_get_gravity_compensation_forces_matches_jacobian_gravity(
    sim, num_articulations, device, add_ground_plane, articulation_type, ordering_mode
):
    """``g(q)`` must equal ``-sum_b J_com_b^T (m_b * g_w)`` at the current configuration.

    Newton computes the gravity compensation force through an RNEA pass
    (``eval_inverse_dynamics_passive``); the static identity above derives the same
    quantity independently from the COM-referenced Jacobian and the per-body masses,
    pinning the sign convention, the DoF ordering (including the 6 floating-base
    entries), and the flat-buffer view gather in one assertion. Non-default joint
    positions and — for floating-base — a rotated, lifted root pose guard the corner
    fixed upstream in newton#2625 (wrong gravity compensation under non-identity
    root pose).

    With ``ordering_mode="reversed"`` a nonidentity joint ordering is active and
    both sides of the identity must be expressed in user joint order: the
    Jacobian gather applies the user->backend permutation, so a
    ``gather_dof_force_rows`` that skips it returns backend-ordered forces and
    breaks the identity row-wise.

    The same fixture also pins:

    * the per-articulation shapes of the Jacobian, mass matrix and gravity compensation
      accessors, and a symmetric, positive-definite mass matrix. Fixed-base (panda):
      ``body_link_jacobian_w`` drops the fixed-root row, so its shape is
      ``(N, num_bodies - 1, 6, num_joints)``. Floating-base (anymal): every body row is kept and
      ``num_base_dofs`` floating-base columns/entries are prepended on the DoF axis, matching the
      cross-library convention (Pinocchio, Drake, MuJoCo, RBDL, OCS2, iDynTree). A zero-padded
      model-wide sizing would surface as non-positive mass-matrix diagonals;
    * that every dynamics accessor reflects a manual joint write without a sim step (the FK
      trigger before ``eval_jacobian``, ``eval_mass_matrix`` and the RNEA pass);
    * the link-origin Jacobian contract: ``J · q_dot`` must encode the link-origin twist
      ``v_origin = v_com - omega x (R · body_com_pos_b)``, which the IsaacLab task-space
      controllers (IK / OSC / RMPFlow) rely on. Newton's ``eval_jacobian`` natively produces
      COM-referenced rows, so the ground truth reads Newton's per-body twist directly from the
      ArticulationView state.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    if ordering_mode == "reversed":
        joint_names = PANDA_JOINT_NAMES if articulation_type == "panda" else ANYMAL_C_PHYSX_JOINT_NAMES
        articulation_cfg = articulation_cfg.replace(joint_ordering=tuple(reversed(joint_names)))
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device, add_ground_plane=True)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    replicate(sim.get_clone_plan())
    sim.reset()
    assert articulation.is_initialized
    assert articulation.is_fixed_base == (articulation_type == "panda")
    # Check buffers that exists and have correct shapes
    assert articulation.data.root_pos_w.torch.shape == (num_articulations, 3)
    assert articulation.data.root_quat_w.torch.shape == (num_articulations, 4)
    assert articulation.data.joint_pos.torch.shape == (num_articulations, 9 if articulation_type == "panda" else 12)
    assert articulation.data.body_mass.torch.shape == (num_articulations, articulation.num_bodies)
    assert articulation.data.body_inertia.torch.shape == (num_articulations, articulation.num_bodies, 9)

    # -- actuator type
    for actuator_name, actuator in articulation.actuators.items():
        is_implicit_model_cfg = isinstance(articulation_cfg.actuators[actuator_name], ImplicitActuatorCfg)
        assert getattr(actuator, "is_implicit_model", False) == is_implicit_model_cfg

    num_dofs = articulation.num_joints + articulation.num_base_dofs
    num_jacobian_bodies = articulation.num_bodies - 1 if articulation.is_fixed_base else articulation.num_bodies

    J = articulation.data.body_link_jacobian_w.torch
    assert J.shape == torch.Size((num_articulations, num_jacobian_bodies, 6, num_dofs)), tuple(J.shape)
    assert J.dtype == torch.float32

    g = articulation.data.gravity_compensation_forces.torch
    assert g.shape == torch.Size((num_articulations, num_dofs)), tuple(g.shape)
    assert g.dtype == torch.float32

    sim.step()
    articulation.update(sim.cfg.dt)

    M = articulation.data.mass_matrix.torch
    assert M.shape == torch.Size((num_articulations, num_dofs, num_dofs)), tuple(M.shape)
    assert M.dtype == torch.float32
    diag = M.diagonal(dim1=-2, dim2=-1)
    assert (diag > 1e-6).all(), f"mass matrix has non-positive diagonal entries: min={diag.min()}"

    # The joint-space inertia is symmetric by construction; asymmetry means a wrong-axis gather or a
    # half-populated buffer. OSC inverts ``J M^-1 J^T`` every step, so ``M`` must also be positive-definite.
    asym = (M - M.transpose(-1, -2)).abs().max().item()
    assert asym < 1e-4, f"|M - M^T|_max = {asym:.3e} — mass matrix is not symmetric"
    # A tiny jitter tolerates the float32 eigenvalue floor without masking real non-PD bugs.
    eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype).expand_as(M)
    torch.linalg.cholesky(M + 1e-6 * eye)

    # Read every accessor at the stepped joint state.
    J_link_0 = articulation.data.body_link_jacobian_w.torch.clone()
    J_com_0 = articulation.data.body_com_jacobian_w.torch.clone()
    M_0 = articulation.data.mass_matrix.torch.clone()
    g_0 = articulation.data.gravity_compensation_forces.torch.clone()

    # Non-trivial configuration via manual writes (no sim step, so the assert
    # compares both quantities at exactly this state): random joint offsets,
    # and for floating-base a non-identity root pose.
    torch.manual_seed(0)
    q = articulation.data.default_joint_pos.torch + 0.3 * torch.randn(
        num_articulations, articulation.num_joints, device=device
    )
    articulation.write_joint_position_to_sim_index(position=q)
    if not articulation.is_fixed_base:
        root_pose = articulation.data.default_root_pose.torch.clone()
        root_pose[:, 2] += 1.0
        # (x, y, z, w) quaternion — 30 deg roll about x.
        root_pose[:, 3:] = torch.tensor([0.2588, 0.0, 0.0, 0.9659], device=device)
        articulation.write_root_pose_to_sim_index(root_pose=root_pose)
        # Guard against a vacuous identity: if root-pose FK invalidation ever
        # regressed, both sides would be evaluated at the stale identity pose and
        # agree trivially, voiding the newton#2625 rotated-root coverage.
        torch.testing.assert_close(articulation.data.root_link_pose_w.torch, root_pose, atol=1e-5, rtol=0.0)

    # With the FK trigger, forward() refreshes body_q to the written state before each accessor evaluates.
    # Without it, body_q stays at the previous state and every accessor returns its previous value.
    assert not torch.allclose(J_link_0, articulation.data.body_link_jacobian_w.torch, atol=1e-3), (
        "body_link_jacobian_w did not change after manual joint write; FK trigger likely missing"
    )
    assert not torch.allclose(J_com_0, articulation.data.body_com_jacobian_w.torch, atol=1e-3), (
        "body_com_jacobian_w did not change after manual joint write; FK trigger likely missing"
    )
    assert not torch.allclose(M_0, articulation.data.mass_matrix.torch, atol=1e-3), (
        "mass_matrix did not change after manual joint write; FK trigger likely missing"
    )
    assert not torch.allclose(g_0, articulation.data.gravity_compensation_forces.torch, atol=1e-3), (
        "gravity_compensation_forces did not change after manual joint write; FK trigger likely missing"
    )

    g_meas = articulation.data.gravity_compensation_forces.torch

    # Independent derivation: generalized gravity load tau_g = sum_b J_lin_b^T (m_b g_w);
    # the compensation force is its negation. The COM-referenced Jacobian is exactly the
    # right lever arm for a point gravity force acting at each body's COM.
    J_com = articulation.data.body_com_jacobian_w.torch  # (N, B_jac, 6, D)
    masses = articulation.data.body_mass.torch  # (N, num_bodies)
    if articulation.is_fixed_base:
        # jacobi_body_idx == body_idx - 1 for fixed-base (fixed-root row excluded).
        masses = masses[:, 1:]
    model = SimulationManager.get_model()
    gravity_w = wp.to_torch(model.gravity[: model.world_count])  # (num_worlds, 3)
    assert gravity_w.shape[0] == num_articulations, "fixture must place one articulation per world"
    f_gravity = masses.unsqueeze(-1) * gravity_w.unsqueeze(1)  # (N, B_jac, 3)
    g_expected = -torch.einsum("nbij,nbi->nj", J_com[:, :, 0:3, :], f_gravity)

    torch.testing.assert_close(g_meas, g_expected, atol=1e-2, rtol=1e-3)

    # Link-origin Jacobian contract. Reproducible non-trivial q_dot — large enough to drive omega
    # well above the floor where COM offset effects would round into noise. The root is at rest so
    # the actuated-only J slice predicts the whole body twist.
    qdot = torch.randn(num_articulations, articulation.num_joints, device=device) * 0.5
    articulation.write_joint_velocity_to_sim_index(velocity=qdot)
    if not articulation.is_fixed_base:
        articulation.write_root_velocity_to_sim_index(root_velocity=torch.zeros(num_articulations, 6, device=device))
    # Refresh kinematics without integrating: for floating bases, an actuator step can
    # introduce root motion that is intentionally absent from the actuated-only J slice.
    sim.forward()
    articulation.update(sim.cfg.dt)

    # body_link_jacobian_w prepends ``num_base_dofs`` floating-base columns; slice past
    # them so the joint axis aligns with joint_vel (actuated-only).
    J = articulation.data.body_link_jacobian_w.torch[..., articulation.num_base_dofs :]
    qdot_view = articulation.data.joint_vel.torch
    v_pred = torch.einsum("nbij,nj->nbi", J, qdot_view)  # (N, B_jac, 6)
    v_pred_lin = v_pred[..., 0:3]
    v_pred_ang = v_pred[..., 3:6]

    # Ground truth from Newton state. ``get_link_velocities`` returns shape
    # (num_instances, 1, num_bodies, 6) — per-articulation grouping with
    # one articulation per instance — so we squeeze the inner dim.
    state = SimulationManager.get_state_0()
    body_qd_view = wp.to_torch(articulation.root_view.get_link_velocities(state)).squeeze(1)
    body_v_com = body_qd_view[..., :3]
    body_omega = body_qd_view[..., 3:]

    # World-frame COM-to-origin offset, derived from already-computed
    # data layer outputs (avoids quaternion-convention pitfalls).
    body_com_pos_w = articulation.data.body_com_pos_w.torch  # (N, num_bodies, 3)
    body_link_pos_w = articulation.data.body_link_pos_w.torch  # (N, num_bodies, 3)
    c_world = body_com_pos_w - body_link_pos_w

    if articulation.is_fixed_base:
        body_v_com = body_v_com[:, 1:]
        body_omega = body_omega[:, 1:]
        c_world = c_world[:, 1:]

    # Expected v_origin = v_com - omega x c_world.
    v_origin_expected = body_v_com - torch.cross(body_omega, c_world, dim=-1)

    # Tolerance: 5 mm absolute. The COM-offset bug produces a ~3 cm bias
    # on the panda hand under the 0.5-rad/s injected qdot, well above
    # this floor; numerical noise from kernel ordering stays under 1 mm.
    torch.testing.assert_close(v_pred_ang, body_omega, atol=5e-3, rtol=1e-2)
    torch.testing.assert_close(v_pred_lin, v_origin_expected, atol=5e-3, rtol=1e-2)


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.isaacsim_ci
def test_get_gravity_compensation_forces_static_equilibrium(sim, num_articulations, device, articulation_type):
    """Newton accuracy: ``τ_gc`` must hold the manipulator in static equilibrium.

    Newton-side variant of the PhysX test of the same name (backend parity).
    The contract is the EOM identity ``M(q) q̈ + C(q,q̇) q̇ + g(q) = τ_input``.
    Setting ``τ_input = g(q)`` at ``q̇ = 0`` gives ``q̈ = 0`` — the arm should
    not move. This pins
    :attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces`
    in isolation: sign errors, frame errors, and DoF-ordering errors all
    surface as joint drift, while a controller-level test would have those
    bugs averaged out by PD damping.
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
    replicate(sim.get_clone_plan())
    sim.reset()
    assert articulation.is_initialized

    # Force a clean static state: default joint positions, zero velocities.
    # ``sim.reset`` may leave residual ``q_dot`` from solver settling under
    # gravity, so we pin it explicitly here.
    default_q = articulation.data.default_joint_pos.torch.clone()
    default_qd = torch.zeros_like(default_q)
    articulation.write_joint_position_to_sim_index(position=default_q)
    articulation.write_joint_velocity_to_sim_index(velocity=default_qd)
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
        # remaining tensor aligns with ``set_joint_effort_target_index`` (actuated only).
        tau_gc = articulation.data.gravity_compensation_forces.torch[:, articulation.num_base_dofs :]
        articulation.set_joint_effort_target_index(target=tau_gc)
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
def test_franka_osc_tracking_accuracy(sim, device, articulation_type, gravity_enabled):
    """Newton-side OSC pose tracking sentinel.

    Mirror of the existing PhysX-side OSC tests in
    :mod:`isaaclab.test.controllers.test_operational_space`, scoped to
    Franka pose-abs tracking on Newton. This test exercises the full controller-bridge pipeline
    (:attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w` +
    :attr:`~isaaclab.assets.BaseArticulationData.mass_matrix`) end-to-end
    and asserts a loose regression bound rather than a tight correctness
    oracle.

    OSC runs with ``gravity_compensation=False`` and scene gravity disabled
    so the sentinel isolates the J/M bridge; the gravity-compensation path is
    covered by :func:`test_franka_osc_gravity_compensation_precision`.
    ``inertial_dynamics_decoupling=True``
    exercises ``mass_matrix`` and the Newton COM-referenced J →
    M_b → J product. The actuator PD is zeroed at cfg time so OSC's
    joint-effort output is not opposed by ``kp·(target − q)``.
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

    pos_mean = _tail_mean(pos_history)
    rot_mean = _tail_mean(rot_history)

    # Regression sentinel: assert on tail mean rather than min. With
    # ``current_ee_vel_b = J · q_dot`` providing OSC's damping term and
    # the actuator PD zeroed, the impedance settles to machine
    # precision. The 5 mm bound is a
    # bridge regression sentinel: a wrong J, wrong mass matrix, or
    # DoF mis-ordering pushes the steady-state error well past it
    # because OSC consumes both ``body_link_jacobian_w`` and
    # ``mass_matrix`` per step.
    assert pos_mean < 5e-3, f"OSC pos_mean {pos_mean:.5f} > 5 mm — bridge regression?"
    assert rot_mean < 5e-2, f"OSC rot_mean {rot_mean:.5f} > 0.05 rad — bridge regression?"


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("articulation_type", ["panda_fine"])
@pytest.mark.parametrize("gravity_enabled", [True])
@pytest.mark.isaacsim_ci
def test_franka_osc_gravity_compensation_precision(sim, device, articulation_type, gravity_enabled):
    """Two-phase EE hold: gravity sag without compensation, tight hold with it.

    Same OSC pose-hold loop as :func:`test_franka_osc_tracking_accuracy`, but
    with scene and per-body gravity ON and the target pinned to the initial EE
    pose, so any steady-state error is pure gravity sag. Phase 1 runs with
    ``gravity_compensation=False`` and must sag past a floor; phase 2 flips
    ``osc.cfg.gravity_compensation`` — read per :meth:`compute` call, so the
    flag is the only variable across phases (the gravity tensor is fetched and
    passed in both) — and must recover the hold to under 0.1 mm.

    The floor assertion keeps the test discriminating: if the task stiffness
    is ever raised high enough to mask gravity, phase 1 stops clearing the
    floor and the test fails loudly instead of silently passing on a
    non-discriminating setup. The gravity feed-forward consumes
    :attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces`
    (Newton RNEA via ``eval_inverse_dynamics_passive``) live in the loop, covering the
    FK-staleness refresh on every step of phase 2.

    The task stiffness (500) deliberately matches the PhysX-side OSC
    gravity-compensation test for a cross-backend-comparable setup, and the
    ``panda_fine`` sim config (4 solver substeps) is load-bearing: at a single
    1/120 substep MJWarp implicitfast mis-integrates the gravity-loaded hold
    into a 0.4-1.0 mm limit cycle that never settles (dt-linear, worsened by
    higher damping gains, gone in zero gravity — solver integration error,
    not a compensation error). With 4 substeps the same per-control-step
    torques hold dead-still at ~1 um, quieter than PhysX (OVPhysX) at ~8 um.
    The uncompensated sag agrees with PhysX to ~1% (23.7 vs 23.9 mm),
    independently validating the gravity forces. Both phases reach a true
    steady state here, enforced by tail-half stationarity guards.
    """
    robot, ee_frame_idx, ee_jacobi_idx, arm_joint_ids = _setup_franka_at_home_pose(
        sim, zero_actuator_pd=True, disable_gravity=False
    )

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
    # Hold the initial EE pose: phase-1 steady-state error is pure gravity sag.
    target_pose_b = _build_relative_pose_target(robot, ee_frame_idx, (0.0, 0.0, 0.0), device)

    def run_phase(num_steps: int) -> list[float]:
        pos_history: list[float] = []
        for _ in range(num_steps):
            jacobian_b = _compute_jacobian_root_frame(robot, ee_jacobi_idx, arm_joint_ids)
            mass_matrix = robot.data.mass_matrix.torch[:, arm_joint_ids, :][:, :, arm_joint_ids]
            gravity = robot.data.gravity_compensation_forces.torch[:, arm_joint_ids]
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
                gravity=gravity,
            )
            robot.set_joint_effort_target(joint_efforts, joint_ids=arm_joint_ids)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.cfg.dt)

            pos_error, _ = compute_pose_error(ee_pos_b, ee_quat_b, target_pose_b[:, 0:3], target_pose_b[:, 3:7])
            pos_history.append(pos_error.norm(dim=-1).max().item())
        return pos_history

    def _stationary_tail_mean(history, label):
        """Mean of the last 200 samples, asserting the two tail halves agree within 25%.

        The relative check carries a 10 µm absolute floor: at the ~1 µm solver noise
        floor of the compensated hold, tail jitter is far below the 0.1 mm verdict
        threshold and cannot flip the outcome, so demanding 25% relative agreement
        of micrometer-scale means would only add GPU-dependent flakiness.
        """
        a = sum(history[-200:-100]) / 100
        b = sum(history[-100:]) / 100
        mean = (a + b) / 2.0
        assert abs(a - b) < 0.25 * max(mean, 1e-5), (
            f"{label} not stationary: tail halves {a:.6f} vs {b:.6f} — extend the phase"
        )
        return mean

    hist_off = run_phase(400)
    osc.cfg.gravity_compensation = True
    hist_on = run_phase(600)

    pos_off = _stationary_tail_mean(hist_off, "phase-1 sag")
    pos_on = _stationary_tail_mean(hist_on, "phase-2 hold")

    # Re-validated on newton 81cdcfc2 / mujoco-warp 3.10.0.2 with 4 substeps:
    # pos_off ~= 0.024, pos_on ~= 1e-6.
    assert pos_off > 1.2e-2, f"uncompensated sag {pos_off:.5f} < 1.2 cm — setup no longer discriminates gravity"
    assert pos_on < 1e-4, f"compensated hold {pos_on:.6f} > 0.1 mm — gravity compensation inaccurate"
    assert pos_on < pos_off / 10.0, f"compensation only improved sag {pos_off:.5f} -> {pos_on:.6f} (<10x)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
