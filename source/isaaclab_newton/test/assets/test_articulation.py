# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

HEADLESS = True

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import sys
from copy import deepcopy

import pytest
import torch
import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton.solvers import SolverNotifyFlags

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ActuatorBase, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
    OperationalSpaceController,
    OperationalSpaceControllerCfg,
)
from isaaclab.envs.mdp.terminations import joint_effort_out_of_limit
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import compute_pose_error, matrix_from_quat, quat_inv, subtract_frame_transforms
from isaaclab.utils.version import get_isaac_sim_version, has_kit

##
# Pre-defined configs
##
from isaaclab_assets import ANYMAL_C_CFG, FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort:skip
# , SHADOW_HAND_CFG  # isort:skip

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
        pytest.skip("Shadow hand is not supported in Newton")
        # articulation_cfg = SHADOW_HAND_CFG
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

    # Fix reversed joints for known-broken USD assets (body0/body1 swapped)
    usd_path = getattr(articulation_cfg.spawn, "usd_path", "")
    if any(name in usd_path for name in _REVERSED_JOINT_USD_FILES):
        import omni.usd

        fix_reversed_joints(omni.usd.get_context().get_stage())

    return articulation, translations


# ---------------------------------------------------------------------------
# Franka task-space tracking helpers (shared between IK and OSC tests).
# ---------------------------------------------------------------------------


def _setup_franka_at_home_pose(sim, *, zero_actuator_pd: bool = False):
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

    Returns:
        Tuple of ``(robot, ee_frame_idx, ee_jacobi_idx, arm_joint_ids)``.
    """
    cfg = FRANKA_PANDA_HIGH_PD_CFG.copy().replace(prim_path="/World/Env_.*/Robot")
    if zero_actuator_pd:
        cfg.actuators["panda_shoulder"].stiffness = 0.0
        cfg.actuators["panda_shoulder"].damping = 0.0
        cfg.actuators["panda_forearm"].stiffness = 0.0
        cfg.actuators["panda_forearm"].damping = 0.0
    sim_utils.create_prim("/World/Env_0", "Xform", translation=(0.0, 0.0, 0.0))
    robot = Articulation(cfg)
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
    ``test_get_jacobians_link_origin_contract``.
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


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["humanoid"])
def test_initialization_floating_base_non_root(sim, num_articulations, device, add_ground_plane, articulation_type):
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
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type, stiffness=0.0, damping=0.0)
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


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("num_articulations", [2, 3])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["humanoid"])
def test_gravity_vec_w_tracks_model_gravity(sim, num_articulations, device, add_ground_plane, articulation_type):
    """Per-env mutations to Newton's ``model.gravity`` reach ``GRAVITY_VEC_W`` and ``projected_gravity_b``.

    Regression for the pre-fix snapshot: ``GRAVITY_VEC_W`` used to be env 0's
    gravity broadcast to every env, hiding per-env gravity randomization (e.g.
    :class:`~isaaclab.envs.mdp.randomize_physics_scene_gravity`).
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type, stiffness=0.0, damping=0.0)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()

    # GRAVITY_VEC_W must share storage with Newton's per-env gravity array.
    model_gravity_arr = SimulationManager.get_model().gravity
    assert articulation.data.GRAVITY_VEC_W.warp.ptr == model_gravity_arr.ptr

    # Mutate model.gravity per-env in place, as randomize_physics_scene_gravity does.
    new_gravity = torch.tensor(
        [[0.1 * (i + 1), 0.2 * (i + 1), -3.0 - float(i)] for i in range(num_articulations)],
        device=device,
        dtype=torch.float32,
    )
    wp.to_torch(model_gravity_arr).copy_(new_gravity)
    SimulationManager.add_model_change(SolverNotifyFlags.MODEL_PROPERTIES)

    # Live view: new per-env values are visible immediately, no invalidation step.
    torch.testing.assert_close(articulation.data.GRAVITY_VEC_W.torch, new_gravity)

    # Recompute the lazily-cached projected_gravity_b without sim.step (which would
    # drift root orientation from the reset state). Project against the same quat
    # buffer the kernel reads so the expectation holds for any default orientation.
    articulation.update(sim.cfg.dt)
    root_quat = articulation.data.root_link_quat_w.torch
    expected = math_utils.quat_apply_inverse(root_quat, torch.nn.functional.normalize(new_gravity, dim=-1))
    torch.testing.assert_close(articulation.data.projected_gravity_b.torch, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_initialization_floating_base(sim, num_articulations, device, add_ground_plane, articulation_type):
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
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type, stiffness=0.0, damping=0.0)
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("articulation_type", ["panda"])
def test_initialization_fixed_base(sim, num_articulations, device, articulation_type):
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
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["single_joint_implicit"])
def test_initialization_fixed_base_single_joint(sim, num_articulations, device, add_ground_plane, articulation_type):
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
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("articulation_type", ["shadow_hand"])
def test_initialization_hand_with_tendons(sim, num_articulations, device, articulation_type):
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
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_initialization_floating_base_made_fixed_base(
    sim, num_articulations, device, add_ground_plane, articulation_type
):
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
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type).copy()
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
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
    4. The articulation can be simulated

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type).copy()
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

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update articulation
        articulation.update(sim.cfg.dt)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["panda"])
def test_out_of_range_default_joint_pos(sim, num_articulations, device, add_ground_plane, articulation_type):
    """Test that the default joint position from configuration is out of range.

    This test verifies that:
    1. The articulation fails to initialize when joint positions are out of range
    2. The error is properly handled

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    # Create articulation
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type).copy()
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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("articulation_type", ["panda"])
def test_out_of_range_default_joint_vel(sim, device, articulation_type):
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["panda"])
def test_joint_pos_limits(sim, num_articulations, device, add_ground_plane, articulation_type):
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
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["panda"])
def test_joint_effort_limits(sim, num_articulations, device, add_ground_plane, articulation_type):
    """Validate joint effort limits via joint_effort_out_of_limit()."""
    # Create articulation
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_external_force_buffer(sim, num_articulations, device, articulation_type):
    """Test if external force buffer correctly updates in the force value is zero case.

    This test verifies that:
    1. External forces can be applied correctly
    2. Force buffers are updated properly
    3. Zero forces are handled correctly

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_external_force_on_single_body(sim, num_articulations, device, articulation_type):
    """Test application of external force on the base of the articulation.

    This test verifies that:
    1. External forces can be applied to specific bodies
    2. The forces affect the articulation's motion correctly
    3. The articulation responds to the forces as expected

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)
    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies("base")
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
        # check condition that the articulations have fallen down
        for i in range(num_articulations):
            assert articulation.data.root_pos_w.torch[i, 2].item() < 0.2


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_external_force_on_single_body_at_position(sim, num_articulations, device, articulation_type):
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
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=sim.device)
    # Play the simulator
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies("base")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 2] = 100.0
    external_wrench_positions_b = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    external_wrench_positions_b[..., 1] = 1.0

    desired_force = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    desired_force[..., 2] = 200.0
    desired_torque = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    desired_torque[..., 0] = 200.0

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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
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
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies(".*_SHANK")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 1] = 200.0

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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
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
    sim.reset()

    # Find bodies to apply the force
    body_ids, _ = articulation.find_bodies(".*_SHANK")
    # Sample a large force
    external_wrench_b = torch.zeros(articulation.num_instances, len(body_ids), 6, device=sim.device)
    external_wrench_b[..., 2] = 100.0
    external_wrench_positions_b = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    external_wrench_positions_b[..., 1] = 1.0

    desired_force = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    desired_force[..., 2] = 200.0
    desired_torque = torch.zeros(articulation.num_instances, len(body_ids), 3, device=sim.device)
    desired_torque[..., 0] = 200.0

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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["humanoid"])
def test_setting_gains_from_cfg(sim, num_articulations, device, add_ground_plane, articulation_type):
    """Test that gains are loaded from the configuration correctly.

    This test verifies that:
    1. Gains are loaded correctly from configuration
    2. The gains match the expected values
    3. The gains are applied correctly to the actuators

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("articulation_type", ["humanoid"])
def test_setting_gains_from_cfg_dict(sim, num_articulations, device, articulation_type):
    """Test that gains are loaded from the configuration dictionary correctly.

    This test verifies that:
    1. Gains are loaded correctly from configuration dictionary
    2. The gains match the expected values
    3. The gains are applied correctly to the actuators

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("vel_limit_sim", [1e5, None])
@pytest.mark.parametrize("vel_limit", [1e2, None])
@pytest.mark.parametrize("add_ground_plane", [False])
@pytest.mark.parametrize("articulation_type", ["single_joint_implicit"])
def test_setting_velocity_limit_implicit(
    sim, num_articulations, device, vel_limit_sim, vel_limit, add_ground_plane, articulation_type
):
    """Test setting of velocity limit for implicit actuators.

    This test verifies that:
    1. Velocity limits can be set correctly for implicit actuators
    2. The limits are applied correctly to the simulation
    3. The limits are handled correctly when both sim and non-sim limits are set

    Args:
        sim: The simulation fixture
        num_articulations: Number of articulations to test
        device: The device to run the simulation on
        vel_limit_sim: The velocity limit to set in simulation
        vel_limit: The velocity limit to set in actuator
    """
    # create simulation
    articulation_cfg = generate_articulation_cfg(
        articulation_type=articulation_type,
        velocity_limit_sim=vel_limit_sim,
        velocity_limit=vel_limit,
    )
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg,
        num_articulations=num_articulations,
        device=device,
    )
    # Play sim
    if vel_limit_sim is not None and vel_limit is not None:
        with pytest.raises(ValueError):
            sim.reset()
        return
    sim.reset()

    # read the values set into the simulation
    newton_vel_limit = wp.to_torch(
        articulation.root_view.get_attribute("joint_velocity_limit", SimulationManager.get_model())
    ).to(device)[:, 0, :]
    # check data buffer
    torch.testing.assert_close(articulation.data.joint_velocity_limits.torch, newton_vel_limit)
    # check actuator has simulation velocity limit
    torch.testing.assert_close(articulation.actuators["joint"].velocity_limit_sim, newton_vel_limit)
    # check that both values match for velocity limit
    torch.testing.assert_close(
        articulation.actuators["joint"].velocity_limit_sim,
        articulation.actuators["joint"].velocity_limit,
    )

    if vel_limit_sim is None:
        # Case 2: both velocity limit and velocity limit sim are not set
        #  This is the case where the velocity limit keeps its USD default value
        # Case 3: velocity limit sim is not set but velocity limit is set
        #   For backwards compatibility, we do not set velocity limit to simulation
        #   Thus, both default to USD default value.
        limit = articulation_cfg.spawn.joint_drive_props.max_joint_velocity
    else:
        # Case 4: only velocity limit sim is set
        #   In this case, the velocity limit is set to the USD value
        limit = vel_limit_sim

    # check max velocity is what we set
    expected_velocity_limit = torch.full_like(newton_vel_limit, limit)
    torch.testing.assert_close(newton_vel_limit, expected_velocity_limit)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("vel_limit_sim", [1e5, None])
@pytest.mark.parametrize("vel_limit", [1e2, None])
@pytest.mark.parametrize("articulation_type", ["single_joint_explicit"])
def test_setting_velocity_limit_explicit(sim, num_articulations, device, vel_limit_sim, vel_limit, articulation_type):
    """Test setting of velocity limit for explicit actuators."""
    articulation_cfg = generate_articulation_cfg(
        articulation_type=articulation_type,
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
    newton_vel_limit = wp.to_torch(
        articulation.root_view.get_attribute("joint_velocity_limit", SimulationManager.get_model())
    ).to(device)[:, 0, :]
    actuator_vel_limit = articulation.actuators["joint"].velocity_limit
    actuator_vel_limit_sim = articulation.actuators["joint"].velocity_limit_sim

    # check data buffer for joint_velocity_limits_sim
    torch.testing.assert_close(articulation.data.joint_velocity_limits.torch, newton_vel_limit)
    # check actuator velocity_limit_sim is set to physx
    torch.testing.assert_close(actuator_vel_limit_sim, newton_vel_limit)

    if vel_limit is not None:
        expected_actuator_vel_limit = torch.full(
            (articulation.num_instances, articulation.num_joints),
            vel_limit,
            device=articulation.device,
        )
        # check actuator is set
        torch.testing.assert_close(actuator_vel_limit, expected_actuator_vel_limit)
        # check physx is not velocity_limit
        assert not torch.allclose(actuator_vel_limit, newton_vel_limit)
    else:
        # check actuator velocity_limit is the same as the PhysX default
        torch.testing.assert_close(actuator_vel_limit, newton_vel_limit)

    # simulation velocity limit is set to USD value unless user overrides
    if vel_limit_sim is not None:
        limit = vel_limit_sim
    else:
        limit = articulation_cfg.spawn.joint_drive_props.max_joint_velocity
    # check physx is set to expected value
    expected_vel_limit = torch.full_like(newton_vel_limit, limit)
    torch.testing.assert_close(newton_vel_limit, expected_vel_limit)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("effort_limit_sim", [1e5, None])
@pytest.mark.parametrize("effort_limit", [1e2, 80.0, None])
@pytest.mark.parametrize("articulation_type", ["single_joint_implicit"])
def test_setting_effort_limit_implicit(
    sim, num_articulations, device, effort_limit_sim, effort_limit, articulation_type
):
    """Test setting of effort limit for implicit actuators.

    This test verifies the effort limit resolution logic for actuator models implemented in :class:`ActuatorBase`:
    - Case 1: If USD value == actuator config value: values match correctly
    - Case 2: If USD value != actuator config value: actuator config value is used
    - Case 3: If actuator config value is None: USD value is used as default
    """
    articulation_cfg = generate_articulation_cfg(
        articulation_type=articulation_type,
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
    newton_effort_limit = wp.to_torch(
        articulation.root_view.get_attribute("joint_effort_limit", SimulationManager.get_model())
    ).to(device)[:, 0, :]

    # check that the two are equivalent
    torch.testing.assert_close(
        articulation.actuators["joint"].effort_limit_sim,
        articulation.actuators["joint"].effort_limit,
    )
    torch.testing.assert_close(articulation.actuators["joint"].effort_limit_sim, newton_effort_limit)

    # decide the limit based on what is set
    if effort_limit_sim is None and effort_limit is None:
        limit = articulation_cfg.spawn.joint_drive_props.max_force
    elif effort_limit_sim is not None and effort_limit is None:
        limit = effort_limit_sim
    elif effort_limit_sim is None and effort_limit is not None:
        limit = effort_limit

    # check that the max force is what we set
    expected_effort_limit = torch.full_like(newton_effort_limit, limit)
    torch.testing.assert_close(newton_effort_limit, expected_effort_limit)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("effort_limit_sim", [1e5, None])
@pytest.mark.parametrize("effort_limit", [80.0, 1e2, None])
@pytest.mark.parametrize("articulation_type", ["single_joint_explicit"])
def test_setting_effort_limit_explicit(
    sim, num_articulations, device, effort_limit_sim, effort_limit, articulation_type
):
    """Test setting of effort limit for explicit actuators.

    This test verifies the effort limit resolution logic for actuator models implemented in :class:`ActuatorBase`:
    - Case 1: If USD value == actuator config value: values match correctly
    - Case 2: If USD value != actuator config value: actuator config value is used
    - Case 3: If actuator config value is None: USD value is used as default

    """

    articulation_cfg = generate_articulation_cfg(
        articulation_type=articulation_type,
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
    newton_effort_limit = wp.to_torch(
        articulation.root_view.get_attribute("joint_effort_limit", SimulationManager.get_model())
    ).to(device)[:, 0, :]
    actuator_effort_limit = articulation.actuators["joint"].effort_limit
    actuator_effort_limit_sim = articulation.actuators["joint"].effort_limit_sim

    # check actuator effort_limit_sim is set to physx
    torch.testing.assert_close(actuator_effort_limit_sim, newton_effort_limit)

    if effort_limit is not None:
        expected_actuator_effort_limit = torch.full_like(actuator_effort_limit, effort_limit)
        # check actuator is set
        torch.testing.assert_close(actuator_effort_limit, expected_actuator_effort_limit)

        # check physx effort limit does not match the one explicit actuator has
        assert not (torch.allclose(actuator_effort_limit, newton_effort_limit))
    else:
        # When effort_limit is None, actuator should use USD default values
        expected_actuator_effort_limit = torch.full_like(newton_effort_limit, usd_default_effort_limit)
        torch.testing.assert_close(actuator_effort_limit, expected_actuator_effort_limit)

    # when using explicit actuators, the limits are set to high unless user overrides
    if effort_limit_sim is not None:
        limit = effort_limit_sim
    else:
        limit = ActuatorBase._DEFAULT_MAX_EFFORT_SIM  # type: ignore
    # check physx internal value matches the expected sim value
    expected_effort_limit = torch.full_like(newton_effort_limit, limit)
    torch.testing.assert_close(actuator_effort_limit_sim, expected_effort_limit)
    torch.testing.assert_close(newton_effort_limit, expected_effort_limit)


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("articulation_type", ["humanoid"])
def test_reset(sim, num_articulations, device, articulation_type):
    """Test that reset method works properly."""
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["panda"])
def test_apply_joint_command(sim, num_articulations, device, add_ground_plane, articulation_type):
    """Test applying of joint position target functions correctly for a robotic arm."""
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.parametrize("articulation_type", ["single_joint_implicit"])
def test_body_root_state(sim, num_articulations, device, with_offset, articulation_type):
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
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)
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
    com = wp.to_torch(articulation.root_view.get_attribute("body_com", SimulationManager.get_model()))
    link_offset = [1.0, 0.0, 0.0]  # the offset from CenterPivot to Arm frames
    new_com = torch.tensor(offset, device=device).repeat(num_articulations, 1, 1)
    com[:, 0, arm_idx, :] = new_com.squeeze(-2)
    articulation.root_view.set_attribute("body_com", SimulationManager.get_model(), wp.from_torch(com, dtype=wp.vec3f))
    with wp.ScopedDevice(device):
        SimulationManager._solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    # check they are set
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_attribute("body_com", SimulationManager.get_model())), com
    )

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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True, False])
@pytest.mark.parametrize("state_location", ["com", "link"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_write_root_state(
    sim, num_articulations, device, with_offset, state_location, gravity_enabled, articulation_type
):
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
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)
    env_idx = torch.tensor([x for x in range(num_articulations)], device=device, dtype=torch.int32)

    # Play sim
    sim.reset()

    # Resolve root body index by name (ordering may differ across physics backends)
    root_idx = articulation.find_bodies("base")[0][0]

    # change center of mass offset from link frame
    if with_offset:
        offset = torch.tensor([1.0, 0.0, 0.0]).repeat(num_articulations, 1, 1)
    else:
        offset = torch.tensor([0.0, 0.0, 0.0]).repeat(num_articulations, 1, 1)

    # create com offsets
    com = wp.to_torch(articulation.root_view.get_attribute("body_com", SimulationManager.get_model()))
    new_com = offset
    com[:, 0, root_idx, :] = new_com.squeeze(-2)
    articulation.root_view.set_attribute("body_com", SimulationManager.get_model(), wp.from_torch(com, dtype=wp.vec3f))
    with wp.ScopedDevice(device):
        SimulationManager._solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    # check they are set
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_attribute("body_com", SimulationManager.get_model())), com
    )

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


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("with_offset", [True])
@pytest.mark.parametrize("state_location", ["com", "link", "root"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_write_root_state_functions_data_consistency(
    sim, num_articulations, device, with_offset, state_location, gravity_enabled, articulation_type
):
    """A root pose/velocity write must refresh the derived cross-frame caches without a sim step.

    Regression coverage for the velocity invalidation cleanup: writing the root center-of-mass
    (or link) velocity must invalidate the derived root link (or com) velocity so the next read
    re-derives it. Linear velocity differs between the two frames, so - as in the rigid object
    test - we compare angular velocity, which is frame-independent and therefore only matches when
    the derived velocity was actually refreshed.
    """
    sim._app_control_on_stop_handle = None
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, env_pos = generate_articulation(articulation_cfg, num_articulations, device)

    # Play sim
    sim.reset()

    # Resolve root body index by name (ordering may differ across physics backends)
    root_idx = articulation.find_bodies("base")[0][0]

    # change center of mass offset from link frame on the root body
    if with_offset:
        offset = torch.tensor([1.0, 0.0, 0.0]).repeat(num_articulations, 1, 1)
    else:
        offset = torch.tensor([0.0, 0.0, 0.0]).repeat(num_articulations, 1, 1)
    com = wp.to_torch(articulation.root_view.get_attribute("body_com", SimulationManager.get_model()))
    com[:, 0, root_idx, :] = offset.squeeze(-2)
    articulation.root_view.set_attribute("body_com", SimulationManager.get_model(), wp.from_torch(com, dtype=wp.vec3f))
    with wp.ScopedDevice(device):
        SimulationManager._solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    rand_state = torch.rand(num_articulations, 13, device=device)
    rand_state[..., :3] += env_pos
    # make quaternion a unit vector
    rand_state[..., 3:7] = torch.nn.functional.normalize(rand_state[..., 3:7], dim=-1)

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

    if state_location == "com":
        articulation.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7])
        articulation.write_root_com_velocity_to_sim_index(root_velocity=rand_state[..., 7:])
    elif state_location == "link":
        articulation.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7])
        articulation.write_root_link_velocity_to_sim_index(root_velocity=rand_state[..., 7:])
    elif state_location == "root":
        articulation.write_root_pose_to_sim_index(root_pose=rand_state[..., :7])
        articulation.write_root_velocity_to_sim_index(root_velocity=rand_state[..., 7:])

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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("articulation_type", ["humanoid"])
def test_setting_articulation_root_prim_path(sim, device, articulation_type):
    """Test that the articulation root prim path can be set explicitly."""
    sim._app_control_on_stop_handle = None
    # Create articulation
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation_cfg.articulation_root_prim_path = "/torso"
    articulation, _ = generate_articulation(articulation_cfg, 1, device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    # Play sim
    sim.reset()
    # Check if articulation is initialized
    assert articulation._is_initialized


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("articulation_type", ["humanoid"])
def test_setting_invalid_articulation_root_prim_path(sim, device, articulation_type):
    """Test that the articulation root prim path can be set explicitly."""
    sim._app_control_on_stop_handle = None
    # Create articulation
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation_cfg.articulation_root_prim_path = "/non_existing_prim_path"
    articulation, _ = generate_articulation(articulation_cfg, 1, device=device)

    # Check that the framework doesn't hold excessive strong references.
    assert sys.getrefcount(articulation) < 10

    # Play sim
    with pytest.raises((RuntimeError, KeyError)):
        sim.reset()


@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_write_joint_state_data_consistency(sim, num_articulations, device, gravity_enabled, articulation_type):
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
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("articulation_type", ["shadow_hand"])
@pytest.mark.skip(reason="Spatial tendons are not supported in Newton yet.")
def test_spatial_tendons(sim, num_articulations, device, articulation_type):
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
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("articulation_type", ["panda"])
def test_write_joint_frictions_to_sim(sim, num_articulations, device, add_ground_plane, articulation_type):
    """Test applying of joint position target functions correctly for a robotic arm."""
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
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
    friction = torch.rand(num_articulations, articulation.num_joints, device=device)

    # The static friction must be set first to be sure the dynamic friction is not greater than static
    # when both are set.
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=friction,
    )
    articulation.write_data_to_sim()

    for _ in range(100):
        # perform step
        sim.step()
        # update buffers
        articulation.update(sim.cfg.dt)

    joint_friction_coeff_sim = wp.to_torch(
        articulation.root_view.get_attribute("joint_friction", SimulationManager.get_model())
    )[:, 0, :]
    assert torch.allclose(joint_friction_coeff_sim, friction)

    # Reset simulator to ensure a clean state for the alternative API path
    sim.reset()

    # Warm up a few steps to populate buffers
    for _ in range(100):
        sim.step()
        articulation.update(sim.cfg.dt)

    # New random coefficients
    friction_2 = torch.rand(num_articulations, articulation.num_joints, device=device)

    # Use the combined setter to write all three at once
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=friction_2,
    )
    articulation.write_data_to_sim()

    # Step to let sim ingest new params and refresh data buffers
    for _ in range(100):
        sim.step()
        articulation.update(sim.cfg.dt)

    joint_friction_coeff_sim_2 = wp.to_torch(
        articulation.root_view.get_attribute("joint_friction", SimulationManager.get_model())
    )[:, 0, :]

    # Validate values propagated
    assert torch.allclose(joint_friction_coeff_sim_2, friction_2)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", ["cuda:0"])
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
                bq = wp.to_torch(cls._state_0.body_q)
                jq = wp.to_torch(cls._state_0.joint_q)
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
@pytest.mark.parametrize("num_articulations", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("articulation_type", ["panda"])
def test_set_material_properties(sim, num_articulations, device, add_ground_plane, articulation_type):
    """Test getting and setting material properties (friction/restitution) via view-level APIs."""
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(
        articulation_cfg=articulation_cfg, num_articulations=num_articulations, device=device
    )

    # Play the simulator
    sim.reset()

    # Get friction/restitution bindings via view-level API
    model = SimulationManager.get_model()
    friction_binding = articulation._root_view.get_attribute("shape_material_mu", model)[:, 0]
    restitution_binding = articulation._root_view.get_attribute("shape_material_restitution", model)[:, 0]
    num_shapes = friction_binding.shape[1]

    # Test 1: Set all shapes via in-place writes to the warp binding
    friction = torch.empty(num_articulations, num_shapes, device=device).uniform_(0.4, 0.8)
    restitution = torch.empty(num_articulations, num_shapes, device=device).uniform_(0.0, 0.2)

    wp.to_torch(friction_binding)[:] = friction
    wp.to_torch(restitution_binding)[:] = restitution
    SimulationManager.add_model_change(SolverNotifyFlags.SHAPE_PROPERTIES)

    # Simulate physics
    sim.step()
    articulation.update(sim.cfg.dt)

    # Verify by reading back from the binding
    mu = wp.to_torch(friction_binding)
    restitution_check = wp.to_torch(restitution_binding)
    torch.testing.assert_close(mu, friction)
    torch.testing.assert_close(restitution_check, restitution)

    # Test 2: Set subset of shapes (only shape 0)
    if num_shapes > 1:
        subset_friction = torch.empty(num_articulations, device=device).uniform_(0.1, 0.2)
        subset_restitution = torch.empty(num_articulations, device=device).uniform_(0.5, 0.6)

        wp.to_torch(friction_binding)[:, 0] = subset_friction
        wp.to_torch(restitution_binding)[:, 0] = subset_restitution
        SimulationManager.add_model_change(SolverNotifyFlags.SHAPE_PROPERTIES)

        sim.step()
        articulation.update(sim.cfg.dt)

        # Check only the subset was updated
        mu_updated = wp.to_torch(friction_binding)
        restitution_updated = wp.to_torch(restitution_binding)
        torch.testing.assert_close(mu_updated[:, 0], subset_friction)
        torch.testing.assert_close(restitution_updated[:, 0], subset_restitution)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_randomize_rigid_body_com(sim, num_articulations, device, add_ground_plane, articulation_type):
    """Test that randomize_rigid_body_com modifies CoM and affects simulation dynamics."""
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)

    sim.reset()
    assert articulation.is_initialized

    original_com = articulation.data.body_com_pos_b.torch.clone()

    com_offset = torch.zeros(num_articulations, articulation.num_bodies, 3, device=device)
    com_offset[..., 0] = 0.5
    new_com = original_com + com_offset
    env_ids = torch.arange(num_articulations, device=device, dtype=torch.int32)
    articulation.set_coms_index(coms=new_com, env_ids=env_ids)

    updated_com = articulation.data.body_com_pos_b.torch
    torch.testing.assert_close(updated_com, new_com, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_articulations", [2])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["anymal"])
def test_randomize_rigid_body_collider_offsets(sim, num_articulations, device, add_ground_plane, articulation_type):
    """Test that Newton collider offset randomization (shape_margin, shape_gap) takes effect."""
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)

    sim.reset()
    assert articulation.is_initialized

    model = SimulationManager.get_model()
    original_margin = wp.to_torch(articulation.root_view.get_attribute("shape_margin", model)).clone()
    original_gap = wp.to_torch(articulation.root_view.get_attribute("shape_gap", model)).clone()

    new_margin = original_margin.clone()
    new_margin[:, 0] += 0.01
    articulation.root_view.set_attribute("shape_margin", model, wp.from_torch(new_margin, dtype=wp.float32))

    new_gap = original_gap.clone()
    new_gap[:, 0] += 0.005
    articulation.root_view.set_attribute("shape_gap", model, wp.from_torch(new_gap, dtype=wp.float32))

    with wp.ScopedDevice(device):
        SimulationManager._solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)

    updated_margin = wp.to_torch(articulation.root_view.get_attribute("shape_margin", model))
    updated_gap = wp.to_torch(articulation.root_view.get_attribute("shape_gap", model))
    torch.testing.assert_close(updated_margin, new_margin)
    torch.testing.assert_close(updated_gap, new_gap)


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.isaacsim_ci
@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "Newton's ArticulationView exposes eval_fk / eval_jacobian /"
        " eval_mass_matrix only — no inverse-dynamics primitive yet."
        " Upstream Newton is actively working on this through the inverse-"
        " dynamics feature request (https://github.com/newton-physics/newton/issues/2497)"
        " and its sub-task for Coriolis + gravity compensation"
        " (https://github.com/newton-physics/newton/issues/2529). A known"
        " correctness bug for floating-base + non-identity root pose is"
        " tracked separately at"
        " https://github.com/newton-physics/newton/issues/2625, and"
        " informs why we deliberately do NOT roll our own J^T·m·g shim in"
        " this PR — Newton's eventual primitive is going through RNEA via"
        " MuJoCo Warp and may differ at corner cases we wouldn't catch."
        " Once the wrapper at"
        " isaaclab_newton.assets.ArticulationData.gravity_compensation_forces"
        " switches from a NotImplementedError stub to a real implementation"
        " (likely calling the new Newton primitive), this XFAIL will turn"
        " into XPASS and fail under strict=True. The maintainer should"
        " then: (1) drop this xfail or invert it into a positive value"
        " assertion against PhysX (the cross-backend accuracy diff), and"
        " (2) remove the OSC config-time guidance about setting"
        " gravity_compensation=False on Newton."
    ),
)
def test_get_gravity_compensation_forces_not_implemented_on_newton(sim, num_articulations, device, articulation_type):
    """Pin the known Newton gravity-compensation gap.

    See the ``xfail`` marker for full rationale. The body simply invokes the
    wrapper and lets the strict-xfail marker handle the expected failure.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    assert articulation.is_initialized

    _ = articulation.data.gravity_compensation_forces


##
# Shape-contract regression tests for the new BaseArticulation accessors.
# These pin the public shape contract so future regressions (e.g., reverting
# to model-wide max sizing or to the wrong fixed-base row offset) fail fast.
##


@pytest.mark.parametrize("num_articulations", [1, 4])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.isaacsim_ci
def test_get_jacobians_shape_fixed_base(sim, num_articulations, device, articulation_type):
    """Fixed-base ``body_link_jacobian_w`` must drop the fixed-root row.

    Contract: shape ``(N, num_bodies - 1, 6, num_joints)``. Catches
    regressions of (a) the link_offset fix that drops Newton's row 0 for
    fixed-base, and (b) the per-articulation output sizing — using
    model-wide ``max_links`` here would over-allocate in heterogeneous
    scenes.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    assert articulation.is_initialized
    assert articulation.is_fixed_base, "panda fixture must be fixed-base for this test"

    J = articulation.data.body_link_jacobian_w.torch

    expected_shape = (num_articulations, articulation.num_bodies - 1, 6, articulation.num_joints)
    assert J.shape == torch.Size(expected_shape), f"expected {expected_shape}, got {tuple(J.shape)}"
    assert J.dtype == torch.float32


@pytest.mark.parametrize("num_articulations", [1, 4])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.isaacsim_ci
def test_get_mass_matrix_shape_and_nonsingular_fixed_base(sim, num_articulations, device, articulation_type):
    """Fixed-base ``mass_matrix`` shape + non-singularity.

    Contract: shape ``(N, num_joints, num_joints)`` and the matrix must be
    non-singular. The non-singularity check catches the heterogeneous
    padding bug — if the wrapper accidentally returns ``model.max_dofs``
    sized output, the padded zero rows/cols make the matrix rank-deficient.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    assert articulation.is_initialized

    sim.step()
    articulation.update(sim.cfg.dt)

    M = articulation.data.mass_matrix.torch

    expected_shape = (num_articulations, articulation.num_joints, articulation.num_joints)
    assert M.shape == torch.Size(expected_shape), f"expected {expected_shape}, got {tuple(M.shape)}"
    assert M.dtype == torch.float32

    # Each diagonal entry is a joint's effective inertia and must be strictly
    # positive for any physical articulation. Padded zero rows/cols (the
    # heterogeneous bug) would surface as zero diagonal entries — much more
    # sensitive than checking the determinant, which can be small purely from
    # numerical conditioning of a well-formed 9x9 mass matrix (Franka det
    # is ~1e-13 in practice).
    diag = M.diagonal(dim1=-2, dim2=-1)
    assert (diag > 1e-6).all(), f"mass matrix has non-positive diagonal entries: min={diag.min()}"


@pytest.mark.parametrize("num_articulations", [1, 4])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["anymal"])
@pytest.mark.isaacsim_ci
def test_get_jacobians_shape_floating_base(sim, num_articulations, device, add_ground_plane, articulation_type):
    """Floating-base ``body_link_jacobian_w`` keeps every body row and prepends 6 base-DoF columns.

    Contract for floating-base: shape
    ``(N, num_bodies, 6, num_joints + num_base_dofs)`` — no fixed-root row
    to drop, and the leading 6 DoF columns are the floating-base spatial-
    velocity columns Newton's ``eval_jacobian`` writes for the free root
    joint. Matches the cross-library industry convention (Pinocchio, Drake,
    MuJoCo, RBDL, OCS2, iDynTree).
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    assert articulation.is_initialized
    assert not articulation.is_fixed_base, "anymal fixture must be floating-base for this test"

    J = articulation.data.body_link_jacobian_w.torch

    expected_shape = (
        num_articulations,
        articulation.num_bodies,
        6,
        articulation.num_joints + articulation.num_base_dofs,
    )
    assert J.shape == torch.Size(expected_shape), f"expected {expected_shape}, got {tuple(J.shape)}"
    assert J.dtype == torch.float32


@pytest.mark.parametrize("num_articulations", [1, 4])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("add_ground_plane", [True])
@pytest.mark.parametrize("articulation_type", ["anymal"])
@pytest.mark.isaacsim_ci
def test_get_mass_matrix_shape_floating_base(sim, num_articulations, device, add_ground_plane, articulation_type):
    """Floating-base ``mass_matrix`` shape ``(N, num_joints + 6, num_joints + 6)``.

    Includes the 6 floating-base rows/cols on the DoF axis, matching the
    cross-library industry convention.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    assert articulation.is_initialized

    sim.step()
    articulation.update(sim.cfg.dt)

    M = articulation.data.mass_matrix.torch

    expected_dofs = articulation.num_joints + articulation.num_base_dofs
    expected_shape = (num_articulations, expected_dofs, expected_dofs)
    assert M.shape == torch.Size(expected_shape), f"expected {expected_shape}, got {tuple(M.shape)}"
    assert M.dtype == torch.float32


@pytest.mark.parametrize("device", ["cuda:0"])
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

    franka_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Env_franka_.*/Robot")
    anymal_cfg = ANYMAL_C_CFG.replace(prim_path="/World/Env_anymal_.*/Robot")

    for i in range(num_per_type):
        sim_utils.create_prim(f"/World/Env_franka_{i}", "Xform", translation=(2.5 * i, 0.0, 0.0))
        sim_utils.create_prim(f"/World/Env_anymal_{i}", "Xform", translation=(2.5 * i, 5.0, 0.0))

    franka = Articulation(franka_cfg)
    anymal = Articulation(anymal_cfg)
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


@pytest.mark.parametrize("num_articulations", [4])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("articulation_type", ["panda", "anymal"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_get_jacobians_link_origin_contract(sim, num_articulations, device, articulation_type, gravity_enabled):
    """``J · q_dot`` must encode the link-origin twist (after the COM->origin shift).

    The IsaacLab task-space controllers (IK / OSC / RMPFlow) silently
    rely on :attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w`
    returning a Jacobian whose linear rows reference each link's origin
    (the body's USD prim transform), not its COM. Newton's ``eval_jacobian``
    natively produces COM-referenced rows; the wrapper applies a per-column
    shift ``v_origin = v_com - omega x (R · body_com_pos_b)`` to honor the
    contract. This test asserts the identity by computing both sides
    independently:

    * Predicted by ``J · q_dot``: takes the (already-shifted) Jacobian
      and the same ``q_dot`` Newton has post-step. Linear rows should
      equal v_origin.
    * Ground truth from ``state.body_qd``: read Newton's per-body spatial
      twist directly via ``ArticulationView.get_link_velocities`` (which
      returns ``(v_com_world, omega_world)``), then apply the same shift
      in python and compare.

    Reading the velocity from the ArticulationView state rather than
    ``data.body_com_lin_vel_w`` bypasses the IsaacLab lazy-buffer chain,
    which is irrelevant to the contract being tested.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    assert articulation.is_initialized

    # Reproducible non-trivial q_dot — large enough to drive omega well above
    # the floor where COM offset effects would round into noise.
    torch.manual_seed(0)
    qdot = torch.randn(num_articulations, articulation.num_joints, device=device) * 0.5
    articulation.write_joint_velocity_to_sim_index(velocity=qdot)
    sim.step()
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


@pytest.mark.parametrize("num_articulations", [4])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("articulation_type", ["panda", "anymal"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_get_mass_matrix_symmetry_pd(sim, num_articulations, device, articulation_type, gravity_enabled):
    """The joint-space mass matrix ``M(q)`` must be square, symmetric, and positive-definite.

    This pins three structural properties of
    :attr:`~isaaclab.assets.BaseArticulationData.mass_matrix`:

    * **Square**: shape ``(N, num_joints + num_base_dofs, num_joints + num_base_dofs)``.
      A transposed gather or a non-square scratch buffer would be caught
      here before downstream OSC inversion silently propagates garbage.
    * **Symmetric**: ``M == M.T`` to numerical precision. The joint-
      space inertia tensor is symmetric by construction; an asymmetric
      result indicates a wrong-axis gather, half-populated buffer, or
      Cholesky-input bug.
    * **Positive-definite**: ``torch.linalg.cholesky(M)`` succeeds. OSC
      computes ``M_b = (J · M^-1 · J^T)^-1`` which requires PD on every
      step. A non-PD M would fail downstream as ``LinAlgError``; this
      test catches it earlier and pinpoints the source.

    Parameterized on both fixed-base (panda) and floating-base (anymal).
    Both backends include the floating-base DoF rows/cols on the front of
    the DoF axis for floating-base assets.
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

    # Symmetric to numerical precision.
    asym = (M - M.transpose(-1, -2)).abs().max().item()
    assert asym < 1e-4, f"|M - M^T|_max = {asym:.3e} — mass matrix is not symmetric"

    # Positive-definite via Cholesky. Adds a tiny diagonal jitter to
    # tolerate the floor of float32 PD eigenvalues without masking real
    # non-PD bugs (the jitter is well below realistic inertia scales).
    eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype).expand_as(M)
    torch.linalg.cholesky(M + 1e-6 * eye)


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("articulation_type", ["panda", "anymal"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_jacobian_refreshes_after_manual_joint_write(
    sim, num_articulations, device, articulation_type, gravity_enabled
):
    """After ``write_joint_position_to_sim_index`` (no sim step), the Jacobian read
    must reflect the new joint state — not the previous one.

    Catches:
      - Missing FK trigger in :attr:`body_com_jacobian_w` (eval_jacobian uses stale
        ``state.body_q``).
      - Missing FK trigger in :attr:`body_link_jacobian_w` shift kernel.

    The contract: ``J`` read directly after a manual write must equal ``J`` read
    after ``sim.step + update`` — the latter is the ground-truth fresh-FK reference.
    """
    articulation_cfg = generate_articulation_cfg(articulation_type=articulation_type)
    articulation, _ = generate_articulation(articulation_cfg, num_articulations, device=device)
    sim.reset()
    sim.step()
    articulation.update(sim.cfg.dt)

    # Read J / M at the baseline joint state.
    q_baseline = articulation.data.joint_pos.torch.clone()
    J_link_0 = articulation.data.body_link_jacobian_w.torch.clone()
    J_com_0 = articulation.data.body_com_jacobian_w.torch.clone()

    # Manually write a different joint state — large delta to make Jacobian change visible.
    # No sim.step / update — FK becomes stale (write_joint_position_to_sim sets _fk_timestamp = -1).
    q_target = q_baseline + 0.5
    env_ids = wp.array([0], dtype=wp.int32, device=device)
    articulation.write_joint_position_to_sim_index(position=q_target, env_ids=env_ids)

    # If the FK trigger works: forward() runs, body_q is refreshed to match q_target,
    # eval_jacobian / shift kernel see fresh body poses, J reflects q_target → differs from J at baseline.
    # If the trigger is missing: body_q stays at baseline, J unchanged from J_link_0 / J_com_0.
    J_link_1 = articulation.data.body_link_jacobian_w.torch.clone()
    J_com_1 = articulation.data.body_com_jacobian_w.torch.clone()

    assert not torch.allclose(J_link_0, J_link_1, atol=1e-3), (
        "body_link_jacobian_w did not change after manual joint write — "
        "FK trigger likely missing (eval_jacobian / shift kernel reading stale state.body_q)."
    )
    assert not torch.allclose(J_com_0, J_com_1, atol=1e-3), (
        "body_com_jacobian_w did not change after manual joint write — FK trigger likely missing before eval_jacobian."
    )


@pytest.mark.parametrize("num_articulations", [1])
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("articulation_type", ["panda", "anymal"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_mass_matrix_refreshes_after_manual_joint_write(
    sim, num_articulations, device, articulation_type, gravity_enabled
):
    """After ``write_joint_position_to_sim_index`` (no sim step), the mass matrix read
    must reflect the new joint state.

    The mass matrix depends on ``q`` (joint positions) through the body-spatial-inertia
    transformation in eval_mass_matrix's ``compute_body_spatial_inertia`` step, which
    reads ``state.body_q``. Same FK-staleness pattern as the Jacobian.
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
        "FK trigger likely missing before eval_mass_matrix (compute_body_spatial_inertia "
        "reads stale state.body_q)."
    )


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_franka_ik_tracking_accuracy(sim, device, articulation_type, gravity_enabled):
    """Newton-side IK convergence sentinel.

    Runs a full IK tracking loop end-to-end through the new
    ``robot.data.body_link_jacobian_w`` accessor and records the steady-state EE
    pose error. With the robot teleported to its configured init_state
    home pose and scene gravity off, Newton's IK converges to
    machine-precision tracking (sub-mm). A bridge regression
    (wrong-reference-frame Jacobian, missing COM->origin shift, DoF
    mis-ordering) would push the steady-state error well above the
    threshold below.

    The pose teleport is deliberate: the standalone test path does not
    invoke a manager-based env reset (which is what normally pushes
    :attr:`~isaaclab.assets.ArticulationData.default_joint_pos` to sim).
    Without it, the robot starts at the URDF-neutral pose where the
    Franka wrist axes nearly align (rank-deficient Jacobian) and DLS
    plateaus at multi-cm error -- a kinematic-singularity artifact, not
    a bridge or Newton issue.

    See ``test_get_jacobians_link_origin_contract`` (above) for the
    sharper unit-level pin on the Jacobian's reference-point contract.
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

    # Print metrics every run for stress-test capture.
    print(f"IK_METRIC pos_min={pos_min:.5f} pos_mean={pos_mean:.5f} rot_min={rot_min:.5f} rot_mean={rot_mean:.5f}")

    # Regression sentinel: assert on tail mean rather than min. Tail
    # min is the bottom of any oscillation envelope and can be tiny
    # while the actual tracking error is much larger. With the
    # configured home pose and scene gravity off, Newton converges to
    # machine precision (sub-mm). The 5 mm bound absorbs any CUDA-
    # kernel-ordering noise while remaining well below the "totally
    # broken" regime: a bridge regression (wrong-frame Jacobian,
    # missing COM->origin shift, DoF mis-ordering) would push the
    # steady-state error well past this bound.
    assert pos_mean < 5e-3, f"IK pos_mean {pos_mean:.5f} > 5 mm — bridge regression?"
    assert rot_mean < 5e-2, f"IK rot_mean {rot_mean:.5f} > 0.05 rad — bridge regression?"


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("articulation_type", ["panda"])
@pytest.mark.parametrize("gravity_enabled", [False])
@pytest.mark.isaacsim_ci
def test_franka_osc_tracking_accuracy(sim, device, articulation_type, gravity_enabled):
    """Newton-side OSC pose tracking sentinel.

    Mirror of the existing PhysX-side OSC tests in
    :mod:`isaaclab.test.controllers.test_operational_space`, scoped to
    Franka pose-abs tracking on Newton. Like the IK sentinel above, this
    test exercises the full controller-bridge pipeline
    (:attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w` +
    :attr:`~isaaclab.assets.BaseArticulationData.mass_matrix`) end-to-end
    and asserts a loose regression bound rather than a tight correctness
    oracle.

    Newton lacks a gravity-comp primitive (see ``xfail`` test below;
    upstream Newton issues #2497, #2529, #2625), so OSC runs with
    ``gravity_compensation=False`` and the test isolates from gravity by
    disabling scene gravity. ``inertial_dynamics_decoupling=True``
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

    pos_min, pos_mean = _summarize_history(pos_history)
    rot_min, rot_mean = _summarize_history(rot_history)

    print(f"OSC_METRIC pos_min={pos_min:.5f} pos_mean={pos_mean:.5f} rot_min={rot_min:.5f} rot_mean={rot_mean:.5f}")

    # Regression sentinel: assert on tail mean rather than min. With
    # ``current_ee_vel_b = J · q_dot`` providing OSC's damping term and
    # the actuator PD zeroed, the impedance settles to machine
    # precision -- same ballpark as the IK test. The 5 mm bound is a
    # bridge regression sentinel: a wrong J, wrong mass matrix, or
    # DoF mis-ordering pushes the steady-state error well past it
    # because OSC consumes both ``body_link_jacobian_w`` and
    # ``mass_matrix`` per step.
    assert pos_mean < 5e-3, f"OSC pos_mean {pos_mean:.5f} > 5 mm — bridge regression?"
    assert rot_mean < 5e-2, f"OSC rot_mean {rot_mean:.5f} > 0.05 rad — bridge regression?"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
