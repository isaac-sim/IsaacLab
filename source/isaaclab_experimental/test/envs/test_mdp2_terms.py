# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity of the built-in MDP2 terms with their MDP1 functions, on the same Newton states."""

import math
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import warp as wp
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2Env, MDP2EnvCfg
from isaaclab_experimental.mdp2 import mdp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.envs.mdp as mdp1
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")

ROBOT_USD = Path(__file__).parents[3] / "isaaclab_newton/test/assets/data/floating_two_leg.usda"
NUM_ENVS = 8
LEG_NAMES = ("LF_.*", "RH_KFE")

robot = mdp2.Entity("robot")
scene = mdp2.Entity("scene")
legs = robot.joints(*LEG_NAMES)


@pytest.fixture(scope="module")
def env() -> Iterator[MDP2Env]:
    """Two-legged floating robots in random joint and root states."""
    scene_cfg = InteractiveSceneCfg(num_envs=NUM_ENVS, env_spacing=3.0)
    scene_cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ROBOT_USD)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), joint_pos={".*HFE": 0.4, ".*KFE": -0.8}),
        actuators={"legs": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )

    task = mdp2.TaskCfg(actions={"joint_effort": mdp2.DirectAction(robot.joint_effort_target)})
    cfg = MDP2EnvCfg(
        sim=SimulationCfg(dt=0.01, device="cuda:0", physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())),
        scene=scene_cfg,
        task=task,
        decimation=1,
        seed=0,
    )

    env = MDP2Env(cfg)
    try:
        articulation = env.scene["robot"]
        generator = torch.Generator(device=env.device).manual_seed(0)

        def rand(*shape, scale=1.0):
            return scale * (2.0 * torch.rand(*shape, device=env.device, generator=generator) - 1.0)

        joint_pos = articulation.data.default_joint_pos.torch + rand(NUM_ENVS, 6, scale=1.5)
        articulation.write_joint_state_to_sim_index(position=joint_pos, velocity=rand(NUM_ENVS, 6, scale=2.0))
        position = env.scene.env_origins + torch.tensor([0.0, 0.0, 0.5], device=env.device) + rand(NUM_ENVS, 3)
        quat = torch.nn.functional.normalize(rand(NUM_ENVS, 4), dim=-1)
        articulation.write_root_pose_to_sim_index(root_pose=torch.cat([position, quat], -1))
        articulation.write_root_velocity_to_sim_index(root_velocity=rand(NUM_ENVS, 6, scale=3.0))
        env.sim.forward()
        env.scene.update(dt=0.0)

        yield env
    finally:
        env.close()


class MDP1Env:
    """What the MDP1 functions read from an environment: the scene, and joint positions as the command."""

    def __init__(self, env: MDP2Env):
        self.scene = env.scene
        self.command_manager = self

    def get_command(self, name: str) -> torch.Tensor:
        return self.scene["robot"].data.joint_pos.torch


def robot_cfg(env: MDP2Env, joint_names: tuple[str, ...] | None = None) -> SceneEntityCfg:
    """Return the resolved MDP1 configuration of the robot, or of the named joints of the robot."""
    cfg = SceneEntityCfg("robot", joint_names=list(joint_names)) if joint_names else SceneEntityCfg("robot")
    cfg.resolve(env.scene)
    return cfg


def mdp2_values(env: MDP2Env, backend: str, uses: dict[str, mdp2.Observation | mdp2.Termination]) -> dict:
    """Evaluate term uses once on the environment's scene data, each in its own observation group or termination."""
    builder = mdp2.MDPBuilder(env.scene_data.schema)
    for name, use in uses.items():
        if isinstance(use, mdp2.Termination):
            builder.add_termination(name, use)
        else:
            builder.add_observation(name, name, use)
    model = builder.finalize(device=env.device, backend=backend, diagnostics=True)

    state, transition = model.state(env.scene_data), model.transition()
    mdp2.evaluate(model, state, transition)

    values = {group: buffer.torch.clone() for group, buffer in transition.observations.items()}
    for op in model.ops[mdp2.Phase.TERMINATIONS]:
        values[op.name.split(".")[-1]] = transition.termination_terms.torch[:, op.index].clone()
    return values


@pytest.mark.parametrize("backend", ["torch", "warp"])
def test_observations_and_rewards_match_mdp1(env: MDP2Env, backend: str):
    joints = {"joint_ids": legs}
    root_pose = {"root_link_pose_w": robot.root_link_pose_w}
    root_state = {**root_pose, "root_com_vel_w": robot.root_com_vel_w}
    gravity = {**root_pose, "gravity_w": scene.gravity_w}
    joint_pos = {"joint_pos": robot.joint_pos, **joints}
    joint_vel = {"joint_vel": robot.joint_vel, **joints}
    uses = {
        "joint_pos": mdp2.Observation(mdp.joint_pos, inputs=joint_pos),
        "joint_pos_rel": mdp2.Observation(
            mdp.joint_pos_rel, inputs={**joint_pos, "default_joint_pos": robot.default_joint_pos}
        ),
        "joint_vel": mdp2.Observation(mdp.joint_vel, inputs=joint_vel),
        "joint_vel_rel": mdp2.Observation(
            mdp.joint_vel_rel, inputs={**joint_vel, "default_joint_vel": robot.default_joint_vel}
        ),
        "joint_pos_limit_normalized": mdp2.Observation(
            mdp.joint_pos_limit_normalized, inputs={**joint_pos, "soft_joint_pos_limits": robot.soft_joint_pos_limits}
        ),
        "root_pos_w": mdp2.Observation(mdp.root_pos_w, inputs={**root_pose, "env_origins": scene.env_origins}),
        "root_quat_w": mdp2.Observation(mdp.root_quat_w, inputs=root_pose, params={"make_quat_unique": True}),
        "base_lin_vel": mdp2.Observation(mdp.base_lin_vel, inputs=root_state),
        "base_ang_vel": mdp2.Observation(mdp.base_ang_vel, inputs=root_state),
        "projected_gravity": mdp2.Observation(mdp.projected_gravity, inputs=gravity),
        "lin_vel_z_l2": mdp2.Observation(mdp.lin_vel_z_l2, inputs=root_state),
        "ang_vel_xy_l2": mdp2.Observation(mdp.ang_vel_xy_l2, inputs=root_state),
        "flat_orientation_l2": mdp2.Observation(mdp.flat_orientation_l2, inputs=gravity),
        "joint_vel_l1": mdp2.Observation(mdp.joint_vel_l1, inputs=joint_vel),
        "joint_vel_l2": mdp2.Observation(mdp.joint_vel_l2, inputs=joint_vel),
        "joint_deviation_l1": mdp2.Observation(
            mdp.joint_deviation_l1, inputs={**joint_pos, "default_joint_pos": robot.default_joint_pos}
        ),
        "joint_pos_target_l2": mdp2.Observation(mdp.joint_pos_target_l2, inputs=joint_pos, params={"target": 0.3}),
        "joint_pos_limits": mdp2.Observation(
            mdp.joint_pos_limits, inputs={**joint_pos, "soft_joint_pos_limits": robot.soft_joint_pos_limits}
        ),
        "joint_torques_l2": mdp2.Observation(
            mdp.joint_torques_l2, inputs={"applied_effort": robot.applied_effort, **joints}
        ),
        "joint_acc_l2": mdp2.Observation(mdp.joint_acc_l2, inputs={"joint_acc": robot.joint_acc, **joints}),
        "base_height_l2": mdp2.Observation(mdp.base_height_l2, inputs=root_pose, params={"target_height": 0.4}),
        # Joint positions stand in for a velocity command.
        "track_lin_vel_xy_exp": mdp2.Observation(
            mdp.track_lin_vel_xy_exp, inputs={**root_state, "command": robot.joint_pos}, params={"std": 0.5}
        ),
        "track_ang_vel_z_exp": mdp2.Observation(
            mdp.track_ang_vel_z_exp, inputs={**root_state, "command": robot.joint_pos}, params={"std": 0.5}
        ),
    }

    values = mdp2_values(env, backend, uses)

    mdp1_env = MDP1Env(env)
    leg_cfg, root_cfg = robot_cfg(env, LEG_NAMES), robot_cfg(env)
    expected = {
        "joint_pos": mdp1.joint_pos(mdp1_env, leg_cfg),
        "joint_pos_rel": mdp1.joint_pos_rel(mdp1_env, leg_cfg),
        "joint_vel": mdp1.joint_vel(mdp1_env, leg_cfg),
        "joint_vel_rel": mdp1.joint_vel_rel(mdp1_env, leg_cfg),
        "joint_pos_limit_normalized": mdp1.joint_pos_limit_normalized(mdp1_env, leg_cfg),
        "root_pos_w": mdp1.root_pos_w(mdp1_env, root_cfg),
        "root_quat_w": mdp1.root_quat_w(mdp1_env, make_quat_unique=True, asset_cfg=root_cfg),
        "base_lin_vel": mdp1.base_lin_vel(mdp1_env, root_cfg),
        "base_ang_vel": mdp1.base_ang_vel(mdp1_env, root_cfg),
        "projected_gravity": mdp1.projected_gravity(mdp1_env, root_cfg),
        "lin_vel_z_l2": mdp1.lin_vel_z_l2(mdp1_env, root_cfg),
        "ang_vel_xy_l2": mdp1.ang_vel_xy_l2(mdp1_env, root_cfg),
        "flat_orientation_l2": mdp1.flat_orientation_l2(mdp1_env, root_cfg),
        "joint_vel_l1": mdp1.joint_vel_l1(mdp1_env, leg_cfg),
        "joint_vel_l2": mdp1.joint_vel_l2(mdp1_env, leg_cfg),
        "joint_deviation_l1": mdp1.joint_deviation_l1(mdp1_env, leg_cfg),
        "joint_pos_target_l2": mdp1.joint_pos_target_l2(mdp1_env, target=0.3, asset_cfg=leg_cfg),
        "joint_pos_limits": mdp1.joint_pos_limits(mdp1_env, leg_cfg),
        "joint_torques_l2": mdp1.joint_torques_l2(mdp1_env, leg_cfg),
        "joint_acc_l2": mdp1.joint_acc_l2(mdp1_env, leg_cfg),
        "base_height_l2": mdp1.base_height_l2(mdp1_env, 0.4, root_cfg),
        "track_lin_vel_xy_exp": mdp1.track_lin_vel_xy_exp(mdp1_env, 0.5, "velocity", root_cfg),
        "track_ang_vel_z_exp": mdp1.track_ang_vel_z_exp(mdp1_env, 0.5, "velocity", root_cfg),
    }

    assert values["joint_pos_limits"].abs().sum() > 0.0  # Some joints are beyond their limits.
    for name, value in expected.items():
        torch.testing.assert_close(values[name], value.reshape(NUM_ENVS, -1), msg=name)


@pytest.mark.parametrize("backend", ["torch", "warp"])
def test_terminations_match_mdp1(env: MDP2Env, backend: str):
    root_pose = {"root_link_pose_w": robot.root_link_pose_w}
    uses = {
        "joint_pos_out_of_manual_limit": mdp2.Termination(
            mdp.joint_pos_out_of_manual_limit,
            inputs={"joint_pos": robot.joint_pos, "joint_ids": legs},
            params={"bounds": (-1.5, 1.5)},
        ),
        "root_height_below_minimum": mdp2.Termination(
            mdp.root_height_below_minimum, inputs=root_pose, params={"minimum_height": 0.5}
        ),
        "bad_orientation": mdp2.Termination(
            mdp.bad_orientation, inputs={**root_pose, "gravity_w": scene.gravity_w}, params={"limit_angle": 1.0}
        ),
    }

    values = mdp2_values(env, backend, uses)

    mdp1_env = MDP1Env(env)
    leg_cfg, root_cfg = robot_cfg(env, LEG_NAMES), robot_cfg(env)
    expected = {
        "joint_pos_out_of_manual_limit": mdp1.joint_pos_out_of_manual_limit(mdp1_env, (-1.5, 1.5), leg_cfg),
        "root_height_below_minimum": mdp1.root_height_below_minimum(mdp1_env, 0.5, root_cfg),
        "bad_orientation": mdp1.bad_orientation(mdp1_env, 1.0, root_cfg),
    }

    for name, value in expected.items():
        assert value.any() and not value.all(), name  # The states cover both outcomes.
        assert torch.equal(values[name], value), name


def test_events_agree_across_backends_and_respect_ranges(env: MDP2Env):
    """Events draw the same samples on both backends, write only the selected worlds, and stay in their ranges."""
    pose_range = ((-0.2, 0.2), (-0.2, 0.2), (0.0, 0.1), (-0.3, 0.3), (-0.3, 0.3), (-math.pi, math.pi))
    velocity_range = ((-0.5, 0.5),) * 6
    events = {
        "reset_base": mdp2.Event(
            mdp.reset_root_state_uniform,
            inputs={
                "default_root_pose": robot.default_root_pose,
                "default_root_vel": robot.default_root_vel,
                "env_origins": scene.env_origins,
            },
            outputs={"root_link_pose_w": robot.root_link_pose_w, "root_com_vel_w": robot.root_com_vel_w},
            params={"pose_range": pose_range, "velocity_range": velocity_range},
        ),
        "push": mdp2.Event(
            mdp.push_by_setting_velocity,
            outputs={"root_com_vel_w": robot.root_com_vel_w},
            params={"velocity_range": velocity_range},
        ),
        "reset_legs": mdp2.Event(
            mdp.reset_joints_by_offset,
            inputs={
                "default_joint_pos": robot.default_joint_pos,
                "default_joint_vel": robot.default_joint_vel,
                "soft_joint_pos_limits": robot.soft_joint_pos_limits,
                "soft_joint_vel_limits": robot.soft_joint_vel_limits,
                "joint_ids": legs,
            },
            outputs={"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel},
            params={"position_range": (-0.1, 0.1), "velocity_range": (-0.2, 0.2)},
        ),
    }
    mask = torch.arange(NUM_ENVS, device=env.device) % 2 == 0

    results = {}
    for backend in ("torch", "warp"):
        builder = mdp2.MDPBuilder(env.scene_data.schema)
        for name, event in events.items():
            builder.add_event(name, event)
        model = builder.finalize(device=env.device, backend=backend)
        state = model.state(env.scene_data, seed=3)

        before = {name: env.scene_data.data("robot", name).torch.clone() for name in ("root_link_pose_w", "joint_pos")}
        mdp2.reset(model, state, ProxyArray(wp.from_torch(mask)))
        results[backend] = {
            name: env.scene_data.data("robot", name).torch.clone()
            for name in ("root_link_pose_w", "root_com_vel_w", "joint_pos", "joint_vel")
        }

        for name, value in before.items():
            assert torch.equal(results[backend][name][~mask], value[~mask]), name
            env.scene_data.data("robot", name).torch.copy_(value)

    for name, value in results["warp"].items():
        torch.testing.assert_close(value, results["torch"][name], msg=name)

    data = env.scene["robot"].data
    offset = (
        results["warp"]["root_link_pose_w"][mask, :3]
        - env.scene.env_origins[mask]
        - data.default_root_pose.torch[mask, :3]
    )
    assert (offset[:, :2].abs() <= 0.2).all() and (offset[:, 2] >= 0.0).all() and (offset[:, 2] <= 0.1).all()
    torch.testing.assert_close(
        results["warp"]["root_link_pose_w"][mask, 3:].norm(dim=-1), torch.ones(NUM_ENVS // 2, device=env.device)
    )

    # The push adds to the reset velocity, so each component moves at most twice the range.
    assert ((results["warp"]["root_com_vel_w"][mask] - data.default_root_vel.torch[mask]).abs() <= 1.0).all()

    leg_ids = env.scene["robot"].find_joints(list(LEG_NAMES), preserve_order=True)[0]
    joint_offset = results["warp"]["joint_pos"][mask] - data.default_joint_pos.torch[mask]
    assert (joint_offset[:, leg_ids].abs() <= 0.1 + 1e-6).all()


def test_heading_command_steers_towards_its_heading(env: MDP2Env):
    """Both backends draw the same commands; heading worlds turn towards their heading as MDP1's command does."""
    command = mdp2.Command(
        mdp.heading_velocity_command,
        inputs={"root_link_pose_w": robot.root_link_pose_w},
        params={
            "lin_vel_x": (-1.0, 1.0),
            "lin_vel_y": (-0.5, 0.5),
            "ang_vel_z": (-1.0, 1.0),
            "heading": (-math.pi, math.pi),
            "rel_standing_envs": 0.25,
            "rel_heading_envs": 0.5,
            "heading_control_stiffness": 0.5,
        },
    )

    results = {}
    for backend in ("torch", "warp"):
        builder = mdp2.MDPBuilder(env.scene_data.schema)
        builder.add_command("base_velocity", command)
        model = builder.finalize(device=env.device, backend=backend)
        state, transition = model.state(env.scene_data, seed=1), model.transition()

        mdp2.reset(model, state, ProxyArray(wp.ones(NUM_ENVS, dtype=wp.bool, device=env.device)))
        sampled = state.fields[mdp2.commands.base_velocity].torch.clone()
        mdp2.evaluate(model, state, transition)
        updated = state.fields[mdp2.commands.base_velocity].torch.clone()
        heading = state.term_state["commands.base_velocity"]["heading_target"].torch.clone()
        results[backend] = (sampled, updated, heading)

    for value, expected in zip(results["warp"], results["torch"]):
        torch.testing.assert_close(value, expected, equal_nan=True)

    sampled, updated, heading = results["warp"]
    standing = (sampled == 0.0).all(-1)
    controlled = ~heading.isnan()
    assert standing.any() and controlled.any() and (~standing & ~controlled).any()
    assert (sampled.abs() <= torch.tensor([1.0, 0.5, 1.0], device=env.device)).all()

    heading_error = math_utils.wrap_to_pi(heading - env.scene["robot"].data.heading_w.torch)
    expected = torch.where(controlled, (0.5 * heading_error).clip(-1.0, 1.0), sampled[:, 2])
    torch.testing.assert_close(updated[:, 2], expected)
    assert heading[standing].isnan().all() and (updated[standing] == 0.0).all()


def test_startup_events_randomize_the_model_alike_on_both_backends(env: MDP2Env):
    """Startup events scale the base mass and offset its center of mass by the same samples on both backends."""
    base = robot.bodies("base")
    events = {
        "base_mass": mdp2.Event(
            mdp.randomize_rigid_body_mass,
            mode=mdp2.EventMode.STARTUP,
            inputs={"body_ids": base},
            params={
                "mass_distribution_params": (0.8, 1.25),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        ),
        "base_com": mdp2.Event(
            mdp.randomize_rigid_body_com,
            mode=mdp2.EventMode.STARTUP,
            inputs={"body_ids": base},
            params={"com_range": ((-0.05, 0.05), (-0.05, 0.05), (-0.01, 0.01))},
        ),
    }
    data = env.scene["robot"].data
    default_mass = data.body_mass.torch.clone()
    default_com = data.body_com_pos_b.torch.clone()
    base_id = env.scene["robot"].find_bodies("base")[0][0]

    results = {}
    for backend in ("torch", "warp"):
        builder = mdp2.MDPBuilder(env.scene_data.schema)
        for name, event in events.items():
            builder.add_event(name, event)
        model = builder.finalize(device=env.device, backend=backend)
        assert model.is_fully_capturable  # Eager startup events run once, so they do not prevent capture.

        mdp2.startup(model, model.state(env.scene_data, seed=2))
        results[backend] = (data.body_mass.torch.clone(), data.body_com_pos_b.torch.clone())
        env.scene["robot"].set_masses_index(masses=default_mass)
        env.scene["robot"].set_coms_index(coms=default_com)

    for value, expected in zip(results["warp"], results["torch"]):
        torch.testing.assert_close(value, expected)

    mass, com = results["warp"]
    ratio = mass[:, base_id] / default_mass[:, base_id]
    assert ((ratio >= 0.8 - 1e-6) & (ratio <= 1.25 + 1e-6)).all() and ratio.std() > 0.0
    assert torch.equal(mass[:, base_id + 1 :], default_mass[:, base_id + 1 :])
    offset = com[:, base_id] - default_com[:, base_id]
    assert (offset[:, :2].abs() <= 0.05 + 1e-6).all() and (offset[:, 2].abs() <= 0.01 + 1e-6).all()
