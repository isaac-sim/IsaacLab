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
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.warp import ProxyArray

import isaaclab_tasks.core.velocity.mdp as velocity_mdp1

pytestmark = [pytest.mark.kitless, pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")]

ROBOT_USD = Path(__file__).parents[3] / "isaaclab_newton/test/assets/data/floating_two_leg.usda"
NUM_ENVS = 8
LEG_NAMES = ("LF_.*", "RH_KFE")

robot = mdp2.Entity("robot")
scene = mdp2.Entity("scene")
joint_wrench = mdp2.Entity("joint_wrench")
legs = robot.joints(*LEG_NAMES)

POSE_COMMAND = (0.2, -0.1, 0.3, 0.0, 0.0, 0.6, 0.8)
"""A pose command in the root frame: a position and a unit quaternion (x, y, z, w)."""

pose_tracking = {
    "command": mdp2.commands.pose,
    "root_link_pose_w": robot.root_link_pose_w,
    "body_link_pose_w": robot.body_link_pose_w,
    "body_ids": robot.bodies("LF_SHANK"),
}


@wp.func
def _fixed_pose_command(world: int, command: wp.array2d(dtype=float)):
    command[world, 0] = 0.2
    command[world, 1] = -0.1
    command[world, 2] = 0.3
    command[world, 3] = 0.0
    command[world, 4] = 0.0
    command[world, 5] = 0.6
    command[world, 6] = 0.8


fixed_pose_command = mdp2.Term(
    torch=lambda mask, command: command.copy_(command.new_tensor(POSE_COMMAND).expand_as(command)),
    warp=_fixed_pose_command,
    shape=("worlds", 7),
)
"""The pose command :data:`POSE_COMMAND` for every world."""


@pytest.fixture(scope="module")
def env() -> Iterator[MDP2Env]:
    """Two-legged floating robots in random joint and root states, with random joint efforts and a joint wrench
    sensor."""
    scene_cfg = InteractiveSceneCfg(num_envs=NUM_ENVS, env_spacing=3.0)
    scene_cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ROBOT_USD)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), joint_pos={".*HFE": 0.4, ".*KFE": -0.8}),
        actuators={"legs": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )
    scene_cfg.joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")

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
        articulation.set_joint_effort_target_index(target=rand(NUM_ENVS, 6, scale=5.0))
        articulation.write_data_to_sim()
        # One physics step, so that the joint wrench sensor reports the wrenches that the joints transmit.
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)

        yield env
    finally:
        env.close()


class MDP1Env:
    """What the MDP1 functions and commands read from an environment: the scene, :data:`POSE_COMMAND` as the
    ``pose`` command, and joint positions as any other command."""

    def __init__(self, env: MDP2Env):
        self.scene = env.scene
        self.sim = env.sim
        self.num_envs = env.num_envs
        self.device = env.device
        self.command_manager = self
        self.terms = {}

    def get_command(self, name: str) -> torch.Tensor:
        if name == "pose":
            return torch.tensor(POSE_COMMAND, device=self.device).expand(self.num_envs, -1)
        return self.scene["robot"].data.joint_pos.torch

    def get_term(self, name: str):
        return self.terms[name]


def robot_cfg(env: MDP2Env, joint_names: tuple[str, ...] | None = None) -> SceneEntityCfg:
    """Return the resolved MDP1 configuration of the robot, or of the named joints of the robot."""
    cfg = SceneEntityCfg("robot", joint_names=list(joint_names)) if joint_names else SceneEntityCfg("robot")
    cfg.resolve(env.scene)
    return cfg


def bodies_cfg(env: MDP2Env, name: str, body_names: list[str]) -> SceneEntityCfg:
    """Return the resolved MDP1 configuration of the named bodies of a scene entity."""
    cfg = SceneEntityCfg(name, body_names=body_names)
    cfg.resolve(env.scene)
    return cfg


def mdp2_values(
    env: MDP2Env, backend: str, uses: dict[str, mdp2.Command | mdp2.Quantity | mdp2.Observation | mdp2.Termination]
) -> dict:
    """Evaluate term uses once on the environment's scene data, each in its own observation group or termination.

    The commands are sampled for every world first.
    """
    builder = mdp2.MDPBuilder(env.scene_data.schema)
    for name, use in uses.items():
        if isinstance(use, mdp2.Command):
            builder.add_command(name, use)
        elif isinstance(use, mdp2.Quantity):
            builder.add_quantity(name, use)
        elif isinstance(use, mdp2.Termination):
            builder.add_termination(name, use)
        else:
            builder.add_observation(name, name, use)
    model = builder.finalize(device=env.device, backend=backend, diagnostics=True)

    state, transition = model.state(env.scene_data), model.transition()
    mdp2.reset(model, state, ProxyArray(wp.ones(NUM_ENVS, dtype=wp.bool, device=env.device)))
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
        "track_lin_vel_xy_yaw_frame_exp": mdp2.Observation(
            mdp.track_lin_vel_xy_yaw_frame_exp, inputs={**root_state, "command": robot.joint_pos}, params={"std": 0.5}
        ),
        "track_ang_vel_z_world_exp": mdp2.Observation(
            mdp.track_ang_vel_z_world_exp,
            inputs={"command": robot.joint_pos, "root_com_vel_w": robot.root_com_vel_w},
            params={"std": 0.5},
        ),
        "pose": mdp2.Command(fixed_pose_command),
        "position_command_error": mdp2.Observation(mdp.position_command_error, inputs=pose_tracking),
        "orientation_command_error": mdp2.Observation(mdp.orientation_command_error, inputs=pose_tracking),
        "body_incoming_wrench": mdp2.Observation(
            mdp.body_incoming_wrench,
            inputs={
                "force": joint_wrench.force,
                "torque": joint_wrench.torque,
                "body_ids": joint_wrench.bodies(".*_SHANK"),
            },
        ),
    }

    values = mdp2_values(env, backend, uses)

    mdp1_env = MDP1Env(env)
    leg_cfg, root_cfg = robot_cfg(env, LEG_NAMES), robot_cfg(env)
    shank_cfg = bodies_cfg(env, "robot", ["LF_SHANK"])
    wrench_cfg = bodies_cfg(env, "joint_wrench", [".*_SHANK"])
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
        "track_lin_vel_xy_yaw_frame_exp": velocity_mdp1.track_lin_vel_xy_yaw_frame_exp(
            mdp1_env, 0.5, "velocity", root_cfg
        ),
        "track_ang_vel_z_world_exp": velocity_mdp1.track_ang_vel_z_world_exp(mdp1_env, "velocity", 0.5, root_cfg),
        "position_command_error": mdp1.position_command_error(mdp1_env, "pose", shank_cfg),
        "orientation_command_error": mdp1.orientation_command_error(mdp1_env, "pose", shank_cfg),
        "body_incoming_wrench": mdp1.body_incoming_wrench(mdp1_env, wrench_cfg),
    }

    assert values["joint_pos_limits"].abs().sum() > 0.0  # Some joints are beyond their limits.
    assert (values["joint_torques_l2"] > 0.0).all()  # The actuators apply efforts.
    assert (values["body_incoming_wrench"] != 0.0).any()  # The joints transmit wrenches.
    for name, value in expected.items():
        torch.testing.assert_close(values[name], value.reshape(NUM_ENVS, -1), msg=name)


@pytest.mark.parametrize("backend", ["torch", "warp"])
def test_terminations_match_mdp1(env: MDP2Env, backend: str):
    mdp1_env = MDP1Env(env)
    leg_cfg, root_cfg = robot_cfg(env, LEG_NAMES), robot_cfg(env)

    # Success thresholds halfway between the middle errors of the worlds, so that each one decides some worlds.
    shank_cfg = bodies_cfg(env, "robot", ["LF_SHANK"])
    thresholds = {}
    for name, error in (("position", mdp1.position_command_error), ("orientation", mdp1.orientation_command_error)):
        errors = error(mdp1_env, "pose", shank_cfg).sort().values
        thresholds[name] = float(errors[NUM_ENVS // 2 - 1 : NUM_ENVS // 2 + 1].mean())
    no_range = (0.0, 0.0)
    pose_command_cfg = mdp1.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="LF_SHANK",
        resampling_time_range=(1.0, 1.0),
        ranges=mdp1.UniformPoseCommandCfg.Ranges(
            pos_x=no_range, pos_y=no_range, pos_z=no_range, roll=no_range, pitch=no_range, yaw=no_range
        ),
        position_success_threshold=thresholds["position"],
        orientation_success_threshold=thresholds["orientation"],
    )
    mdp1_env.terms["pose"] = mdp1.UniformPoseCommand(pose_command_cfg, mdp1_env)
    mdp1_env.terms["pose"].pose_command_b[:] = mdp1_env.get_command("pose")

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
        "pose": mdp2.Command(fixed_pose_command),
        "position_error": mdp2.Quantity(mdp.position_command_error, inputs=pose_tracking),
        "orientation_error": mdp2.Quantity(mdp.orientation_command_error, inputs=pose_tracking),
        "pose_command_success": mdp2.Termination(
            mdp.pose_command_success,
            inputs={
                "position_error": mdp2.quantities.position_error,
                "orientation_error": mdp2.quantities.orientation_error,
            },
            params={
                "position_threshold": thresholds["position"],
                "orientation_threshold": thresholds["orientation"],
            },
        ),
    }

    values = mdp2_values(env, backend, uses)

    expected = {
        "joint_pos_out_of_manual_limit": mdp1.joint_pos_out_of_manual_limit(mdp1_env, (-1.5, 1.5), leg_cfg),
        "root_height_below_minimum": mdp1.root_height_below_minimum(mdp1_env, 0.5, root_cfg),
        "bad_orientation": mdp1.bad_orientation(mdp1_env, 1.0, root_cfg),
        "pose_command_success": mdp1.pose_command_success(mdp1_env, "pose"),
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

        fields = ("root_link_pose_w", "root_com_vel_w", "joint_pos", "joint_vel")
        before = {name: env.scene_data.data("robot", name).torch.clone() for name in fields}
        mdp2.reset(model, state, ProxyArray(wp.from_torch(mask)))
        results[backend] = {name: env.scene_data.data("robot", name).torch.clone() for name in fields}

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
    """Both backends draw the same commands; heading worlds turn towards their heading as MDP1's command does.

    The steered variant gives at its resample what the plain command gives at its first update: MDP1's
    ``UniformVelocityCommand`` steers right after it resamples, since its manager updates the commands after a reset.
    """
    params = {
        "lin_vel_x": (-1.0, 1.0),
        "lin_vel_y": (-0.5, 0.5),
        "ang_vel_z": (-1.0, 1.0),
        "heading": (-math.pi, math.pi),
        "rel_standing_envs": 0.25,
        "rel_heading_envs": 0.5,
        "heading_control_stiffness": 0.5,
    }

    terms = {"plain": mdp.heading_velocity_command, "steered": mdp.heading_velocity_command_steered_on_resample}
    results = {}
    for backend in ("torch", "warp"):
        for variant, term in terms.items():
            command = mdp2.Command(term, inputs={"root_link_pose_w": robot.root_link_pose_w}, params=params)
            builder = mdp2.MDPBuilder(env.scene_data.schema)
            builder.add_command("base_velocity", command)
            model = builder.finalize(device=env.device, backend=backend)
            state, transition = model.state(env.scene_data, seed=1), model.transition()

            mdp2.reset(model, state, ProxyArray(wp.ones(NUM_ENVS, dtype=wp.bool, device=env.device)))
            sampled = state.arrays[mdp2.commands.base_velocity].torch.clone()
            mdp2.evaluate(model, state, transition)
            updated = state.arrays[mdp2.commands.base_velocity].torch.clone()
            heading = state.arrays[mdp2.term_state("commands.base_velocity", "heading_target")].torch.clone()
            results[backend, variant] = (sampled, updated, heading)

    for variant in terms:
        for value, expected in zip(results["warp", variant], results["torch", variant]):
            torch.testing.assert_close(value, expected, equal_nan=True)

    sampled, updated, heading = results["warp", "plain"]
    standing = (sampled == 0.0).all(-1)
    controlled = ~heading.isnan()
    assert standing.any() and controlled.any() and (~standing & ~controlled).any()
    assert (sampled.abs() <= torch.tensor([1.0, 0.5, 1.0], device=env.device)).all()

    heading_error = math_utils.wrap_to_pi(heading - env.scene["robot"].data.heading_w.torch)
    expected = torch.where(controlled, (0.5 * heading_error).clip(-1.0, 1.0), sampled[:, 2])
    torch.testing.assert_close(updated[:, 2], expected)
    assert heading[standing].isnan().all() and (updated[standing] == 0.0).all()

    steered_sampled, steered_updated, steered_heading = results["warp", "steered"]
    torch.testing.assert_close(steered_sampled, updated)
    torch.testing.assert_close(steered_updated, updated)
    torch.testing.assert_close(steered_heading, heading, equal_nan=True)


def test_model_events_randomize_the_startup_model_alike_on_both_backends(env: MDP2Env):
    """Events scale the base mass and offset its center of mass by the same samples on both backends, each time from
    the values at startup, so resets do not compound them."""
    base = robot.bodies("base")
    events = {
        "base_mass": mdp2.Event(
            mdp.randomize_rigid_body_mass,
            inputs={"asset": robot, "body_ids": base},
            params={
                "mass_distribution_params": (0.8, 1.25),
                "operation": "scale",
                "distribution": "log_uniform",
            },
        ),
        "base_com": mdp2.Event(
            mdp.randomize_rigid_body_com,
            inputs={"asset": robot, "body_ids": base},
            params={"com_range": ((-0.05, 0.05), (-0.05, 0.05), (-0.01, 0.01))},
        ),
    }
    data = env.scene["robot"].data
    default_mass = data.body_mass.torch.clone()
    default_inertia = data.body_inertia.torch.clone()
    default_com = data.body_com_pos_b.torch.clone()
    base_id = env.scene["robot"].find_bodies("base")[0][0]

    results = {}
    for backend in ("torch", "warp"):
        builder = mdp2.MDPBuilder(env.scene_data.schema)
        for name, event in events.items():
            builder.add_event(name, event)
        model = builder.finalize(device=env.device, backend=backend)
        state = model.state(env.scene_data, seed=2)

        mdp2.startup(model, state)
        for _ in range(2):
            mdp2.reset(model, state, ProxyArray(wp.ones(NUM_ENVS, dtype=wp.bool, device=env.device)))
        results[backend] = (data.body_mass.torch.clone(), data.body_com_pos_b.torch.clone())
        env.scene["robot"].set_masses_index(masses=default_mass)
        env.scene["robot"].set_inertias_index(inertias=default_inertia)
        env.scene["robot"].set_coms_index(coms=default_com)

    for value, expected in zip(results["warp"], results["torch"]):
        torch.testing.assert_close(value, expected)

    mass, com = results["warp"]
    ratio = mass[:, base_id] / default_mass[:, base_id]
    assert ((ratio >= 0.8 - 1e-6) & (ratio <= 1.25 + 1e-6)).all() and ratio.std() > 0.0
    assert torch.equal(mass[:, base_id + 1 :], default_mass[:, base_id + 1 :])
    offset = com[:, base_id] - default_com[:, base_id]
    assert (offset[:, :2].abs() <= 0.05 + 1e-6).all() and (offset[:, 2].abs() <= 0.01 + 1e-6).all()


def test_model_events_reject_unsupported_randomizations(env: MDP2Env):
    """Like MDP1, model events raise on an operation or distribution that they do not support, at startup."""
    for params in ({"operation": "multiply"}, {"operation": "scale", "distribution": "normal"}):
        builder = mdp2.MDPBuilder(env.scene_data.schema)
        event = mdp2.Event(
            mdp.randomize_rigid_body_mass,
            inputs={"asset": robot, "body_ids": robot.bodies("base")},
            params={"mass_distribution_params": (0.8, 1.25), **params},
        )
        builder.add_event("base_mass", event)
        model = builder.finalize(device=env.device, backend="torch")
        with pytest.raises(ValueError, match="Unsupported"):
            mdp2.startup(model, model.state(env.scene_data))


@pytest.mark.parametrize("backend", ["torch", "warp"])
def test_difficulty_scheduler_starts_at_its_initial_difficulty(env: MDP2Env, backend: str):
    """Difficulties start at ``init_difficulty``, as in MDP1, so failed first episodes demote them from there."""
    builder = mdp2.MDPBuilder(env.scene_data.schema)
    root_state = {"root_link_pose_w": robot.root_link_pose_w, "root_com_vel_w": robot.root_com_vel_w}
    builder.add_quantity("succeeded", mdp2.Quantity(mdp.lin_vel_z_l2, inputs=root_state))
    curriculum = mdp2.Event(
        mdp.difficulty_scheduler,
        inputs={"succeeded": mdp2.quantities.succeeded},
        params={"init_difficulty": 4, "min_difficulty": 2, "max_difficulty": 10, "min_progress": 0.0},
    )
    builder.add_event("curriculum", curriculum)
    model = builder.finalize(device=env.device, backend=backend)
    state = model.state(env.scene_data)

    # The quantity is not computed yet, so no episode succeeded.
    mdp2.startup(model, state)
    mdp2.reset(model, state, ProxyArray(wp.ones(NUM_ENVS, dtype=wp.bool, device=env.device)))
    progress = state.arrays[mdp2.term_state("events.curriculum", "progress")].torch
    torch.testing.assert_close(progress, torch.full_like(progress, 0.3))
