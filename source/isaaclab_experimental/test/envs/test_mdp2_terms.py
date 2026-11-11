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
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")

ROBOT_USD = Path(__file__).parents[3] / "isaaclab_newton/test/assets/data/floating_two_leg.usda"
NUM_ENVS = 8
robot = mdp2.Entity("robot")
scene = mdp2.Entity("scene")
legs = robot.joints("LF_.*", "RH_KFE")


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
    """What the MDP1 functions read from an environment."""

    def __init__(self, env: MDP2Env):
        self.scene = env.scene


def asset_cfg(env: MDP2Env, joint_names: list[str] | None = None) -> SceneEntityCfg:
    cfg = SceneEntityCfg("robot", joint_names=joint_names or [".*"]) if joint_names else SceneEntityCfg("robot")
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
    }
    values = mdp2_values(env, backend, uses)

    mdp1_env = MDP1Env(env)
    leg_cfg = asset_cfg(env, ["LF_.*", "RH_KFE"])
    root_cfg = asset_cfg(env)
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
    }
    assert values["joint_pos_limits"].abs().sum() > 0.0  # some joints are beyond their limits
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
    leg_cfg, root_cfg = asset_cfg(env, ["LF_.*", "RH_KFE"]), asset_cfg(env)
    expected = {
        "joint_pos_out_of_manual_limit": mdp1.joint_pos_out_of_manual_limit(mdp1_env, (-1.5, 1.5), leg_cfg),
        "root_height_below_minimum": mdp1.root_height_below_minimum(mdp1_env, 0.5, root_cfg),
        "bad_orientation": mdp1.bad_orientation(mdp1_env, 1.0, root_cfg),
    }
    for name, value in expected.items():
        assert value.any() and not value.all(), name  # the states cover both outcomes
        assert torch.equal(values[name], value), name


def test_events_agree_across_backends_and_respect_ranges(env: MDP2Env):
    """Events draw the same samples on both backends, write only the selected worlds, and stay in their ranges."""
    pose_range = ((-0.2, 0.2), (-0.2, 0.2), (0.0, 0.1), (-0.3, 0.3), (-0.3, 0.3), (-math.pi, math.pi))
    velocity_range = ((-0.5, 0.5),) * 6
    articulation = env.scene["robot"]
    mask = torch.arange(NUM_ENVS, device=env.device) % 2 == 0

    results = {}
    for backend in ("torch", "warp"):
        builder = mdp2.MDPBuilder(env.scene_data.schema)
        builder.add_event(
            "reset_base",
            mdp2.Event(
                mdp.reset_root_state_uniform,
                inputs={
                    "default_root_pose": robot.default_root_pose,
                    "default_root_vel": robot.default_root_vel,
                    "env_origins": scene.env_origins,
                },
                outputs={"root_link_pose_w": robot.root_link_pose_w, "root_com_vel_w": robot.root_com_vel_w},
                params={"pose_range": pose_range, "velocity_range": velocity_range},
            ),
        )
        builder.add_event(
            "push",
            mdp2.Event(
                mdp.push_by_setting_velocity,
                outputs={"root_com_vel_w": robot.root_com_vel_w},
                params={"velocity_range": velocity_range},
            ),
        )
        builder.add_event(
            "reset_legs",
            mdp2.Event(
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
        )
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

    data = articulation.data
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
    joint_offset = results["warp"]["joint_pos"][mask] - data.default_joint_pos.torch[mask]
    assert (joint_offset[:, legs_ids(env)].abs() <= 0.1 + 1e-6).all()


def legs_ids(env: MDP2Env) -> list[int]:
    return env.scene["robot"].find_joints(["LF_.*", "RH_KFE"], preserve_order=True)[0]
