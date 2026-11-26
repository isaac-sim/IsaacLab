# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MDP2RLEnv on a headless Newton Cartpole, built from a local USD file."""

import importlib.util
import math
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import (
    MDP2Env,
    MDP2EnvCfg,
    MDP2RLEnv,
    MDP2RLEnvCfg,
    ResetStep,
    Stage,
    Step,
    mdp2_env,
    mdp2_scene_data,
)
from isaaclab_experimental.mdp2 import mdp, rng
from isaaclab_experimental.mdp2.config import group_terms
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonOptixRendererCfg, NewtonWarpRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg, RemotizedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext

pytestmark = [pytest.mark.kitless, pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")]

CARTPOLE_USD = Path(__file__).parents[3] / "isaaclab_newton/test/assets/data/fixed_cartpole.usda"
NUM_ENVS = 16


robot = mdp2.Entity("robot")
cart, pole, joints = robot.joints("slider_to_cart"), robot.joints("cart_to_pole"), robot.joints(".*")


def reset_joints(subset: mdp2.SubsetSelector, position_range: tuple, velocity_range: tuple) -> mdp2.Event:
    """Return an event that offsets the subset's joint state from its default by uniform samples."""
    return mdp2.Event(
        mdp.reset_joints_by_offset,
        inputs={
            "default_joint_pos": robot.default_joint_pos,
            "default_joint_vel": robot.default_joint_vel,
            "soft_joint_pos_limits": robot.soft_joint_pos_limits,
            "soft_joint_vel_limits": robot.soft_joint_vel_limits,
            "joint_ids": subset,
        },
        outputs={"joint_pos": robot.joint_pos, "joint_vel": robot.joint_vel},
        params={"position_range": position_range, "velocity_range": velocity_range},
    )


def host_joint_vel_l1(joint_vel, joint_ids):
    joint_vel = joint_vel[:, joint_ids.long()]
    # Reading a value on the host fails inside a CUDA graph, which hooks that are not capturable never run in.
    if joint_vel.isfinite().all().item():
        return joint_vel.abs().sum(dim=-1)
    return torch.zeros_like(joint_vel[:, 0])


def cartpole_cfg(backend: str = "warp", capture: bool = True, host: bool = False) -> MDP2RLEnvCfg:
    """The Cartpole task of ``Isaac-Cartpole``, on the fixed Cartpole of the Newton tests, with a 1 s episode.

    With ``host``, the cart velocity penalty runs on the host, between the CUDA graphs.
    """
    scene = InteractiveSceneCfg(num_envs=NUM_ENVS, env_spacing=4.0)
    scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(CARTPOLE_USD)),
        actuators={
            "cart": ImplicitActuatorCfg(
                joint_names_expr=["slider_to_cart"], joint_effort_limit=400.0, stiffness=0.0, damping=10.0
            ),
            "pole": ImplicitActuatorCfg(
                joint_names_expr=["cart_to_pole"], joint_effort_limit=400.0, stiffness=0.0, damping=0.0
            ),
        },
    )

    terminated = {"terminated": mdp2.terminated}
    task = mdp2.TaskCfg(
        actions={"joint_effort": mdp2.DirectAction(robot.joint_effort_target, subset=cart, scale=100.0)},
        observations={
            "policy": {
                "joint_pos_rel": mdp2.Observation(
                    mdp.joint_pos_rel,
                    inputs={
                        "joint_pos": robot.joint_pos,
                        "default_joint_pos": robot.default_joint_pos,
                        "joint_ids": joints,
                    },
                ),
                "joint_vel_rel": mdp2.Observation(
                    mdp.joint_vel_rel,
                    inputs={
                        "joint_vel": robot.joint_vel,
                        "default_joint_vel": robot.default_joint_vel,
                        "joint_ids": joints,
                    },
                ),
            }
        },
        rewards={
            "alive": mdp2.Reward(mdp.is_alive, inputs=terminated),
            "terminating": mdp2.Reward(mdp.is_terminated, inputs=terminated, weight=-2.0),
            "pole_pos": mdp2.Reward(
                mdp.joint_pos_target_l2,
                inputs={"joint_pos": robot.joint_pos, "joint_ids": pole},
                params={"target": 0.0},
                weight=-1.0,
            ),
            "cart_vel": mdp2.Reward(
                mdp2.Term(torch=host_joint_vel_l1, capturable=False) if host else mdp.joint_vel_l1,
                inputs={"joint_vel": robot.joint_vel, "joint_ids": cart},
                weight=-0.01,
            ),
            "pole_vel": mdp2.Reward(
                mdp.joint_vel_l1, inputs={"joint_vel": robot.joint_vel, "joint_ids": pole}, weight=-0.005
            ),
        },
        terminations={
            "time_out": mdp2.TimeLimit(episode_length_s=1.0),
            "cart_out_of_bounds": mdp2.Termination(
                mdp.joint_pos_out_of_manual_limit,
                inputs={"joint_pos": robot.joint_pos, "joint_ids": cart},
                params={"bounds": (-1.5, 1.5)},
            ),
        },
        events={
            "reset_cart_position": reset_joints(cart, (-1.0, 1.0), (-0.5, 0.5)),
            "reset_pole_position": reset_joints(
                pole, (-0.25 * math.pi, 0.25 * math.pi), (-0.25 * math.pi, 0.25 * math.pi)
            ),
        },
    )

    return MDP2RLEnvCfg(
        sim=SimulationCfg(dt=1 / 120, device="cuda:0", physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())),
        scene=scene,
        task=task,
        decimation=2,
        backend=mdp2.Backend(backend),
        seed=42,
        capture=capture,
    )


def rollout(cfg: MDP2RLEnvCfg, num_steps: int) -> tuple[int, list[dict[str, torch.Tensor]]]:
    """Reset, step random actions, and close; return the graphs that a captured step replays and the step results."""
    env = MDP2RLEnv(cfg)
    try:
        generator = torch.Generator(device=env.device).manual_seed(0)
        # Half the carts are pushed each way, so that they leave their bounds.
        direction = torch.ones(env.num_envs, 1, device=env.device)
        direction[::2] = -1.0

        observations, _ = env.reset()
        history = [{"policy": observations["policy"].clone()}]
        for _ in range(num_steps):
            actions = direction * (torch.rand(env.num_envs, 1, device=env.device, generator=generator) + 0.5)
            observations, rewards, terminated, truncated, _ = env.step(actions)
            history.append(
                {
                    "policy": observations["policy"].clone(),
                    "rewards": rewards.clone(),
                    "terminated": terminated.clone(),
                    "truncated": truncated.clone(),
                }
            )

        num_graphs = env.num_graphs
    finally:
        env.close()
    return num_graphs, history


step_count = mdp2.Term(
    torch=mdp2.Implementation(compute=lambda steps: steps.clone(), update=lambda steps: steps.add_(1.0)),
    state={"steps": mdp2.State()},
)
"""Steps of the episode before the current step, which its update counts after it."""


def shift_root_torch(mask, root_link_pose_w, default_root_pose, env_origins, *, offset):
    pose = default_root_pose.clone()
    pose[:, :3] += env_origins
    pose[:, 0] += offset
    root_link_pose_w.copy_(torch.where(mask[:, None], pose, root_link_pose_w))


def test_env_follows_the_gym_interface_and_commits_resets(monkeypatch):
    """Steps follow the Gym interface, render when rendering is on, return the observations before resets, and the
    solver sees the reset joint state and the moved fixed base."""
    cfg = cartpole_cfg()
    cfg.compute_final_obs = True
    cfg.task.events["shift_root"] = mdp2.Event(
        mdp2.Term(torch=shift_root_torch),
        inputs={"default_root_pose": robot.default_root_pose, "env_origins": mdp2.Entity("scene").env_origins},
        outputs={"root_link_pose_w": robot.root_link_pose_w},
        params={"offset": 0.5},
    )
    # A random seed (-1) reaches the MDP's streams: they start from the seed that was drawn.
    cfg.seed = -1
    monkeypatch.setattr(mdp2_env, "configure_seed", lambda seed: 123)
    env = MDP2RLEnv(cfg)
    try:
        assert torch.equal(env.state.rng.torch, rng.streams(123, env.mdp.streams, NUM_ENVS, env.device))
        robot_data = env.scene["robot"].data
        assert env.single_observation_space["policy"].shape == (4,)
        assert env.single_action_space.shape == (1,)
        assert env.max_episode_length == 60

        observations, extras = env.reset()

        assert observations["policy"].shape == (NUM_ENVS, 4)
        assert (robot_data.joint_pos.torch[:, 0].abs() <= 1.0).all() and robot_data.joint_pos.torch.std(0).min() > 0
        with pytest.raises(ValueError, match=r"Expected actions of shape \(16, 1\), got \(1, 1\)"):
            env.step(torch.ones(1, 1, device=env.device))

        renders = []
        monkeypatch.setattr(SimulationContext, "is_rendering", property(lambda sim: True))
        monkeypatch.setattr(env.sim, "render", lambda: renders.append(env.common_step_counter))
        ended = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.device)
        for _ in range(env.max_episode_length):
            observations, rewards, terminated, truncated, extras = env.step(torch.ones(NUM_ENVS, 1, device=env.device))
            assert rewards.dtype == torch.float32 and terminated.dtype == truncated.dtype == torch.bool
            assert not (terminated & truncated).any()
            # A cart that left its bounds is observed there before the reset, and back within them after it.
            assert (extras["final_obs"]["policy"][terminated, 0].abs() > 1.5).all()
            assert (observations["policy"][terminated, 0].abs() <= 1.0).all()
            assert (env.scene["robot"].actuators.target_command.effort.torch[:, 0] == 100.0).all()
            ended |= terminated | truncated

            # Reset worlds start from their new joint state, which the solver's body poses follow.
            cart_x = robot_data.body_pos_w.torch[:, 1, 0] - robot_data.body_pos_w.torch[:, 0, 0]
            torch.testing.assert_close(cart_x, robot_data.joint_pos.torch[:, 0], atol=1e-5, rtol=0.0)

        assert env.is_captured and ended.all()
        # Every control step renders once, after it, also when it replays as a graph.
        assert renders == list(range(1, env.max_episode_length + 1))
        # The fixed base stays where the reset moved it, through the solver's steps.
        base_x = env.scene.env_origins[:, 0] + robot_data.default_root_pose.torch[:, 0] + 0.5
        torch.testing.assert_close(robot_data.body_pos_w.torch[:, 0, 0], base_x, atol=1e-5, rtol=0.0)
        assert torch.equal(observations["policy"][:, :2], robot_data.joint_pos.torch)

        # A seed also reseeds the MDP's streams, which the reset events draw from; the config is left as it was.
        seeded = [env.reset(seed=7)[0]["policy"].clone() for _ in range(2)]
        assert torch.equal(seeded[0], seeded[1])
        # And on reset, they restart from it.
        env.reset(seed=-1)
        expected = rng.streams(123, env.mdp.streams, NUM_ENVS, env.device)
        rng.advance(expected)  # Once by the reset
        assert torch.equal(env.state.rng.torch, expected)
        assert cfg.scene.lazy_sensor_update
    finally:
        env.close()


def delayed_cart(cfg: MDP2RLEnvCfg, tmp_path: Path | None = None) -> None:
    cfg.scene.robot.actuators["cart"] = DelayedPDActuatorCfg(
        joint_names_expr=["slider_to_cart"], joint_effort_limit=400.0, stiffness=0.0, damping=10.0, max_delay=2
    )


def remotized_cart(cfg: MDP2RLEnvCfg, tmp_path: Path) -> None:
    """A remotized actuator, which runs as an Isaac Lab model without Newton's own actuators."""
    cfg.sim.use_newton_actuators = False
    cfg.scene.robot.actuators["cart"] = RemotizedPDActuatorCfg(
        joint_names_expr=["slider_to_cart"],
        joint_parameter_lookup=[[-2.0, 1.0, 400.0], [2.0, 1.0, 400.0]],
        joint_effort_limit=400.0,
        stiffness=0.0,
        damping=10.0,
    )


def reordered_bodies(cfg: MDP2RLEnvCfg, tmp_path: Path) -> None:
    cfg.scene.robot.body_ordering = ["rail", "pole", "cart"]


def ball_joint_tip(cfg: MDP2RLEnvCfg, tmp_path: Path) -> None:
    """Hang a tip from the pole on a ball joint."""
    tip = """
    def Xform "tip" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 0.1
        float3 physics:diagonalInertia = (0.01, 0.01, 0.01)
    }

    def PhysicsSphericalJoint "pole_to_tip"
    {
        rel physics:body0 = </Robot/pole>
        rel physics:body1 = </Robot/tip>
        point3f physics:localPos0 = (0.0, 0.0, 1.0)
    }
}
"""
    usd = tmp_path / "ball_joint_cartpole.usda"
    usd.write_text(CARTPOLE_USD.read_text().rstrip().removesuffix("}") + tip)
    cfg.scene.robot.spawn.usd_path = str(usd)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (remotized_cart, r"'robot' has the actuator groups \['cart'\]"),
        (reordered_bodies, r"'robot' sets a joint or body ordering"),
    ],
    ids=("actuator-with-state", "body-ordering"),
)
def test_env_rejects_assets_it_cannot_reset_or_write_in_place(change, message, tmp_path):
    """Captured steps replay the actuators, so actuators that are not capturable are refused; and so are body
    orderings, whose poses are copies that the asset refreshes outside the graphs."""
    cfg = cartpole_cfg()
    change(cfg, tmp_path)
    with pytest.raises(ValueError, match=message):
        MDP2RLEnv(cfg)


def test_reset_writes_reach_the_solver_of_ball_joint_articulations(tmp_path):
    """With a ball joint, the asset's joint positions are a copy of the solver's coordinates: a commit passes the
    written ones on through the asset, so the solver starts the episode where the reset event put the cart."""
    cfg = cartpole_cfg()
    ball_joint_tip(cfg, tmp_path)
    cfg.task.events["reset_cart_position"] = reset_joints(cart, (0.7, 0.7), (0.0, 0.0))
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        for _ in range(3):
            env.step(torch.zeros(NUM_ENVS, 1, device=env.device))
        joint_pos = env.scene["robot"].data.joint_pos.torch
        torch.testing.assert_close(
            joint_pos[:, 0], torch.full((NUM_ENVS,), 0.7, device=env.device), atol=1e-2, rtol=0.0
        )
        assert env.is_captured
    finally:
        env.close()


def test_asset_writers_match_writes_to_the_solver_state(monkeypatch):
    """Committing through the assets' writers, as on a backend whose data is not the solver's own, steps exactly like
    Newton's fast path, which writes the solver state in place."""
    _, fast = rollout(cartpole_cfg(), num_steps=70)
    monkeypatch.setattr(mdp2_scene_data, "_newton_backend", lambda sim: None)
    _, writers = rollout(cartpole_cfg(), num_steps=70)
    for fast_step, writers_step in zip(fast, writers):
        for key, value in fast_step.items():
            assert torch.equal(value, writers_step[key]), key


def test_captured_steps_match_eager_steps_and_the_torch_backend():
    """A captured step replays one graph, or graphs around the hooks that run on the host, and gives bitwise the results
    of uncaptured steps."""
    num_steps = 80  # Past the time-out of every world.
    num_graphs, captured_run = rollout(cartpole_cfg(), num_steps)
    eager_graphs, eager_run = rollout(cartpole_cfg(capture=False), num_steps)
    assert num_graphs == 1 and eager_graphs == 0
    assert any(step["terminated"].any() for step in captured_run[1:])

    # The Torch backend with a host hook: the capture split does not depend on the backend.
    torch_graphs, torch_run = rollout(cartpole_cfg(backend="torch", host=True), num_steps)
    _, torch_eager_run = rollout(cartpole_cfg(backend="torch", capture=False, host=True), num_steps)
    # The host reward splits evaluate: physics and the terms before it, then the rest of the step.
    assert torch_graphs == 2
    for captured_step, eager_step, torch_step in zip(captured_run, eager_run, torch_run):
        for key, value in captured_step.items():
            assert torch.equal(value, eager_step[key])
            torch.testing.assert_close(torch_step[key], value, atol=1e-4, rtol=1e-4)
    for captured_step, eager_step in zip(torch_run, torch_eager_run):
        for key, value in captured_step.items():
            assert torch.equal(value, eager_step[key])


@pytest.mark.parametrize("task_worlds", [None, [12, 4]])
def test_env_splits_the_worlds_between_tasks_and_logs_terms_where_they_run(task_worlds):
    cfg = cartpole_cfg()
    # A logged quantity that a reset restarts and the observations of the new episode read again.
    cfg.task.quantities = {"steps": mdp2.Quantity(step_count, record=True)}
    observation = mdp2.Observation(mdp.value, inputs={"value": mdp2.quantities.steps})
    cfg.task.observations["policy"] = {
        **cfg.task.observations["policy"],
        "steps": observation,
    }
    without_alive = replace(cfg.task, rewards={**cfg.task.rewards, "alive": None})
    # A logged quantity of one task only, averaged over its worlds.
    alive = replace(cfg.task, quantities={**cfg.task.quantities, "alive_steps": mdp2.Quantity(step_count, record=True)})
    worlds = None if task_worlds is None else dict(zip(("alive", "still"), task_worlds))
    cfg.task = mdp2.split(alive=alive, still=without_alive, worlds=worlds)
    env = MDP2RLEnv(cfg)
    try:
        # Even shares, or the given number of worlds per task.
        first = 8 if task_worlds is None else task_worlds[0]
        assert [config.worlds for config in env.mdp.world_configs] == [range(0, first), range(first, 16)]

        env.reset()
        for _ in range(env.max_episode_length):
            _, _, _, _, extras = env.step(torch.zeros(NUM_ENVS, 1, device=env.device))

        # Unpushed carts stay in bounds, so every world was alive for its whole episode.
        assert extras["log"]["Episode_Termination/cart_out_of_bounds"] == 0.0
        assert extras["log"]["Episode_Termination/time_out"] == 1.0
        assert extras["log"]["Episode_Reward/alive"].item() == pytest.approx(1.0)
        # The episodes ended at their last step, not at the restart of the next one.
        assert extras["log"]["Metrics/steps"].item() == env.max_episode_length - 1
        assert extras["log"]["Metrics/alive_steps"].item() == pytest.approx(env.max_episode_length - 1)
        assert env.is_captured
    finally:
        env.close()


@wp.func
def zero_gravity_warp(world: int, gravity_w: wp.array(dtype=wp.vec3f)):
    gravity_w[world] = wp.vec3f(0.0, 0.0, 0.0)


def test_startup_and_interval_events_reach_the_solver():
    """A startup event that removes gravity, and an interval event that holds the joints, reach a solver that
    otherwise ignores writes to its state: every step starts from the held joints, without gravity."""
    cfg = cartpole_cfg()
    cfg.sim.physics.solver_cfg.update_data_interval = 0
    cfg.task.events["zero_gravity"] = mdp2.Event(
        mdp2.Term(warp=zero_gravity_warp),
        outputs={"gravity_w": mdp2.Entity("scene").gravity_w},
        mode=mdp2.EventMode.STARTUP,
    )
    hold = reset_joints(joints, (0.1, 0.1), (0.0, 0.0))
    cfg.task.events["hold"] = replace(hold, mode=mdp2.EventMode.INTERVAL, interval_range_s=(0.01, 0.01))
    # The joint speed after the physics of a step, before the interval event.
    cfg.task.quantities["speed"] = mdp2.Quantity(
        mdp.joint_vel_l1, inputs={"joint_vel": robot.joint_vel, "joint_ids": joints}
    )
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        actions = torch.zeros(NUM_ENVS, 1, device=env.device)
        actions[::2] = 1.0
        speeds = []
        for _ in range(10):
            env.step(actions)
            speeds.append(env.state.arrays[mdp2.quantities.speed].torch.clone())

        # After the first step, the joints start each step held at rest: pushed carts reach the same speed every
        # step, and the joints of the others do not move.
        speeds = torch.stack(speeds[1:])
        assert env.is_captured and (speeds[:, ::2] > 0.0).all() and (speeds[:, 1::2] < 1e-5).all()
        torch.testing.assert_close(speeds, speeds[:1].expand_as(speeds))
    finally:
        env.close()


def raise_pole_armature(mask, robot):
    armature = robot.data.joint_armature.torch.clone()
    armature[:, 1] += 0.1
    robot.write_joint_armature_to_sim_mask(armature=armature, env_mask=wp.from_torch(mask))


def raise_pole_armature_stage(env):
    raise_pole_armature(env.transition.done.torch, env.scene["robot"])


@pytest.mark.parametrize(
    ("where", "capturable"),
    [("event", True), ("event", False), ("after-commit", True), ("end-of-step", True)],
    ids=("event-captured", "event-between-graphs", "stage-after-commit", "stage-at-end-of-step"),
)
def test_model_changes_reach_the_solver_of_captured_steps(where, capturable):
    """Model changes that a reset event or a stage makes through the robot reach the solver in every step that
    replays: from inside the graph, from between the graphs, and from stages after the step's commit."""
    cfg = cartpole_cfg()
    cfg.task.terminations["time_out"] = mdp2.TimeLimit(episode_length_s=0.05)
    if where == "event":
        cfg.task.events["raise_pole_armature"] = mdp2.Event(
            mdp2.Term(torch=raise_pole_armature, capturable=capturable), inputs={"robot": robot}
        )
    elif where == "after-commit":
        cfg.reset_step.insert(cfg.reset_step.index(ResetStep.OBSERVE), Stage(raise_pole_armature_stage))
    else:
        cfg.step.append(Stage(raise_pole_armature_stage))
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        for _ in range(10):
            env.step(torch.zeros(NUM_ENVS, 1, device=env.device))

        backend = env.sim.backend_manager.backend
        model = backend.model.joint_armature.numpy().reshape(NUM_ENVS, -1)
        solver = backend.solver.mjw_model.dof_armature.numpy().reshape(NUM_ENVS, -1)
        # Raised in several resets, the last ones by replayed steps.
        assert env.is_captured and env.num_graphs == (1 if capturable else 2) and model[0, 1] > 0.25
        np.testing.assert_allclose(solver, model, rtol=1e-6)
    finally:
        env.close()


def test_stages_run_where_they_are_in_the_step_inside_and_between_graphs():
    """Stages run every step in their place among the built-in work, inside the graph when capturable, and those of
    the reset step also in reset()."""
    clock = torch.zeros(1, device="cuda:0")
    stamps = torch.zeros(8, device="cuda:0")  # When each stage last ran, on the device clock
    counts = torch.zeros(8, device="cuda:0")

    def stamp(k: int) -> Stage:
        def fn(env):
            stamps[k].copy_(clock[0])
            counts[k].add_(1.0)
            clock.add_(1.0)

        return Stage(fn)

    # A stage before and after every built-in step: 0-3 in the step, 4-7 in the reset step.
    cfg = cartpole_cfg()
    cfg.step = [stamp(0), Step.PHYSICS, stamp(1), Step.EVALUATE, stamp(2), Step.RESET, stamp(3)]
    cfg.reset_step = [stamp(4), ResetStep.EVENTS, stamp(5), ResetStep.COMMIT, stamp(6), ResetStep.OBSERVE, stamp(7)]
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        for _ in range(5):
            env.step(torch.zeros(NUM_ENVS, 1, device=env.device))
        assert env.is_captured and env.num_graphs == 1
        assert counts.tolist() == [5.0] * 4 + [6.0] * 4
        # The reset step runs inside the step's Step.RESET.
        order = [0, 1, 2, 4, 5, 6, 7, 3]
        assert stamps[order].tolist() == sorted(stamps.tolist())
    finally:
        env.close()

    # A stage that cannot be captured runs between two graphs.
    counts.zero_()
    cfg = cartpole_cfg()
    cfg.step.insert(cfg.step.index(Step.PHYSICS) + 1, Stage(lambda env: counts[0].add_(1.0), capturable=False))
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        for _ in range(3):
            env.step(torch.zeros(NUM_ENVS, 1, device=env.device))
        assert env.num_graphs == 2 and counts[0].item() == 3
    finally:
        env.close()


def test_built_in_steps_must_run_once_in_order():
    """A step or reset step that reorders, repeats, or leaves out built-in work is refused before anything is built."""
    cfg = cartpole_cfg()
    for step in ([Step.EVALUATE, Step.PHYSICS, Step.RESET], [Step.PHYSICS, Step.EVALUATE, Step.RESET, Step.RESET]):
        cfg.step = step
        with pytest.raises(ValueError, match="MDP2EnvCfg.step must hold"):
            MDP2RLEnv(cfg)

    cfg = cartpole_cfg()
    cfg.reset_step = [ResetStep.EVENTS, ResetStep.OBSERVE, ResetStep.COMMIT]
    with pytest.raises(ValueError, match="MDP2EnvCfg.reset_step must hold"):
        MDP2RLEnv(cfg)


def test_env_refuses_physics_other_than_newton(monkeypatch):
    """MDP2 binds and captures Newton's physics, so a manager of another backend is refused before the scene is
    built, and the simulation context is cleared."""
    monkeypatch.setattr(mdp2_env, "NewtonManager", type("OtherManager", (), {}))
    with pytest.raises(ValueError, match="MDP2 runs on Newton only, got NewtonManager"):
        MDP2RLEnv(cartpole_cfg())
    assert SimulationContext.instance() is None


after_steps = mdp2.Term(torch=lambda episode_length, *, steps: (episode_length >= steps).float())
"""One from the step ``steps`` of the episode on."""


def test_env_runs_the_tasks_of_a_sequence_in_turn_and_logs_their_progress():
    """A sequence of two tasks, each with its own quantity and reward, steps the same captured and uncaptured, and
    the environment logs how often each task was done in the last episodes."""

    def stage(steps: int, weight: float) -> mdp2.TaskCfg:
        ready = mdp2.Quantity(after_steps, inputs={"episode_length": mdp2.episode_length}, params={"steps": steps})
        # The weights differ, so that the bonus is a use of each task rather than one of both.
        bonus = mdp2.Reward(mdp.is_alive, inputs={"terminated": mdp2.terminated}, weight=weight)
        return mdp2.TaskCfg(quantities={"ready": ready}, rewards={"bonus": bonus}, done=mdp2.quantities.ready)

    cfg = cartpole_cfg()
    cfg.task.terminations["cart_out_of_bounds"] = None
    cfg.task = mdp2.compose(cfg.task, mdp2.sequence(warm=stage(3, 1.0), hold=stage(6, 2.0)))
    cfg.verify_capture = 10
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        for _ in range(env.max_episode_length):
            _, _, _, _, extras = env.step(torch.zeros(NUM_ENVS, 1, device=env.device))
        log = extras["log"]
        assert log["Progress/warm"] == 1.0 and log["Progress/hold"] == 1.0 and log["Progress/task"] == 1.0
        # The bonus of each task, per second of the episode: warm ran 3 steps, hold the others.
        steps = env.max_episode_length
        assert log["Episode_Reward/warm.bonus"].item() == pytest.approx(3 / steps)
        assert log["Episode_Reward/hold.bonus"].item() == pytest.approx(2 * (steps - 3) / steps)
    finally:
        env.close()


def test_progress_is_logged_over_the_episodes_that_ran():
    """Under an episode budget of one episode, the progress of each task is logged over that episode, not over every
    world, most of which finished none."""

    def stage(steps: int) -> mdp2.TaskCfg:
        ready = mdp2.Quantity(after_steps, inputs={"episode_length": mdp2.episode_length}, params={"steps": steps})
        return mdp2.TaskCfg(quantities={"ready": ready}, done=mdp2.quantities.ready)

    cfg = cartpole_cfg()
    cfg.task.terminations["cart_out_of_bounds"] = None
    cfg.task = mdp2.compose(cfg.task, mdp2.sequence(warm=stage(3), hold=stage(6)))
    cfg.episode_budget = 1
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        for _ in range(2 * env.max_episode_length):
            _, _, _, _, extras = env.step(torch.zeros(NUM_ENVS, 1, device=env.device))
            if env.episodes_completed:
                break
        assert env.episodes_completed == 1
        log = extras["log"]
        assert log["Progress/warm"] == 1.0 and log["Progress/hold"] == 1.0 and log["Progress/task"] == 1.0
    finally:
        env.close()


def test_verify_capture_finds_python_state_that_the_graph_freezes():
    """The debug check passes a task whose captured steps are exact, and names a stage whose Python-side state the
    graph froze at capture."""
    cfg = cartpole_cfg()
    cfg.verify_capture = 4
    MDP2RLEnv(cfg).close()

    calls = weakref.WeakKeyDictionary()  # Per environment, so that each run of the check counts from zero

    def drifting_reward_weights(env):
        # A Python counter: replayed steps keep the weight it gave at capture.
        calls[env] = calls.get(env, 0) + 1
        env.state.reward_weights.torch.fill_(float(calls[env]))

    cfg.step.insert(cfg.step.index(Step.PHYSICS) + 1, Stage(drifting_reward_weights))
    with pytest.raises(RuntimeError, match=r"Captured step \d+ differs .* at 'rewards'.*stage drifting_reward_weights"):
        MDP2RLEnv(cfg)

    # A reset event with Python-side state: the worlds that reset in captured steps get the value of the capture. Only
    # time-outs end episodes here, after 60 steps, so the check itself must make the worlds reset.
    cfg.task.terminations["cart_out_of_bounds"] = None

    def counted_reset(mask, joint_pos, cache):
        # Counted in the use's cache, which each environment has its own of.
        cache["calls"] = cache.get("calls", 0) + 1
        joint_pos.add_(mask[:, None] * float(cache["calls"]))

    cfg.step = [Step.PHYSICS, Step.EVALUATE, Step.RESET]
    cfg.task.events["counted_reset"] = mdp2.Event(
        mdp2.Term(torch=counted_reset), outputs={"joint_pos": robot.joint_pos}
    )
    with pytest.raises(RuntimeError, match=r"Captured step \d+ differs.*events.counted_reset"):
        MDP2RLEnv(cfg)


def test_joint_accelerations_start_from_the_reset_velocities():
    """The first acceleration of an episode differences the velocities against the ones the reset gave, not against
    those the last episode ended with: resets pass joint velocities through the asset, which keeps them."""
    cfg = cartpole_cfg()
    cfg.decimation = 1  # One physics step per control step, so the acceleration is the step's finite difference
    cfg.task.terminations["time_out"] = mdp2.TimeLimit(episode_length_s=0.05)
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        joint_vel = env.scene_data.data("robot", "joint_vel").torch
        joint_acc = env.scene_data.data("robot", "joint_acc").torch
        previous, reset = joint_vel.clone(), torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.device)
        first_steps = 0
        for _ in range(14):
            _, _, terminated, truncated, _ = env.step(torch.ones(NUM_ENVS, 1, device=env.device))
            started = reset & ~(terminated | truncated)
            expected = (joint_vel - previous) / env.step_dt
            torch.testing.assert_close(joint_acc[started], expected[started], rtol=1e-4, atol=1e-2)
            first_steps += int(started.sum())
            reset, previous = terminated | truncated, joint_vel.clone()
        assert env.is_captured and first_steps > 0
    finally:
        env.close()


def move_reset_carts(env):
    robot_asset = env.scene["robot"]
    joint_pos = robot_asset.data.joint_pos.torch.clone()
    joint_pos[:, 0] = 0.3
    robot_asset.write_joint_position_to_sim_mask(position=joint_pos, env_mask=wp.from_torch(env.transition.done.torch))


cart_offset = mdp2.Term(torch=lambda body_link_pose_w: body_link_pose_w[:, 1, 0] - body_link_pose_w[:, 0, 0])
"""The cart's position along the rail, from the body poses."""


def test_stages_that_write_the_scene_are_seen_by_later_work():
    """A stage after the resets that moves the reset carts through the robot is seen by the observations of the same
    step, which read the body poses: the step recomputes the kinematics after the stage."""
    cfg = cartpole_cfg()
    cfg.task.terminations["time_out"] = mdp2.TimeLimit(episode_length_s=0.05)
    cfg.task.observations["policy"]["cart_offset"] = mdp2.Observation(
        cart_offset, inputs={"body_link_pose_w": robot.body_link_pose_w}
    )
    cfg.reset_step.insert(cfg.reset_step.index(ResetStep.OBSERVE), Stage(move_reset_carts))
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        restarted = 0
        for _ in range(8):
            observations, _, terminated, truncated, _ = env.step(torch.zeros(NUM_ENVS, 1, device=env.device))
            ended = terminated | truncated
            torch.testing.assert_close(
                observations["policy"][ended, -1], torch.full_like(observations["policy"][ended, -1], 0.3)
            )
            restarted += int(ended.sum())
        assert env.is_captured and restarted > 0
    finally:
        env.close()


def test_cameras_render_in_the_captured_step_and_the_budget_records_episodes():
    """Camera images are observed as uint8 terms and rendered by every replayed step, also with two cameras, and an
    episode budget records each episode, then stops the worlds."""
    cfg = cartpole_cfg()
    # The Cartpole has no geometry to render, so the camera watches a cube fall in front of it.
    cfg.scene.cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2), rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(), mass_props=sim_utils.MassCfg(mass=1.0)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.2)),
    )
    cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(pos=(0.0, -1.0, 1.0), rot=(0.0, 0.0, 0.7071, 0.7071), convention="world"),
        data_types=["rgb"],
        width=96,
        height=64,
        spawn=sim_utils.PinholeCameraCfg(focal_length=2.0, horizontal_aperture=2.0, clipping_range=(0.01, 20.0)),
        renderer_cfg=NewtonWarpRendererCfg(),
    )
    # A second camera, watching from the other side; cameras share their renderer.
    cfg.scene.camera_back = cfg.scene.camera.replace(
        prim_path="{ENV_REGEX_NS}/CameraBack",
        offset=CameraCfg.OffsetCfg(pos=(0.0, 1.0, 1.0), rot=(0.0, 0.0, -0.7071, 0.7071), convention="world"),
    )
    cfg.task.observations["cameras"] = mdp2.ObservationGroup(
        terms={
            name: mdp2.Observation(mdp.image_rgb, inputs={"image": mdp2.Entity(camera).rgb})
            for name, camera in (("rgb", "camera"), ("rgb_back", "camera_back"))
        },
        concatenate=False,
    )
    # Indexing with a 1-dimensional tensor, as a 0-dimensional one would be read on the host in the graph.
    cart_x = mdp2.Term(torch=lambda joint_pos, joint_ids: joint_pos[:, joint_ids][:, 0])
    cfg.task.quantities["cart_x"] = mdp2.Quantity(
        cart_x, inputs={"joint_pos": robot.joint_pos, "joint_ids": cart}, record=True
    )
    # Cameras that never render would observe stale images. Cameras that no term reads do not render.
    cfg.step = [Step.PHYSICS, Step.EVALUATE, Step.RESET]
    with pytest.raises(ValueError, match="must hold Step.RENDER"):
        MDP2RLEnv(cfg)
    unread = replace(cfg, task=replace(cfg.task, observations={"policy": cfg.task.observations["policy"]}))
    MDP2RLEnv(unread).close()
    cfg.step = [Step.PHYSICS, Step.RENDER, Step.EVALUATE, Step.RESET]
    budget = NUM_ENVS + 4
    cfg.episode_budget = budget
    env = MDP2RLEnv(cfg)
    try:
        assert env.single_observation_space["cameras"]["rgb"].shape == (64, 96, 3)
        observations, _ = env.reset()
        assert observations["cameras"]["rgb"].shape == (NUM_ENVS, 64, 96, 3)
        assert observations["cameras"]["rgb"].dtype == torch.uint8
        images = []
        for step in range(13):
            observations, *_ = env.step(torch.zeros(NUM_ENVS, 1, device=env.device))
            if step >= 9:
                images.append({name: image.clone() for name, image in observations["cameras"].items()})
        assert env.is_captured and env.num_graphs == 1
        for name in ("rgb", "rgb_back"):
            # The cube falls by pixels per step by then, so every replayed step renders a new image.
            assert all(not torch.equal(before[name], after[name]) for before, after in zip(images, images[1:]))
            # Lit: a stale render graph draws the geometry black.
            assert images[-1][name].amax(dim=-1).min() > 0

        # Push every cart out of its bounds until the budget is spent.
        ended = []
        while env.episodes_completed < budget:
            _, _, terminated, truncated, _ = env.step(torch.ones(NUM_ENVS, 1, device=env.device))
            ended.append(int((terminated | truncated).sum()))
        records = env.episode_records()
        assert sum(ended) == budget and len(records["world"]) == budget
        # Every world ran an episode, the first worlds to end a second one, and none ended a third.
        episodes = torch.bincount(records["world"].long(), minlength=NUM_ENVS)
        assert episodes.min() == 1 and episodes.sum() == budget
        assert records["termination/cart_out_of_bounds"].sum() + records["truncated"].sum() == budget
        # Recorded values are those of the episode's last step, before its reset: out of bounds where it terminated.
        out_of_bounds = records["cart_x"].abs() > 1.5
        assert torch.equal(out_of_bounds, records["terminated"].bool())
        for _ in range(env.max_episode_length):
            _, _, terminated, truncated, _ = env.step(torch.ones(NUM_ENVS, 1, device=env.device))
            assert not (terminated | truncated).any()
        assert env.episodes_completed == budget
    finally:
        env.close()


def test_play_mode_removes_observation_noise_and_keeps_each_group():
    """Play mode drops observation noise, in concatenated groups and in groups of a tensor per term alike."""
    cfg = cartpole_cfg()
    noisy = mdp2.Observation(mdp.joint_pos, inputs={"joint_pos": robot.joint_pos}, noise=mdp2.UniformNoise(-0.1, 0.1))
    cfg.task.observations["separate"] = mdp2.ObservationGroup(terms={"joint_pos": noisy}, concatenate=False)
    cfg.play_mode()
    for group, concatenate in (("policy", True), ("separate", False)):
        terms, concatenated = group_terms(cfg.task.observations[group])
        assert concatenated is concatenate
        assert all(use.noise is None for use in terms.values() if use is not None)


@pytest.mark.skipif(importlib.util.find_spec("optix") is None, reason="needs the optix extra")
def test_optix_cameras_render_between_the_graphs_of_a_step():
    """OptiX refits its acceleration structures on the host, so its cameras render between two graphs, and every
    replayed step still renders a new image."""
    cfg = cartpole_cfg()
    cfg.scene.cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2), rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(), mass_props=sim_utils.MassCfg(mass=1.0)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.2)),
    )
    cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(pos=(0.0, -1.0, 1.0), rot=(0.0, 0.0, 0.7071, 0.7071), convention="world"),
        data_types=["rgb"],
        width=96,
        height=64,
        spawn=sim_utils.PinholeCameraCfg(focal_length=2.0, horizontal_aperture=2.0, clipping_range=(0.01, 20.0)),
        renderer_cfg=NewtonOptixRendererCfg(),
    )
    image = mdp2.Observation(mdp.image_rgb, inputs={"image": mdp2.Entity("camera").rgb})
    cfg.task.observations["cameras"] = mdp2.ObservationGroup(terms={"rgb": image}, concatenate=False)
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        images = []
        for step in range(10):
            observations, *_ = env.step(torch.zeros(NUM_ENVS, 1, device=env.device))
            if step >= 6:
                images.append(observations["cameras"]["rgb"].clone())
        assert env.is_captured and env.num_graphs == 2
        assert all(not torch.equal(before, after) for before, after in zip(images, images[1:]))
    finally:
        env.close()


lifetime_steps = mdp2.Term(
    torch=mdp2.Implementation(compute=lambda steps: steps.clone(), update=lambda steps: steps.add_(1.0)),
    state={"steps": mdp2.State(persistent=True)},
)
"""Steps since the start, across episodes."""


def test_update_task_keeps_the_state_of_the_uses_both_tasks_have():
    """Replacing the task on a running environment keeps the state, random streams, and tuned values of the uses
    that both tasks have, takes the other weights from the new task, truncates the episodes, and replays the new
    task's step as a graph. Without truncating, the episodes go on with the reward sums of the rewards both have."""
    cfg = cartpole_cfg()
    cfg.task.quantities = {"lifetime": mdp2.Quantity(lifetime_steps)}
    alive = replace(cfg.task.rewards["alive"], weight=mdp2.Tunable(1.0))
    cfg.task = replace(cfg.task, rewards={**cfg.task.rewards, "alive": alive})
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        actions = torch.zeros(NUM_ENVS, 1, device=env.device)
        for _ in range(3):
            env.step(actions)
        assert env.is_captured and env.num_graphs == 1
        env.tune("rewards.alive.weight", 7.0, at="step")
        streams = {name: env.state.rng.torch[row].clone() for row, name in enumerate(env.mdp.streams)}

        task = replace(
            cfg.task,
            rewards={
                **cfg.task.rewards,
                "terminating": replace(cfg.task.rewards["terminating"], weight=-5.0),
                "upright": mdp2.Reward(mdp.is_alive, inputs={"terminated": mdp2.terminated}),
            },
        )
        env.update_task(task)
        rewards = [op.name for op in env.mdp.ops[mdp2.Phase.REWARDS]]
        lifetime = env.state.arrays[mdp2.term_state("quantities.lifetime", "steps")].torch
        assert (lifetime == 3.0).all() and (env.episode_length_buf == 0).all()
        weights = env.state.reward_weights.torch
        assert (weights[:, rewards.index("rewards.alive")] == 7.0).all()  # Tuned: kept
        assert (weights[:, rewards.index("rewards.terminating")] == -5.0).all()  # Edited: the new task's
        assert weights[:, rewards.index("rewards.upright")].tolist() == [1.0] * NUM_ENVS
        for name, values in streams.items():
            # The reset that truncated the episodes advanced every stream once.
            rng.advance(values[None])
            assert torch.equal(env.state.rng.torch[env.mdp.streams.index(name)], values), name

        _, _, _, _, extras = env.step(actions)
        env.step(actions)
        assert env.is_captured and env.num_graphs == 1 and "Episode_Reward/upright" in extras["log"]
        assert (lifetime == 5.0).all()

        sums = env.step_buffers()["episode_rewards"].clone()
        env.update_task(cfg.task, truncate=False)
        kept = [rewards.index(op.name) for op in env.mdp.ops[mdp2.Phase.REWARDS]]
        assert (env.episode_length_buf == 2).all()
        assert torch.equal(env.step_buffers()["episode_rewards"], sums[kept])
    finally:
        env.close()


def test_env_without_rl_config_updates_its_task_without_truncating():
    """An MDP2Env, whose config has no episode budget, updates its task without truncating: the episodes go on."""
    rl = cartpole_cfg()
    cfg = MDP2EnvCfg(
        sim=rl.sim, scene=rl.scene, task=rl.task, decimation=rl.decimation, backend=rl.backend, seed=rl.seed
    )
    env = MDP2Env(cfg)
    try:
        env.reset()
        for _ in range(2):
            env.step(torch.zeros(NUM_ENVS, 1, device=env.device))
        env.update_task(cfg.task, truncate=False)
        assert (env.state.episode_length.torch == 2).all()
    finally:
        env.close()


def test_delayed_actuators_run_in_captured_steps():
    """Delayed actuators, which Newton runs natively, replay in the step's graph: the cart follows changing actions
    as it does stepping eagerly."""
    velocities = []
    for capture in (True, False):
        cfg = cartpole_cfg()
        cfg.capture = capture
        delayed_cart(cfg)
        env = MDP2RLEnv(cfg)
        try:
            env.reset(seed=0)
            for step in range(6):
                env.step(torch.full((NUM_ENVS, 1), (-1.0) ** step, device=env.device))
                velocities.append(env.scene["robot"].data.joint_vel.torch[:, 0].clone())
            assert env.is_captured == capture
        finally:
            env.close()
    torch.testing.assert_close(torch.stack(velocities[:6]), torch.stack(velocities[6:]))
    assert torch.stack(velocities).abs().amax() > 0.0


def test_physics_that_cannot_capture_its_step_steps_between_the_graphs(monkeypatch):
    """A physics backend that cannot capture its step steps eagerly, between the graphs of the MDP, which the backend
    still records with its own capture."""
    cfg = cartpole_cfg()
    env = MDP2RLEnv(cfg)
    try:
        manager = env.sim.backend_manager
        captures = []
        capture = manager.capture
        monkeypatch.setattr(manager, "supports_step_capture", lambda: False)
        monkeypatch.setattr(manager, "capture", lambda fn: captures.append(fn) or capture(fn))
        env._work = env._step_work()
        env.reset()
        actions = torch.ones(NUM_ENVS, 1, device=env.device)
        joint_pos = env.scene["robot"].data.joint_pos.torch
        start = joint_pos.clone()
        for _ in range(3):
            env.step(actions)
        # The physics steps eagerly, and the rest of the step replays as one graph after it.
        assert env.is_captured and env.num_graphs == 1 and len(captures) == 1
        assert (joint_pos[:, 0] != start[:, 0]).all()  # The cart moved
    finally:
        env.close()


def test_replayed_steps_do_the_host_work_of_their_physics_steps():
    """Replayed steps count their physics steps and advance the simulation time, as stepping eagerly does."""
    env = MDP2RLEnv(cartpole_cfg())
    try:
        env.reset()
        steps, time = env.sim.get_physics_step_count(), env.sim.backend_manager.get_simulation_time()
        for _ in range(5):
            env.step(torch.zeros(NUM_ENVS, 1, device=env.device))
        assert env.is_captured
        assert env.sim.get_physics_step_count() - steps == 5 * env.cfg.decimation
        assert env.sim.backend_manager.get_simulation_time() - time == pytest.approx(5 * env.step_dt)
    finally:
        env.close()


def push_pole_torch(mask, joint_vel, joint_ids):
    joint_vel[:, joint_ids.long()] += 5.0 * mask[:, None]


def test_tunables_switches_stages_variations_and_control_actions_run_in_the_captured_step():
    """A task with a tunable weight, a switched reward, a variation, an action at the control rate, and a sequence
    whose second task pushes the pole when it starts, steps alike captured and uncaptured; the solver sees the push,
    and tunings change replayed steps."""
    cfg = cartpole_cfg()
    task = cfg.task
    task = replace(
        task,
        actions={"joint_effort": replace(task.actions["joint_effort"], rate=mdp2.ActionRate.CONTROL)},
        rewards={
            **task.rewards,
            "alive": replace(task.rewards["alive"], weight=mdp2.Tunable(1.0)),
            "pole_vel": replace(task.rewards["pole_vel"], when=mdp2.Tunable(True)),
        },
    )
    ready = mdp2.Quantity(after_steps, inputs={"episode_length": mdp2.episode_length}, params={"steps": 3})
    push = mdp2.Event(
        mdp2.Term(torch=push_pole_torch),
        inputs={"joint_ids": pole},
        outputs={"joint_vel": robot.joint_vel},
        mode=mdp2.EventMode.START,
    )
    cfg.task = mdp2.compose(
        task,
        mdp.variation("cart_start", robot.joint_pos, cart, mdp.Uniform(-0.1, 0.1)),
        mdp2.TaskCfg(quantities={"ready": ready}),
        mdp2.sequence(
            warm=mdp2.Milestone(mdp2.quantities.ready),
            push=mdp2.TaskCfg(events={"push": push}, done=mdp2.terminated),
        ),
    )
    cfg.task.terminations["cart_out_of_bounds"] = None
    cfg.verify_capture = 10
    env = MDP2RLEnv(cfg)
    try:
        env.reset()
        cart_start = env.state.arrays[mdp2.quantities.cart_start].torch
        actions = torch.zeros(NUM_ENVS, 1, device=env.device)
        pole_vel = env.scene["robot"].data.joint_vel.torch[:, 1]
        for step in range(1, 6):
            _, _, _, _, extras = env.step(actions)
            if step == 2:
                still = pole_vel.abs().max().item()
        # The push task started at the end of step 3, and the solver stepped the pushed pole from step 4.
        assert env.is_captured and still < 2.0 and (pole_vel.abs() > 3.0).all()
        assert ((cart_start >= -0.1) & (cart_start < 0.1)).all() and cart_start.std() > 0
        rewards = [op.name for op in env.mdp.ops[mdp2.Phase.REWARDS]]
        alive, pole_vel_term = rewards.index("rewards.alive"), rewards.index("rewards.pole_vel")
        env.tune("rewards.alive.weight", 0.0, at="step")
        env.tune("rewards.pole_vel.when", False, at="step")
        env.step(actions)
        assert (env.transition.reward_terms.torch[:, alive] == 0.0).all()
        env.step(actions)  # The switch turned off at the end of the step before
        assert (env.transition.reward_terms.torch[:, pole_vel_term] == 0.0).all()
        assert env.state.revision.torch.tolist() == [2] * NUM_ENVS
    finally:
        env.close()
