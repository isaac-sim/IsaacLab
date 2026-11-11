# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MDP2 Kuka Allegro lift task on Newton."""

import math
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_experimental import mdp2
from isaaclab_experimental.envs import MDP2RLEnv

import isaaclab.envs.mdp as mdp1
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.warp import ProxyArray

from isaaclab_tasks.core.lift import mdp as lift_mdp1
from isaaclab_tasks.mdp2.lift_kuka_allegro.lift_env_cfg import KukaAllegroLiftEnvCfg, KukaAllegroLiftPhysicsCfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Newton needs a GPU")

NUM_ENVS = 16
FINGERTIPS = ("index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3")


@pytest.fixture(scope="module")
def scenario() -> Iterator[SimpleNamespace]:
    """The task after its first reset and a few random steps, with objects near the palm, goals near the objects,
    fingertip forces around the contact thresholds, and joint speeds above the limits in some worlds."""
    cfg = KukaAllegroLiftEnvCfg()
    cfg.sim.physics = KukaAllegroLiftPhysicsCfg().newton_mjwarp
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    cfg.task.events["reset"].params["bank_size"] = 2
    env = MDP2RLEnv(cfg)
    try:
        generator = torch.Generator(device=env.device).manual_seed(0)

        def rand(*shape, scale=1.0):
            return scale * (2.0 * torch.rand(*shape, device=env.device, generator=generator) - 1.0)

        env.reset()
        restored = bank_and_scene_state(env, env.state)
        for _ in range(10):
            env.step(rand(NUM_ENVS, env.mdp.action_dim))

        robot, target = env.scene["robot"], env.scene["object"]
        palm = robot.data.body_link_pose_w.torch[:, robot.body_names.index("palm_link")]
        pose = torch.cat([palm[:, :3] + rand(NUM_ENVS, 3, scale=0.05), palm[:, 3:7]], dim=-1)
        target.write_root_pose_to_sim_index(root_pose=pose)
        speeds = robot.data.soft_joint_vel_limits.torch * rand(NUM_ENVS, 1, scale=1.2)
        robot.write_joint_velocity_to_sim_index(velocity=speeds)
        env.sim.forward()
        env.scene.update(dt=0.0)
        forces = env.scene["fingertip_contacts"].data.normal_force_matrix_w.torch
        forces.copy_(rand(*forces.shape, scale=0.5))

        root = robot.data.root_link_pose_w.torch
        goal, _ = math_utils.subtract_frame_transforms(
            root[:, :3], root[:, 3:7], pose[:, :3] + rand(NUM_ENVS, 3, scale=0.05)
        )
        command = torch.cat([goal, torch.nn.functional.normalize(rand(NUM_ENVS, 4), dim=-1)], dim=-1)
        best_error = rand(NUM_ENVS, scale=0.05).abs() + 0.02
        yield SimpleNamespace(env=env, cfg=cfg, restored=restored, command=command, best_error=best_error)
    finally:
        env.close()


def bank_and_scene_state(env: MDP2RLEnv, state: mdp2.MDPState) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return, per banked field, the banked state that each world plays and the field's value in the scene."""
    bank = state.term_state["events.reset"]
    rows = torch.arange(NUM_ENVS, device=env.device), bank["playing"].torch.long()
    robot, target = env.scene["robot"].data, env.scene["object"].data
    fields = {
        "joint_pos": robot.joint_pos,
        "joint_vel": robot.joint_vel,
        "object_pose_w": target.root_link_pose_w,
        "object_vel_w": target.root_com_vel_w,
    }
    return {name: (bank[f"bank_{name}"].torch[rows].clone(), field.torch.clone()) for name, field in fields.items()}


def build(scenario: SimpleNamespace, backend: str) -> tuple[mdp2.MDP, mdp2.MDPState]:
    """Compile the task for a backend, with the environment's term state: its point clouds and its bank."""
    env = scenario.env
    builder = mdp2.MDPBuilder(env.scene_data.schema)
    builder.add_task(scenario.cfg.task)
    mdp = builder.finalize(device=env.device, backend=backend, diagnostics=True)
    state = mdp.state(env.scene_data)

    for op_name, values in env.state.term_state.items():
        for name, value in values.items():
            state.term_state[op_name][name].torch.copy_(value.torch)
    return mdp, state


def evaluate(scenario: SimpleNamespace, backend: str, progress: float) -> tuple[mdp2.MDP, mdp2.MDPState, dict]:
    """Evaluate the task once from the scenario's state, at the given curriculum progress."""
    mdp, state = build(scenario, backend)
    transition = mdp.transition()

    # A progress the objects can beat.
    state.term_state["events.adr"]["progress"].torch.fill_(progress)
    state.fields[mdp2.commands.object_pose].torch.copy_(scenario.command)
    state.term_state["rewards.position_tracking"]["last_command"].torch.copy_(scenario.command)
    state.term_state["rewards.position_tracking"]["best_error"].torch.copy_(scenario.best_error)

    mdp2.evaluate(mdp, state, transition)
    results = {
        **{group: values.torch.clone() for group, values in transition.observations.items()},
        "rewards": transition.reward_terms.torch.clone(),
        "terminations": transition.termination_terms.torch.clone(),
    }
    return mdp, state, results


def test_bank_restores_and_terms_agree_across_backends(scenario: SimpleNamespace):
    """The first reset fills the bank and restarts every world from it, and observations, rewards, and terminations
    agree on both backends, halfway through the curriculum."""
    for name, (banked, restored) in scenario.restored.items():
        torch.testing.assert_close(restored, banked, atol=1e-5, rtol=0.0, msg=lambda message, name=name: name)

    (mdp, _, torch_results), (_, _, warp_results) = (evaluate(scenario, backend, 0.5) for backend in ("torch", "warp"))

    names = [op.name for op in mdp.ops[mdp2.Phase.REWARDS]]
    for name in ("rewards.position_tracking", "rewards.success", "rewards.good_finger_contact"):
        assert torch_results["rewards"][:, names.index(name)].any(), name  # The goal and contact terms are exercised.
    for key, value in warp_results.items():
        torch.testing.assert_close(
            torch_results[key], value, atol=1e-5, rtol=1e-5, msg=lambda message, key=key: f"{key}: {message}"
        )


class MDP1Scene:
    """The scene as MDP1's lift terms see it, with one contact sensor per fingertip."""

    def __init__(self, scene, forces: torch.Tensor):
        self.scene = scene
        self.num_envs = scene.num_envs
        self.env_origins = scene.env_origins
        self.sensors = {
            f"{name}_object_s": SimpleNamespace(
                data=SimpleNamespace(normal_force_matrix_w=SimpleNamespace(torch=forces[:, k].contiguous()))
            )
            for k, name in enumerate(FINGERTIPS)
        }

    def __getitem__(self, name: str):
        return self.scene[name]


def test_terms_match_mdp1(scenario: SimpleNamespace):
    """Observations, rewards, the success state, and terminations match MDP1's lift terms on the same state."""
    env = scenario.env
    mdp, state, results = evaluate(scenario, "warp", 0.0)

    forces = env.scene["fingertip_contacts"].data.normal_force_matrix_w.torch
    mdp1_env = SimpleNamespace(
        scene=MDP1Scene(env.scene, forces),
        num_envs=NUM_ENVS,
        device=env.device,
        command_manager=SimpleNamespace(get_command=lambda name: scenario.command),
    )
    robot, target = SceneEntityCfg("robot"), SceneEntityCfg("object")
    hand = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])
    for cfg in (robot, target, hand):
        cfg.resolve(env.scene)
    thumb = {"thumb_name": "thumb_link_3_object_s", "finger_names": [f"{name}_object_s" for name in FINGERTIPS[:3]]}
    goal = {"command_name": "object_pose", "asset_cfg": robot, "align_asset_cfg": target, **thumb}

    progress = lift_mdp1.position_command_progress(RewardTermCfg(func=None, params={}), mdp1_env)
    progress.best_error.copy_(scenario.best_error)
    progress._prev_command = scenario.command.clone()
    success = lift_mdp1.success_reward(RewardTermCfg(func=None, params={}), mdp1_env)
    out_of_bound = lift_mdp1.out_of_bound(TerminationTermCfg(func=None, params={"asset_cfg": target}), mdp1_env)
    body_state = lift_mdp1.body_state_b(None, mdp1_env)
    joint_vel_out_of_limit = mdp1.joint_vel_out_of_limit(
        TerminationTermCfg(func=None, params={"asset_cfg": robot}), mdp1_env
    )
    sensor_names = [f"{name}_object_s" for name in FINGERTIPS]
    expected = {
        "observations.policy.object_quat_b": lift_mdp1.object_quat_b(mdp1_env, robot, target),
        # Observations see the command that this evaluation resampled; rewards see the scenario's.
        "observations.policy.target_object_pose_b": state.fields[mdp2.commands.object_pose].torch,
        "observations.proprio.joint_pos": env.scene["robot"].data.joint_pos.torch,
        "observations.proprio.joint_vel": env.scene["robot"].data.joint_vel.torch,
        "observations.proprio.hand_tips_state_b": body_state(mdp1_env, hand, robot, include_vel=False).clamp(-2, 2),
        "observations.proprio.contact": lift_mdp1.fingers_contact_force_b(mdp1_env, sensor_names, robot).clamp(-20, 20),
        "rewards.fingers_to_object": lift_mdp1.object_ee_distance(mdp1_env, 0.4, asset_cfg=hand, **thumb),
        "rewards.position_tracking": progress(mdp1_env, **goal, contact_threshold=0.1, min_improvement=0.0025),
        "rewards.success": success(mdp1_env, **goal, pos_std=0.05, contact_threshold=0.01),
        "rewards.good_finger_contact": lift_mdp1.contacts(mdp1_env, 0.01, **thumb),
        "rewards.contact_count": lift_mdp1.contact_count(mdp1_env, 0.01, sensor_names),
        "terminations.object_out_of_bound": out_of_bound(
            mdp1_env, target, {"x": (-1.5, 0.5), "y": (-2.0, 2.0), "z": (0.3, 2.0)}
        ),
        "terminations.abnormal_robot": joint_vel_out_of_limit(mdp1_env, robot),
    }

    # The latest value of each observation is the last block of its history.
    values = {}
    for op in mdp.ops[mdp2.Phase.OBSERVATIONS]:
        width = math.prod(op.shape)
        values[op.name] = results[op.group][:, op.columns][:, -width:]
    for op in mdp.ops[mdp2.Phase.REWARDS]:
        values[op.name] = results["rewards"][:, op.index] / (op.weights[0] * env.step_dt)
    for op in mdp.ops[mdp2.Phase.TERMINATIONS]:
        values[op.name] = results["terminations"][:, op.index]

    assert expected["terminations.abnormal_robot"].any() and not expected["terminations.abnormal_robot"].all()
    assert expected["rewards.position_tracking"].any() and expected["rewards.success"].any()
    for name, value in expected.items():
        expected_value = value.reshape(NUM_ENVS, -1).squeeze(-1).to(values[name].dtype)
        torch.testing.assert_close(values[name], expected_value, msg=lambda message, name=name: f"{name}: {message}")
    assert torch.equal(state.fields[mdp2.quantities.success].torch > 0.5, success.succeeded)


def test_step_replays_as_one_graph_with_curriculum_and_bank(scenario: SimpleNamespace):
    """The whole step is one CUDA graph, in which resets restore banked states on both backends alike, and the
    curriculum's gravity reaches the solver."""
    env = scenario.env
    assert env.mdp.is_fully_capturable and env.is_captured

    # Both backends draw the same banked state for every world, and restore it.
    mask = ProxyArray(wp.ones(NUM_ENVS, dtype=wp.bool, device=env.device))
    playing = {}
    for backend in ("torch", "warp"):
        mdp, state = build(scenario, backend)
        mdp2.reset(mdp, state, mask)
        playing[backend] = state.term_state["events.reset"]["playing"].torch.clone()
        for name, (banked, restored) in bank_and_scene_state(env, state).items():
            torch.testing.assert_close(restored, banked, atol=1e-5, rtol=0.0, msg=lambda message, name=name: name)
    assert torch.equal(playing["torch"], playing["warp"])

    # At full progress, the worlds that the replayed graph resets get the final gravity.
    env.state.term_state["events.adr"]["progress"].torch.fill_(1.0)
    env.episode_length_buf = torch.full_like(env.episode_length_buf, env.max_episode_length)
    env.step(torch.zeros(NUM_ENVS, env.mdp.action_dim, device=env.device))

    final = torch.full((NUM_ENVS,), -9.81, device=env.device)
    solver_gravity = wp.to_torch(env.sim.backend_manager.backend.solver.mjw_model.opt.gravity)
    torch.testing.assert_close(env.scene_data.data("scene", "gravity_w").torch[:, 2], final)
    torch.testing.assert_close(solver_gravity[:NUM_ENVS, 2], final)
