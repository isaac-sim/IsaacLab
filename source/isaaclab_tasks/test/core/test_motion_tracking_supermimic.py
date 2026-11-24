# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab_tasks.utils import resolve_task_config

TASK = "Isaac-Motion-Tracking-G1"


def test_motion_tracking_observation_preset_preserves_policy_and_mdp() -> None:
    """The physical groups are opt-in and leave the policy observations and the MDP unchanged."""
    default, _ = resolve_task_config(TASK, None, overrides=[])
    model_based, agent = resolve_task_config(TASK, "rsl_rl_supermimic_cfg_entry_point", overrides=["presets=state_obs"])
    assert default.observations.state is None
    assert model_based.observations.policy.to_dict() == default.observations.policy.to_dict()
    for name in ("rewards", "terminations", "events", "commands", "actions", "scene"):
        assert getattr(default, name).to_dict() == getattr(model_based, name).to_dict()
    for group in ("state", "goal", "context", "labels"):
        assert not getattr(model_based.observations, group).enable_corruption
    rewards = model_based.rewards
    positive = sum(term.weight for term in vars(rewards).values() if term is not None and term.weight > 0)
    assert agent.algorithm.max_step_reward == pytest.approx(model_based.sim.dt * model_based.decimation * positive)


@pytest.mark.kitless
def test_motion_state_representation_roundtrips_increments() -> None:
    """World-model tangent predictions roundtrip rotations and keep finite derivatives at zero error."""
    from isaaclab_tasks.core.motion_tracking.supermimic import MotionStateRepresentation

    representation = MotionStateRepresentation(num_joints=3)
    flat = torch.randn(2, 4, 19, requires_grad=True)
    state = list(representation.unflatten(flat))
    state[1] = torch.nn.functional.normalize(state[1], dim=-1)
    state = tuple(state)
    increment = torch.randn(2, 4, representation.increment_dim) * 0.05
    prediction = representation.integrate(state, increment)
    torch.testing.assert_close(representation.increments(state, prediction), increment)
    representation.features(prediction).square().sum().backward()
    assert torch.isfinite(flat.grad).all()
    differentiable = tuple(value.detach().requires_grad_() for value in state)
    target = tuple(value.detach().clone() for value in state)
    representation.error(differentiable, target).square().sum().backward()
    assert all(torch.isfinite(value.grad).all() for value in differentiable)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires Newton on CUDA.")
def test_g1_differentiable_motion_tracking_matches_simulation_and_managers() -> None:
    """Forward kinematics, rewards, terminations and actor inputs match the simulated environment."""
    import gymnasium as gym

    from isaaclab.app import launch_simulation

    from isaaclab_tasks.core.motion_tracking.config.g1.supermimic import G1MotionTrackingTask

    cfg, _ = resolve_task_config(TASK, None, overrides=["presets=state_obs", "visualizer=none"])
    num_envs = 16
    cfg.scene.num_envs = num_envs
    cfg.seed = 42
    cfg.final_obs_every_step = True
    cfg.observations.policy.enable_corruption = False
    with launch_simulation(cfg, {"visualizer": ["none"]}):
        env = gym.make(TASK, cfg=cfg)
        try:
            obs, _ = env.reset()
            owner = env.unwrapped
            task = G1MotionTrackingTask(env=env, device=owner.device)
            action_term = owner.action_manager.get_term(owner.action_manager.active_terms[0])
            reflect = torch.tensor([1.0, -1.0, 1.0], device=task.device)
            robot = owner.scene["robot"]
            phase = torch.linspace(0, 2 * torch.pi, num_envs * task.num_actions, device=task.device)
            phase = phase.reshape(num_envs, -1)
            terminations = 0
            for step in range(48):
                state = task.representation.unflatten(obs["state"].clone())
                goal = obs["goal"].clone()
                history = torch.stack((owner.action_manager.prev_action, owner.action_manager.action), -2)
                pos, quat, lin, ang = task.kinematics.forward(state)
                data = robot.data
                torch.testing.assert_close(pos, data.body_link_pos_w.torch - owner.scene.env_origins[:, None])
                torch.testing.assert_close(quat, data.body_link_quat_w.torch, atol=1e-4, rtol=0)
                torch.testing.assert_close(lin, data.body_lin_vel_w.torch, atol=1e-3, rtol=1e-4)
                torch.testing.assert_close(ang, data.body_ang_vel_w.torch, atol=1e-3, rtol=1e-4)
                inputs = task.actor_obs(state, goal, obs["context"], history)
                frame = inputs["policy"]
                if step < 4:
                    # the reference action commands the reference joint positions
                    action_term.process_actions(task.reference_action(state, goal))
                    reference_targets = task.ref_joint_pos[task._frame(goal)][..., task.action_joints]
                    torch.testing.assert_close(action_term.processed_actions, reference_targets)
                    # the mirror reflects every body across the sagittal plane and is its own inverse
                    mirrored = task.mirror_state(obs["state"])
                    for mirror, value in (
                        (task.mirror_state, obs["state"]),
                        (task.mirror_actions, history),
                        (task.mirror_context, obs["context"]),
                        (task.mirror_labels, obs["labels"]),
                    ):
                        torch.testing.assert_close(mirror(mirror(value)), value)
                    mirrored_pos = task.kinematics.forward(task.representation.unflatten(mirrored))[0]
                    torch.testing.assert_close(mirrored_pos[:, task.body_perm], pos * reflect, atol=5e-3, rtol=0)
                command = 2 * task.num_joints
                torch.testing.assert_close(inputs["actor_goal"], obs["policy"][..., :command])
                torch.testing.assert_close(inputs["actor_goal"], obs["actor_goal"])
                torch.testing.assert_close(task.reference_action(state, goal), obs["reference_action"])
                torch.testing.assert_close(frame, obs["policy"][..., command:], atol=1e-4, rtol=1e-4)
                actions = 0.5 * torch.sin(phase + step * 0.3)
                obs, _, terminated, _, extras = env.step(actions)
                final = extras["final_obs"]
                successor = task.representation.unflatten(final["state"])
                terms = task.reward_terms(state, actions, successor, goal, history, final["labels"])
                actual = {name: [] for name in terms}
                for index in range(num_envs):
                    for name, values in owner.reward_manager.get_active_iterable_terms(index):
                        actual[name].append(values[0] * task.dt)
                for name, predicted in terms.items():
                    reference = torch.tensor(actual[name], device=task.device)
                    torch.testing.assert_close(predicted, reference, atol=1e-4, rtol=1e-4)
                for name, margin in zip(task.terminations, task._termination_margins(successor, goal)):
                    failed = owner.termination_manager.get_term(name)
                    torch.testing.assert_close(margin > 0, failed, msg=lambda message: f"{name}: {message}")
                    terminations += int(failed.sum())
            assert terminations > 0
            differentiable = tuple(value.detach().clone().requires_grad_() for value in successor)
            action = actions.detach().clone().requires_grad_()
            task.reward(state, action, differentiable, goal, history, final["labels"]).sum().backward()
            assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
            assert all(value.grad is None or torch.isfinite(value.grad).all() for value in differentiable)
            assert differentiable[4].grad.abs().sum() > 0
        finally:
            env.close()
