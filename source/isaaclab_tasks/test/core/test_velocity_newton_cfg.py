# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg

from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_tasks.utils.hydra import collect_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def test_g1_rough_newton_has_sufficient_constraint_capacity():
    env_cfg = G1RoughEnvCfg()

    solver_cfg = env_cfg.sim.physics.newton_mjwarp.solver_cfg

    assert isinstance(solver_cfg, MJWarpSolverCfg)
    assert solver_cfg.njmax == 300


def test_flat_velocity_observation_preset_preserves_policy_and_mdp():
    """Optional physical groups leave policy and downstream observation customizations intact."""
    task = "Isaac-Velocity-Flat-G1"
    raw = load_cfg_from_registry(task, "env_cfg_entry_point")
    assert "state_obs" in collect_presets(raw)["observations.state"]
    default, _ = resolve_task_config(task, None, overrides=[])
    model_based, _ = resolve_task_config(task, None, overrides=["presets=state_obs"])
    assert model_based.observations.policy.to_dict() == default.observations.policy.to_dict()
    assert default.observations.policy.height_scan is None
    for name in ("rewards", "terminations", "events", "commands", "actions", "scene"):
        assert getattr(default, name).to_dict() == getattr(model_based, name).to_dict()
    for group in ("state", "goal", "context", "labels"):
        assert not getattr(model_based.observations, group).enable_corruption

    navigation = load_cfg_from_registry("IsaacContrib-Navigation-Flat-AnymalC", "env_cfg_entry_point")
    assert navigation.actions.pre_trained_policy_action.low_level_observations.height_scan is None


def test_velocity_state_representation_manifold_and_analytic_clocks():
    """World-model tangent predictions must roundtrip rotations without learning clocks."""
    from isaaclab_tasks.core.velocity.supermimic import VelocityStateRepresentation

    representation = VelocityStateRepresentation(num_joints=3, num_feet=2)
    flat = torch.randn(2, 4, 29, requires_grad=True)
    state = list(representation.unflatten(flat))
    state[2] = torch.nn.functional.normalize(state[2], dim=-1)
    state = tuple(state)
    increment = torch.randn(2, 4, representation.increment_dim) * 0.05
    prediction = representation.integrate(state, increment)
    torch.testing.assert_close(representation.increments(state, prediction), increment)
    for actual, expected in zip(prediction[7:], state[7:]):
        torch.testing.assert_close(actual, expected)
    representation.features(prediction).square().sum().backward()
    assert torch.isfinite(flat.grad).all()
    zero = representation.integrate(state, torch.zeros_like(increment))
    for actual, expected in zip(zero, state):
        torch.testing.assert_close(actual, expected)
    differentiable = tuple(value.detach().requires_grad_() for value in state)
    target = tuple(value.detach().clone() for value in state)
    representation.error(differentiable, target).square().sum().backward()
    assert all(value.grad is None or torch.isfinite(value.grad).all() for value in differentiable)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires Newton on CUDA.")
def test_g1_differentiable_velocity_matches_reward_and_termination_managers():
    """Compare all 16 terms against the actual physical managers on transitions.

    PD torque and policy-rate qacc are approximations, with explicit absolute
    tolerances in weighted, dt-scaled reward units for this bounded-action probe.
    Contact rewards use observed labels/timers here; imagined contact probabilities
    and policy-resolution clock advancement are separate model approximations.
    """
    import gymnasium as gym

    from isaaclab.app import launch_simulation

    from isaaclab_tasks.core.velocity.supermimic import VelocityTask

    cfg, _ = resolve_task_config(
        "Isaac-Velocity-Flat-G1",
        None,
        overrides=["presets=state_obs", "physics=newton_mjwarp", "visualizer=none"],
    )
    cfg.scene.num_envs = 16
    cfg.seed = 42
    cfg.final_obs_every_step = True
    cfg.observations.policy.enable_corruption = False
    approximation_tolerances = {"dof_torques_l2": 0.0001, "dof_acc_l2": 0.005}
    with launch_simulation(cfg, {"visualizer": ["none"]}):
        env = gym.make("Isaac-Velocity-Flat-G1", cfg=cfg)
        try:
            obs, _ = env.reset()
            task = VelocityTask(env=env, device=env.unwrapped.device)
            assert task.dt == pytest.approx(0.02)
            assert len(task.rewards) == 16
            phase = torch.linspace(0, 2 * torch.pi, 16 * task.num_actions, device=task.device).reshape(16, -1)
            for step in range(32):
                state = task.representation.unflatten(obs["state"].clone())
                goal = obs["goal"].clone()
                history = torch.stack(
                    (env.unwrapped.action_manager.prev_action, env.unwrapped.action_manager.action), -2
                )
                frame = task.actor_obs(state, goal, obs["context"], history)
                torch.testing.assert_close(frame, torch.cat((obs["policy"][..., :9], obs["policy"][..., 12:]), -1))
                # Privileged masses and contact clocks cannot enter the actor.
                torch.testing.assert_close(frame, task.actor_obs(state, goal, 2 * obs["context"], history))
                actions = 0.1 * torch.sin(phase + step * 0.3)
                obs, _, terminated, _, extras = env.step(actions)
                final = extras["final_obs"]
                successor = task.representation.unflatten(final["state"])
                terms = task.reward_terms(state, actions, successor, goal, history, final["labels"])
                actual = {name: [] for name in terms}
                for index in range(16):
                    for name, values in env.unwrapped.reward_manager.get_active_iterable_terms(index):
                        actual[name].append(values[0] * task.dt)
                for name, predicted in terms.items():
                    reference = torch.tensor(actual[name], device=task.device)
                    torch.testing.assert_close(
                        predicted, reference, atol=approximation_tolerances.get(name, 1e-6), rtol=1e-5
                    )
                torch.testing.assert_close(task.termination_prob(successor, goal, final["labels"]), terminated.float())
            # A positive physical contact prevents an always-zero termination
            # head from passing the otherwise mostly upright transition sample.
            robot = env.unwrapped.scene["robot"]
            fallen_pose = robot.data.root_link_pose_w.torch[:1].clone()
            fallen_pose[..., 2] = env.unwrapped.scene.env_origins[0, 2] + 0.05
            fallen_pose[..., 3:] = torch.tensor([0.0, 2**-0.5, 0.0, 2**-0.5], device=task.device)
            robot.write_root_pose_to_sim_index(root_pose=fallen_pose, env_ids=[0])
            for _ in range(4):
                _, _, terminated, _, extras = env.step(torch.zeros_like(actions))
                contact_state = task.representation.unflatten(extras["final_obs"]["state"])
                contact_probability = task.termination_prob(contact_state, goal, extras["final_obs"]["labels"])
                torch.testing.assert_close(contact_probability, terminated.float())
                if terminated[0]:
                    break
            assert terminated[0] and contact_probability[0] == 1
            # Analytic clocks follow differentiable soft events and remain outside
            # the world-model increment loss. These gradients must survive rollout.
            labels = final["labels"].detach().clone().requires_grad_()
            clock_state = list(successor)
            clock_state[7] = torch.full_like(clock_state[7], 0.2)
            clock_state[8] = torch.full_like(clock_state[8], 0.1)
            clock_state[9] = torch.full_like(clock_state[9], 0.15)
            advanced = task.post_dynamics(tuple(clock_state), successor, labels)
            sum(value.sum() for value in advanced[7:]).backward()
            assert torch.isfinite(labels.grad).all() and labels.grad.abs().sum() > 0
            # Imagined rewards must differentiate with respect to actions and state.
            differentiable = tuple(value.detach().clone().requires_grad_() for value in successor)
            action = actions.detach().clone().requires_grad_()
            task.reward(state, action, differentiable, goal, history, final["labels"]).sum().backward()
            assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
            assert all(value.grad is None or torch.isfinite(value.grad).all() for value in differentiable)
            # Resolve the real policy noise cfg through the task constructor;
            # corruption must preserve the actor's input gradients and actions.
            cfg.observations.policy.enable_corruption = True
            noisy_task = VelocityTask(env=env, device=task.device)
            noise_input = frame.detach().clone().requires_grad_()
            noisy = noisy_task.add_noise(noise_input)
            assert noisy.shape == noise_input.shape and not torch.equal(noisy, noise_input)
            torch.testing.assert_close(noisy[..., -task.num_actions :], noise_input[..., -task.num_actions :])
            noisy.sum().backward()
            torch.testing.assert_close(noise_input.grad, torch.ones_like(noise_input))
        finally:
            env.close()
