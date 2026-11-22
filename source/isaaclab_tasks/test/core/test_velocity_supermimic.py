# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_tasks.utils.hydra import collect_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASKS = ("Isaac-Velocity-Flat-G1", "Isaac-Velocity-Rough-G1")
AGENTS = ("rsl_rl_supermimic_cfg_entry_point", "rsl_rl_blinkmimic_cfg_entry_point")


@pytest.mark.parametrize("task", TASKS)
def test_velocity_observation_preset_preserves_policy_and_mdp(task):
    """Optional physical groups leave policy and downstream observation customizations intact."""
    raw = load_cfg_from_registry(task, "env_cfg_entry_point")
    assert "state_obs" in collect_presets(raw)["observations.state"]
    default, _ = resolve_task_config(task, None, overrides=[])
    model_based, _ = resolve_task_config(task, None, overrides=["presets=state_obs"])
    assert model_based.observations.policy.to_dict() == default.observations.policy.to_dict()
    assert (default.observations.policy.height_scan is None) == (task == "Isaac-Velocity-Flat-G1")
    for name in ("rewards", "terminations", "events", "commands", "actions", "scene"):
        assert getattr(default, name).to_dict() == getattr(model_based, name).to_dict()
    for group in ("state", "goal", "context", "labels"):
        assert not getattr(model_based.observations, group).enable_corruption

    navigation = load_cfg_from_registry("IsaacContrib-Navigation-Flat-AnymalC", "env_cfg_entry_point")
    assert navigation.actions.pre_trained_policy_action.low_level_observations.height_scan is None


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("agent", AGENTS)
def test_g1_supermimic_max_step_reward_matches_reward_terms(task, agent):
    env_cfg, agent_cfg = resolve_task_config(task, agent, overrides=[])
    rewards = env_cfg.rewards
    assert {name for name, term in vars(rewards).items() if term is not None and term.weight > 0} == {
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "feet_air_time",
    }
    assert rewards.feet_air_time.func is mdp.feet_air_time_positive_biped
    max_step_reward = (
        env_cfg.sim.dt
        * env_cfg.decimation
        * (
            rewards.track_lin_vel_xy_exp.weight
            + rewards.track_ang_vel_z_exp.weight
            + rewards.feet_air_time.weight * rewards.feet_air_time.params["threshold"]
        )
    )
    assert agent_cfg.algorithm.max_step_reward == pytest.approx(max_step_reward)


def test_velocity_state_representation_manifold_and_analytic_clocks():
    """World-model tangent predictions must roundtrip rotations without learning clocks."""
    from isaaclab_tasks.core.velocity.supermimic import VelocityStateRepresentation

    representation = VelocityStateRepresentation(num_joints=3, num_feet=2)
    flat = torch.randn(2, 4, 25, requires_grad=True)
    state = list(representation.unflatten(flat))
    state[2] = torch.nn.functional.normalize(state[2], dim=-1)
    state = tuple(state)
    increment = torch.randn(2, 4, representation.increment_dim) * 0.05
    prediction = representation.integrate(state, increment)
    torch.testing.assert_close(representation.increments(state, prediction), increment)
    for actual, expected in zip(prediction[6:], state[6:]):
        torch.testing.assert_close(actual, expected)
    representation.features(prediction).square().sum().backward()
    assert torch.isfinite(flat.grad).all()
    # the world position stays out of the dynamics features
    assert flat.grad[..., 10:13].abs().sum() == 0
    zero = representation.integrate(state, torch.zeros_like(increment))
    for actual, expected in zip(zero, state):
        torch.testing.assert_close(actual, expected)
    differentiable = tuple(value.detach().requires_grad_() for value in state)
    target = tuple(value.detach().clone() for value in state)
    representation.error(differentiable, target).square().sum().backward()
    assert all(value.grad is None or torch.isfinite(value.grad).all() for value in differentiable)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires Newton on CUDA.")
@pytest.mark.parametrize("task_name", TASKS)
def test_g1_differentiable_velocity_matches_simulation_and_managers(task_name):
    """Compare actor inputs, all 16 reward terms and terminations against the physical managers on transitions.

    PD torque and policy-rate qacc are approximations, with explicit absolute
    tolerances in weighted, dt-scaled reward units for this bounded-action probe.
    Contact rewards use observed labels/timers here; imagined contact probabilities
    and policy-resolution clock advancement are separate model approximations.
    The heightfield scan matches the ray-cast scan except within one heightfield
    cell of vertical steps.
    """
    import gymnasium as gym

    from isaaclab.app import launch_simulation

    from isaaclab_tasks.core.velocity.config.g1.supermimic import G1VelocityTask

    cfg, _ = resolve_task_config(
        task_name, None, overrides=["presets=state_obs", "physics=newton_mjwarp", "visualizer=none"]
    )
    num_envs = 32
    cfg.scene.num_envs = num_envs
    cfg.seed = 42
    cfg.final_obs_every_step = True
    cfg.observations.policy.enable_corruption = False
    approximation_tolerances = {"dof_torques_l2": 0.0001, "dof_acc_l2": 0.005}
    with launch_simulation(cfg, {"visualizer": ["none"]}):
        env = gym.make(task_name, cfg=cfg)
        try:
            obs, _ = env.reset()
            owner = env.unwrapped
            task = G1VelocityTask(env=env, device=owner.device)
            assert task.dt == pytest.approx(0.02)
            assert len(task.rewards) == 16
            num_scan = 0 if task.scan_points is None else len(task.scan_points)
            assert num_scan == (187 if task_name == "Isaac-Velocity-Rough-G1" else 0)
            phase = torch.linspace(0, 2 * torch.pi, num_envs * task.num_actions, device=task.device)
            phase = phase.reshape(num_envs, -1)
            scan_errors = []
            for step in range(32):
                state = task.representation.unflatten(obs["state"].clone())
                goal = obs["goal"].clone()
                history = torch.stack((owner.action_manager.prev_action, owner.action_manager.action), -2)
                frame = task.actor_obs(state, goal, obs["context"], history)
                policy = torch.cat((obs["policy"][..., :9], obs["policy"][..., 12:]), -1)
                proprio = frame.shape[-1] - num_scan
                torch.testing.assert_close(frame[..., :proprio], policy[..., :proprio])
                scan_errors.append((frame[..., proprio:] - policy[..., proprio:]).abs().flatten())
                # Privileged masses and contact clocks cannot enter the actor.
                torch.testing.assert_close(frame, task.actor_obs(state, goal, 2 * obs["context"], history))
                if step < 4:
                    # The mirror is an involution that reflects the feet and the terrain below them.
                    for mirror, value in (
                        (task.mirror_state, obs["state"]),
                        (task.mirror_actions, history),
                        (task.mirror_context, obs["context"]),
                        (task.mirror_labels, obs["labels"]),
                    ):
                        torch.testing.assert_close(mirror(mirror(value)), value)
                    reflected = task.representation.unflatten(task.mirror_state(obs["state"]))
                    feet = task._bodies(state)[0][:, : task.num_feet]
                    reflected_feet = task._bodies(reflected)[0][:, : task.num_feet]
                    torch.testing.assert_close(
                        reflected_feet,
                        feet[:, task.foot_perm] * task.vector_sign + task.position_offset,
                        atol=1e-4,
                        rtol=0,
                    )
                    # Each reflected sole point sees the terrain of its counterpart on the other foot.
                    soles = task.sole_points[0]
                    corners = (soles[:, None] * task.vector_sign - soles).norm(dim=-1).argmin(-1)
                    num_soles = task.num_feet * len(soles)
                    context = task.dynamics_context(state, obs["context"], history)[:, -num_soles - 1 :]
                    reflected_context = task.dynamics_context(
                        reflected, task.mirror_context(obs["context"]), task.mirror_actions(history)
                    )[:, -num_soles - 1 :]
                    clearance = context[:, 1:].reshape(num_envs, task.num_feet, -1)[:, task.foot_perm][..., corners]
                    torch.testing.assert_close(
                        reflected_context, torch.cat((context[:, :1], clearance.flatten(1)), -1), atol=1e-3, rtol=0
                    )
                actions = 0.1 * torch.sin(phase + step * 0.3)
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
                    torch.testing.assert_close(
                        predicted, reference, atol=approximation_tolerances.get(name, 1e-6), rtol=1e-5
                    )
                torch.testing.assert_close(task.termination_prob(successor, goal, final["labels"]), terminated.float())
            if num_scan:
                scan_errors = torch.cat(scan_errors)
                assert scan_errors.median() < 1e-3
                assert (scan_errors > 0.01).float().mean() < 0.03
            # A positive physical contact prevents an always-zero termination
            # head from passing the otherwise mostly upright transition sample.
            robot = owner.scene["robot"]
            fallen_pose = robot.data.root_link_pose_w.torch[:1].clone()
            fallen_pose[..., 2] = owner.scene.env_origins[0, 2] + 0.05
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
            clock_state[6] = torch.full_like(clock_state[6], 0.2)
            clock_state[7] = torch.full_like(clock_state[7], 0.1)
            clock_state[8] = torch.full_like(clock_state[8], 0.15)
            advanced = task.post_dynamics(tuple(clock_state), successor, labels)
            sum(value.sum() for value in advanced[6:]).backward()
            assert torch.isfinite(labels.grad).all() and labels.grad.abs().sum() > 0
            # Imagined rewards, actor inputs and terrain heights must differentiate with respect to actions and state.
            differentiable = tuple(value.detach().clone().requires_grad_() for value in successor)
            action = actions.detach().clone().requires_grad_()
            (
                task.reward(state, action, differentiable, goal, history, final["labels"]).sum()
                + task.actor_obs(differentiable, goal, obs["context"], history).sum()
                + task.dynamics_context(differentiable, obs["context"], history).sum()
            ).backward()
            assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
            assert all(value.grad is None or torch.isfinite(value.grad).all() for value in differentiable)
            assert differentiable[3].grad[:, 2].abs().sum() > 0
            if num_scan:
                assert differentiable[3].grad[:, :2].abs().sum() > 0
            # Resolve the real policy noise cfg through the task constructor;
            # corruption must preserve the actor's input gradients and actions.
            cfg.observations.policy.enable_corruption = True
            noisy_task = G1VelocityTask(env=env, device=task.device)
            noise_input = frame.detach().clone().requires_grad_()
            noisy = noisy_task.add_noise(noise_input)
            assert noisy.shape == noise_input.shape and not torch.equal(noisy, noise_input)
            actions_end = noisy.shape[-1] - num_scan
            torch.testing.assert_close(
                noisy[..., actions_end - task.num_actions : actions_end],
                noise_input[..., actions_end - task.num_actions : actions_end],
            )
            noisy.sum().backward()
            assert torch.isfinite(noise_input.grad).all()
            torch.testing.assert_close(
                noise_input.grad[..., :actions_end], torch.ones_like(noise_input[..., :actions_end])
            )
        finally:
            env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires Newton on CUDA.")
@pytest.mark.parametrize("task_name", TASKS)
def test_g1_velocity_supermimic_learns_in_cuda_graphs(task_name):
    """The BlinkMimic agent captures collection and learner iterations, including mirrored batches, in CUDA graphs."""
    import gymnasium as gym

    from isaaclab.app import launch_simulation

    from isaaclab_rl.entrypoints.backends.runner_rsl_rl import configure_rsl_rl_env_cfg, create_rsl_rl_runner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    cfg, agent = resolve_task_config(
        task_name,
        "rsl_rl_blinkmimic_cfg_entry_point",
        overrides=["presets=state_obs", "physics=newton_mjwarp", "visualizer=none"],
    )
    cfg.scene.num_envs = 64
    configure_rsl_rl_env_cfg(cfg, agent)
    agent.actor.encoder_hidden_dims = agent.actor.decoder_hidden_dims = (64, 64)
    agent.world_model.hidden_dims = (64, 64)
    agent.critic.hidden_dims = [64]
    algorithm = agent.algorithm
    algorithm.replay_buffer_capacity = 64 * 64
    algorithm.num_warmup_steps = 8
    algorithm.updates_per_step = 1.0
    algorithm.world_model_batch_size = algorithm.policy_batch_size = 64
    algorithm.horizon = 4
    assert algorithm.use_cuda_graph and algorithm.use_mixed_precision and algorithm.world_model_mirror_fraction
    with launch_simulation(cfg, {"visualizer": ["none"]}):
        env = RslRlVecEnvWrapper(gym.make(task_name, cfg=cfg), clip_actions=agent.clip_actions)
        try:
            runner = create_rsl_rl_runner(env, agent, log_dir=None)
            runner.learn(num_learning_iterations=16)
            learner = runner.alg
            assert learner.iteration >= 8 and learner._graph_updates and learner._act_graphs
            parameters = (*learner.actor.parameters(), *learner.world_model.parameters(), *learner.critic.parameters())
            assert all(torch.isfinite(parameter).all() for parameter in parameters)
        finally:
            env.close()
