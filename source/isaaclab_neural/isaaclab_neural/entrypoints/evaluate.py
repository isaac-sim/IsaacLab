# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate one explicitly selected policy on independently selected task physics."""

import argparse
from collections.abc import Callable

import gymnasium as gym
import torch
from torch import Tensor

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.entrypoints.backends.runner_rsl_rl import create_rsl_rl_runner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli


@torch.inference_mode()
def score(env: RslRlVecEnvWrapper, policy: Callable[..., Tensor]) -> tuple[int, float, float, dict[str, float]]:
    """Report rewards, durations and termination fractions for each world's first episode.

    Aggregated command metrics cannot distinguish first endings from later autoresets, so they are not
    reconstructed from extras['log']. All worlds count once, regardless of their episode lengths.
    """
    task = env.unwrapped
    obs, _ = env.reset()
    reward, seconds = (torch.zeros(task.num_envs, device=task.device) for _ in range(2))
    running = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)  # Environments in their first episode
    endings = dict.fromkeys(task.termination_manager.active_terms, 0)
    for _ in range(task.max_episode_length):
        obs, step_reward, dones, extras = env.step(policy(obs))
        policy.reset(dones)  # The shared RSL-RL policy interface clears recurrent state at episode endings.
        reward += step_reward * running
        seconds += task.step_dt * running
        ended = running & dones.bool()
        running &= ~ended
        count = int(ended.sum())
        if count == 0:
            continue
        for name in endings:
            endings[name] += int((ended & task.termination_manager.get_term(name)).sum())
        if not running.any():
            break
    episode_count = int((~running).sum())
    divisor = episode_count or float("nan")
    return (
        episode_count,
        reward[~running].sum().item() / divisor,
        seconds[~running].sum().item() / divisor,
        {name: count / divisor for name, count in endings.items()},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trained_on", required=True, help="Physics preset used to train the policy.")
    parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, argv)
    # Explicitly replace only physics; domain presets and policy overrides retain their normal meaning.
    policy_overrides = [
        override
        for override in overrides
        if not override.startswith(("physics=", "env.sim.physics=", "env.sim.physics."))
    ]
    policy_overrides.append(f"physics={args.trained_on}")
    _, agent_cfg = resolve_task_config(args.task, args.agent, overrides=policy_overrides)
    env_cfg, _ = resolve_task_config(args.task, args.agent, play_mode=True, overrides=overrides)
    env_cfg.scene.num_envs, env_cfg.seed = args.num_envs, args.seed
    checkpoint = retrieve_file_path(args.checkpoint)
    with launch_simulation(env_cfg, args):
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        try:
            agent_cfg.device = env.unwrapped.device
            runner = create_rsl_rl_runner(env, agent_cfg, inference=True)
            runner.load(checkpoint, map_location=env.unwrapped.device)
            episodes, reward, seconds, endings = score(env, runner.get_inference_policy(device=env.unwrapped.device))
            print(f"Checkpoint {checkpoint}; trained on {args.trained_on}")
            print(f"{episodes} episodes, mean reward {reward:.3f}, mean duration {seconds:.3f} s")
            for name, fraction in endings.items():
                print(f"  {name}: {fraction:.2%}")
        finally:
            env.close()
