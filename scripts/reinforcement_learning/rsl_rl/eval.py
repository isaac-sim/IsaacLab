# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a trained RSL-RL policy over N episodes and report success rate and metrics."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate a trained RSL-RL policy.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_episodes", type=int, default=100, help="Total number of episodes to evaluate.")
parser.add_argument(
    "--success_term",
    type=str,
    default=None,
    help=(
        "Name of a termination term that indicates task success. "
        "If not set, only timeout vs. early-termination statistics are reported."
    ),
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time
from collections import defaultdict

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate a trained RSL-RL policy."""
    # -- resolve task name and checkpoint -----------------------------------------------
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # -- create environment -------------------------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    num_envs = env.unwrapped.num_envs

    # -- load policy --------------------------------------------------------------------
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # -- resolve which termination terms exist ------------------------------------------
    unwrapped = env.unwrapped
    has_termination_manager = hasattr(unwrapped, "termination_manager")
    term_names: list[str] = []
    if has_termination_manager:
        term_names = unwrapped.termination_manager.active_terms
        print(f"[INFO] Termination terms: {term_names}")

    success_term = args_cli.success_term
    if success_term and success_term not in term_names:
        print(
            f"[WARN] --success_term '{success_term}' not found among termination terms {term_names}. "
            "Success rate will not be tracked."
        )
        success_term = None

    has_reward_manager = hasattr(unwrapped, "reward_manager")
    reward_term_names: list[str] = []
    if has_reward_manager:
        reward_term_names = unwrapped.reward_manager.active_terms
        print(f"[INFO] Reward terms: {reward_term_names}")

    # -- tracking state -----------------------------------------------------------------
    total_episodes_target = args_cli.num_episodes
    completed_episodes = 0

    # per-env accumulators (reset when that env's episode ends)
    ep_returns = torch.zeros(num_envs, device=unwrapped.device)
    ep_lengths = torch.zeros(num_envs, device=unwrapped.device, dtype=torch.long)
    ep_reward_terms = {name: torch.zeros(num_envs, device=unwrapped.device) for name in reward_term_names}

    # aggregate stats across finished episodes
    all_returns: list[float] = []
    all_lengths: list[int] = []
    termination_counts: dict[str, int] = defaultdict(int)
    timeout_count = 0
    success_count = 0
    reward_term_sums: dict[str, list[float]] = {name: [] for name in reward_term_names}

    # extras-log keys seen across episodes (scalars injected by reward terms)
    extras_log_sums: dict[str, list[float]] = defaultdict(list)

    # -- evaluation loop ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Evaluating for {total_episodes_target} episodes  |  {num_envs} parallel envs")
    print(f"{'='*70}\n")

    obs = env.get_observations()
    start_time = time.time()

    while completed_episodes < total_episodes_target and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            policy_nn.reset(dones)

        ep_returns += rewards
        ep_lengths += 1

        # accumulate per-term rewards if the manager exposes step_reward
        if has_reward_manager:
            step_reward = unwrapped.reward_manager._step_reward  # (num_envs, num_terms)
            for idx, name in enumerate(reward_term_names):
                ep_reward_terms[name] += step_reward[:, idx]

        # find which envs finished this step
        done_mask = dones.bool()
        done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)

        if done_ids.numel() > 0:
            for env_id in done_ids.tolist():
                if completed_episodes >= total_episodes_target:
                    break

                # record episode stats
                all_returns.append(ep_returns[env_id].item())
                all_lengths.append(ep_lengths[env_id].item())

                for name in reward_term_names:
                    reward_term_sums[name].append(ep_reward_terms[name][env_id].item())

                # determine termination cause
                if has_termination_manager:
                    for t_name in term_names:
                        term_val = unwrapped.termination_manager.get_term(t_name)
                        if term_val[env_id]:
                            termination_counts[t_name] += 1

                if extras.get("time_outs") is not None and extras["time_outs"][env_id]:
                    timeout_count += 1

                if success_term:
                    term_val = unwrapped.termination_manager.get_term(success_term)
                    if term_val[env_id]:
                        success_count += 1

                completed_episodes += 1

                # reset per-env accumulators
                ep_returns[env_id] = 0.0
                ep_lengths[env_id] = 0
                for name in reward_term_names:
                    ep_reward_terms[name][env_id] = 0.0

        # collect extras["log"] scalars (these are per-step averages across envs)
        if "log" in extras:
            for key, val in extras["log"].items():
                if isinstance(val, (int, float)):
                    extras_log_sums[key].append(val)

        # periodic progress
        if completed_episodes > 0 and completed_episodes % max(1, total_episodes_target // 10) == 0:
            elapsed = time.time() - start_time
            print(
                f"  [{completed_episodes}/{total_episodes_target}] episodes  "
                f"| mean return: {sum(all_returns) / len(all_returns):.3f}  "
                f"| elapsed: {elapsed:.1f}s"
            )

    elapsed_total = time.time() - start_time

    # -- report -------------------------------------------------------------------------
    n = len(all_returns)
    if n == 0:
        print("[WARN] No episodes completed.")
        env.close()
        return

    returns_t = torch.tensor(all_returns)
    lengths_t = torch.tensor(all_lengths, dtype=torch.float)
    step_dt = unwrapped.step_dt
    max_ep_len = unwrapped.max_episode_length

    print(f"\n{'='*70}")
    print(f"  EVALUATION RESULTS  ({n} episodes, {elapsed_total:.1f}s)")
    print(f"{'='*70}")
    print(f"  Checkpoint : {resume_path}")
    print(f"  Num envs   : {num_envs}")
    print(f"  Step dt    : {step_dt:.4f}s   |  Max episode steps: {max_ep_len}")

    print(f"\n  --- Episode Returns ---")
    print(f"    Mean   : {returns_t.mean().item():.4f}")
    print(f"    Std    : {returns_t.std().item():.4f}")
    print(f"    Min    : {returns_t.min().item():.4f}")
    print(f"    Max    : {returns_t.max().item():.4f}")

    print(f"\n  --- Episode Length (steps) ---")
    print(f"    Mean   : {lengths_t.mean().item():.1f}")
    print(f"    Std    : {lengths_t.std().item():.1f}")
    print(f"    Min    : {lengths_t.min().item():.0f}")
    print(f"    Max    : {lengths_t.max().item():.0f}")

    if success_term:
        print(f"\n  --- Success Rate (term: '{success_term}') ---")
        print(f"    Success: {success_count}/{n}  ({100.0 * success_count / n:.1f}%)")

    print(f"\n  --- Termination Breakdown ---")
    print(f"    Timeouts       : {timeout_count}/{n}  ({100.0 * timeout_count / n:.1f}%)")
    for t_name in term_names:
        cnt = termination_counts.get(t_name, 0)
        print(f"    {t_name:30s}: {cnt}/{n}  ({100.0 * cnt / n:.1f}%)")

    if reward_term_names:
        print(f"\n  --- Mean Episodic Reward Terms ---")
        for name in reward_term_names:
            vals = reward_term_sums[name]
            mean_val = sum(vals) / len(vals) if vals else 0.0
            print(f"    {name:40s}: {mean_val:.6f}")

    if extras_log_sums:
        print(f"\n  --- Mean Extras/Log Metrics (per-step averages) ---")
        for key in sorted(extras_log_sums.keys()):
            # skip Episode_Reward/Episode_Termination keys (already covered above)
            if key.startswith("Episode_Reward/") or key.startswith("Episode_Termination/"):
                continue
            vals = extras_log_sums[key]
            mean_val = sum(vals) / len(vals) if vals else 0.0
            print(f"    {key:50s}: {mean_val:.6f}")

    print(f"\n{'='*70}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
