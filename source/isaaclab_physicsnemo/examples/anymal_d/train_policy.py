# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an ANYmal-D velocity-tracking PPO policy with MuJoCo Warp or learned physics, evaluated with MuJoCo Warp.

This is the FlashNeRD locomotion experiment: a policy trained only in the learned environment transfers
to the analytical simulator if the model reproduces the dynamics that locomotion exercises.
"""

import argparse
from pathlib import Path

import torch
from isaaclab_physicsnemo import apply_learned_physics
from isaaclab_physicsnemo.tasks import ANYMAL_D_CONTINUITY, AnymalDFlatTeacherEnvCfg
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext
from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.core.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg import AnymalDFlatPPORunnerCfg


def make_cfg(num_envs: int, seed: int, artifact: Path | None) -> AnymalDFlatTeacherEnvCfg:
    cfg = AnymalDFlatTeacherEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    if artifact is not None:
        apply_learned_physics(cfg, artifact, continuity=ANYMAL_D_CONTINUITY)
    return cfg


def evaluate(policy: torch.nn.Module, num_envs: int, steps: int, seed: int) -> dict[str, float]:
    """Run the policy with mean actions in MuJoCo Warp and report the training reward and falls."""
    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=make_cfg(num_envs, seed, artifact=None)))
    returns = torch.zeros(num_envs, device=env.device)
    finished_returns, fell = [], []
    reward_sum, observations = 0.0, env.get_observations()
    with torch.inference_mode():
        for _ in range(steps):
            observations, rewards, dones, extras = env.step(policy(observations))
            reward_sum += rewards.mean().item()
            returns += rewards
            done = dones.bool()
            if done.any():
                finished_returns.append(returns[done].clone())
                fell.append(~extras["time_outs"][done])
                returns[done] = 0.0
    env.close()
    SimulationContext.clear_instance()
    fell = torch.cat(fell) if fell else torch.zeros(0, dtype=torch.bool)
    return {
        "mean_reward_per_step": reward_sum / steps,
        "finished_episodes": float(fell.numel()),
        "fall_rate": fell.float().mean().item() if fell.numel() else 0.0,
        "mean_episode_return": torch.cat(finished_returns).mean().item() if finished_returns else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, default=None, help="Train in learned physics from this artifact.")
    parser.add_argument("--output", type=Path, required=True, help="Directory receiving the policy checkpoints.")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--eval-envs", type=int, default=256)
    parser.add_argument("--eval-steps", type=int, default=1000, help="Policy steps of the MuJoCo Warp evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent_cfg = AnymalDFlatPPORunnerCfg()
    agent_cfg.max_iterations = args.iterations
    agent_cfg.seed = args.seed
    env = RslRlVecEnvWrapper(
        ManagerBasedRLEnv(cfg=make_cfg(args.num_envs, args.seed, args.artifact)), clip_actions=agent_cfg.clip_actions
    )
    runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=str(args.output), device=agent_cfg.device)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    policy = runner.get_inference_policy(device=env.device)
    env.close()
    SimulationContext.clear_instance()

    physics = "learned" if args.artifact is not None else "MuJoCo Warp"
    metrics = evaluate(policy, args.eval_envs, args.eval_steps, args.seed + 1)
    print(f"Policy trained with {physics} physics, evaluated with MuJoCo Warp:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4g}")


if __name__ == "__main__":
    main()
