# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a Cartpole PPO policy with MuJoCo Warp or learned physics and evaluate it with MuJoCo Warp.

Training in the learned environment and evaluating in the teacher environment is the downstream test
of a learned dynamics model: the policy only transfers if the model reproduces the dynamics that the
task exercises.
"""

import argparse
from pathlib import Path

import torch
from cartpole_env_cfg import CartpoleTeacherEnvCfg
from isaaclab_physicsnemo import apply_learned_physics
from rsl_rl.runners import OnPolicyRunner

from isaaclab.sim import SimulationContext
from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg import CartpolePPORunnerCfg
from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv

# The Isaac Lab Cartpole task applies each action for two physics steps.
POLICY_DECIMATION = 2


def make_cfg(num_envs: int, seed: int, artifact: Path | None) -> CartpoleTeacherEnvCfg:
    cfg = CartpoleTeacherEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.decimation = POLICY_DECIMATION
    cfg.seed = seed
    if artifact is not None:
        apply_learned_physics(cfg, artifact)
    return cfg


def evaluate(policy: torch.nn.Module, num_envs: int, steps: int, seed: int) -> dict[str, float]:
    """Run the policy in the MuJoCo Warp environment and report finished-episode statistics."""
    env = RslRlVecEnvWrapper(CartpoleEnv(cfg=make_cfg(num_envs, seed, artifact=None)))
    returns = torch.zeros(num_envs, device=env.device)
    finished_returns, finished_lengths, survived = [], [], []
    try:
        observations = env.get_observations()
        with torch.inference_mode():
            for _ in range(steps):
                observations, rewards, dones, extras = env.step(policy(observations))
                returns += rewards
                done = dones.bool()
                if done.any():
                    finished_returns.append(returns[done].clone())
                    finished_lengths.append(env.unwrapped.episode_length_buf[done].clone())
                    survived.append(extras["time_outs"][done].float())
                    returns[done] = 0.0
    finally:
        # A failed evaluation must not leave a simulation behind for the next one.
        env.close()
        SimulationContext.clear_instance()
    if not survived:
        # No episode finished within the evaluated steps.
        return {"episodes": 0.0, "success_rate": float("nan"), "mean_return": float("nan")}
    survived = torch.cat(survived)
    return {
        "episodes": float(survived.numel()),
        "success_rate": survived.mean().item(),
        "mean_return": torch.cat(finished_returns).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, default=None, help="Train in learned physics from this artifact.")
    parser.add_argument("--output", type=Path, required=True, help="Directory receiving the policy checkpoints.")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--eval-envs", type=int, default=512)
    parser.add_argument("--eval-steps", type=int, default=900, help="Policy steps of the MuJoCo Warp evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent_cfg = CartpolePPORunnerCfg()
    agent_cfg.max_iterations = args.iterations
    agent_cfg.seed = args.seed
    env = RslRlVecEnvWrapper(
        CartpoleEnv(cfg=make_cfg(args.num_envs, args.seed, args.artifact)), clip_actions=agent_cfg.clip_actions
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
