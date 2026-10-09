# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an ANYmal-C velocity-tracking PPO policy with MuJoCo Warp or learned physics, evaluated with MuJoCo Warp.

This is the FlashNeRD locomotion experiment (16,384 environments, 300 iterations): a policy trained only in
the learned environment transfers to the analytical simulator if the model reproduces the dynamics that
locomotion exercises, including the actuator response. With
``--eval-artifact`` the policy is also evaluated in the learned environment, and with ``--checkpoint`` an
existing policy is evaluated without training, for example a MuJoCo Warp policy in learned physics.
"""

import argparse
from pathlib import Path

import torch
from isaaclab_physicsnemo import apply_learned_physics
from isaaclab_physicsnemo.tasks import ANYMAL_D_CONTINUITY, AnymalCFlatTeacherEnvCfg, AnymalCTargetControlEnvCfg
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext
from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.core.velocity.config.anymal_d.agents.rsl_rl_ppo_cfg import AnymalDFlatPPORunnerCfg


def make_cfg(num_envs: int, seed: int, artifact: Path | None) -> AnymalCFlatTeacherEnvCfg:
    """MuJoCo Warp with the actuator network, or the learned model of the 20 ms control period."""
    cfg = AnymalCFlatTeacherEnvCfg() if artifact is None else AnymalCTargetControlEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    if artifact is not None:
        apply_learned_physics(cfg, artifact, continuity=ANYMAL_D_CONTINUITY)
    return cfg


def evaluate(policy: torch.nn.Module, num_envs: int, steps: int, seed: int, artifact: Path | None) -> dict[str, float]:
    """Run the policy with mean actions and report the training reward, command tracking, and falls."""
    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=make_cfg(num_envs, seed, artifact)))
    robot, commands = env.unwrapped.scene["robot"], env.unwrapped.command_manager
    returns = torch.zeros(num_envs, device=env.device)
    finished_returns, fell = [], []
    reward_sum, tracking_error_sum, observations = 0.0, 0.0, env.get_observations()
    with torch.inference_mode():
        for _ in range(steps):
            observations, rewards, dones, extras = env.step(policy(observations))
            reward_sum += rewards.mean().item()
            velocity_error = commands.get_command("base_velocity")[:, :2] - robot.data.root_lin_vel_b.torch[:, :2]
            tracking_error_sum += velocity_error.norm(dim=-1).mean().item()
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
        "linear_velocity_tracking_error_m_s": tracking_error_sum / steps,
        "finished_episodes": float(fell.numel()),
        "fall_rate": fell.float().mean().item() if fell.numel() else 0.0,
        "mean_episode_return": torch.cat(finished_returns).mean().item() if finished_returns else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, default=None, help="Train in learned physics from this artifact.")
    parser.add_argument("--output", type=Path, required=True, help="Directory receiving the policy checkpoints.")
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--eval-envs", type=int, default=256)
    parser.add_argument("--eval-steps", type=int, default=1000, help="Policy steps of each evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Evaluate this RSL-RL checkpoint, no training.")
    parser.add_argument(
        "--eval-artifact", type=Path, default=None, help="Also evaluate in learned physics. Defaults to --artifact."
    )
    args = parser.parse_args()

    agent_cfg = AnymalDFlatPPORunnerCfg()
    agent_cfg.max_iterations = args.iterations
    agent_cfg.seed = args.seed
    num_envs = args.eval_envs if args.checkpoint is not None else args.num_envs
    env = RslRlVecEnvWrapper(
        ManagerBasedRLEnv(cfg=make_cfg(num_envs, args.seed, args.artifact)), clip_actions=agent_cfg.clip_actions
    )
    runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=str(args.output), device=agent_cfg.device)
    if args.checkpoint is not None:
        runner.load(str(args.checkpoint))
    else:
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    policy = runner.get_inference_policy(device=env.device)
    env.close()
    SimulationContext.clear_instance()

    source = args.checkpoint or f"training with {'learned' if args.artifact is not None else 'MuJoCo Warp'} physics"
    evaluations: dict[str, Path | None] = {"MuJoCo Warp": None}
    if (args.eval_artifact or args.artifact) is not None:
        evaluations["learned"] = args.eval_artifact or args.artifact
    for physics, artifact in evaluations.items():
        metrics = evaluate(policy, args.eval_envs, args.eval_steps, args.seed + 1, artifact)
        print(f"Policy from {source}, evaluated with {physics} physics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4g}")


if __name__ == "__main__":
    main()
