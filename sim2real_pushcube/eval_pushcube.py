"""Evaluate a ManiSkill PushCube-v1 RGB-PPO checkpoint inside IsaacLab.

Runs the ManiSkill-trained policy (loaded from a ``.pt`` state_dict saved by
ppo_rgb.py) on the IsaacLab PushCube env in ``pushcube_isaaclab_env`` and prints
the success rate over a number of episodes.

Usage (on a machine that can run Isaac Sim):
    ./isaaclab.sh -p sim2real_pushcube/eval_pushcube.py \
        --ckpt /path/to/runs/PushCube-v1__ppo_rgb__1__<ts>/final_ckpt.pt \
        --num_envs 8 --num_eval_episodes 50 \
        --headless --enable_cameras

Notes:
- ``--enable_cameras`` is required (the policy needs the 128x128 rgb observation).
- ``--headless`` + ``--enable_cameras`` selects the headless kit WITH RTX rendering.
- Success is sticky per episode (cube reaches the goal at any step during the
  50-step episode). Reported as successes / episodes.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# --- argparse + AppLauncher MUST come before any isaaclab.* import ---
parser = argparse.ArgumentParser(description="Evaluate ManiSkill PushCube ckpt in IsaacLab")
parser.add_argument("--ckpt", type=str, required=True, help="path to a ppo_rgb.py .pt checkpoint (state_dict)")
parser.add_argument("--num_envs", type=int, default=8, help="number of parallel envs")
parser.add_argument("--num_eval_episodes", type=int, default=50, help="total episodes to evaluate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- now safe to import isaaclab / torch / local modules ---
import os
import sys

import torch

# make sibling modules (mani_skill_agent, pushcube_isaaclab_env) importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mani_skill_agent import build_agent
from pushcube_isaaclab_env import PushCubeIsaacLabEnv


def main():
    torch.manual_seed(args_cli.seed)
    device = args_cli.device

    # build env + load policy
    env = PushCubeIsaacLabEnv(num_envs=args_cli.num_envs, device=device)
    agent = build_agent(
        ckpt_path=args_cli.ckpt,
        num_envs=args_cli.num_envs,
        device=torch.device(device),
        rgb_shape=(128, 128, 3),
        state_dim=35,
        action_dim=8,
    )
    print(f"[eval] env ready ({args_cli.num_envs} envs), policy loaded from {args_cli.ckpt}")

    obs = env.reset()
    episodes_done = 0
    successes = 0
    target = args_cli.num_eval_episodes
    # generous safety bound so the loop can't run forever if episodes never end
    safety_max_steps = target * (env.max_steps + 5) + 100
    step = 0

    while episodes_done < target and step < safety_max_steps:
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)  # (N,8)
        obs, done, info = env.step(action)
        step += 1

        reset_ids = done.nonzero(as_tuple=True)[0]
        if len(reset_ids) > 0:
            # record sticky success for episodes that just ended, THEN reset (reset clears ever_success)
            successes += int(env.ever_success[reset_ids].sum().item())
            episodes_done += len(reset_ids)
            obs = env.reset(reset_ids)

        if step % 5 == 0:
            rate = successes / max(episodes_done, 1)
            print(
                f"[step {step}] episodes={episodes_done}/{target} "
                f"successes={successes} running_rate={rate:.3f}",
                flush=True,
            )

    print("\n" + "=" * 60)
    if episodes_done > 0:
        print(f"Eval result: {successes}/{episodes_done} = {successes / episodes_done:.3f} success rate")
    else:
        print("Eval result: 0 episodes completed (check that episodes terminate).")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
