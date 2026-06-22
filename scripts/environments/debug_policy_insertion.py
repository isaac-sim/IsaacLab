# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy insertion debugger — answers two questions per step:

  1. Is the policy actually trying to insert?
     → Logs raw 7-joint action vector, norm, and EEF Z motion (ΔEEF_Z).
       If ACT_NORM is large but ΔEEF_Z ≈ 0 the arm controller isn't tracking.
       If ACT_NORM is tiny, the policy is outputting near-zero actions.

  2. Is a collision preventing insertion?
     → Compares ΔEEF_Z (EEF moved down) vs ΔPLUG_Z (plug moved down).
       If EEF is descending but plug is NOT → collision blocking the plug.
       If both ≈ 0 → policy is idle or arm is stalled.

Output columns (printed every --log_interval steps):
  STEP    : simulation step number
  ACT     : sum |joint_delta| across 7 arm joints (policy activity)
  dEEF_Z  : EEF Z change since last log [mm, negative = descending]
  dPLUG_Z : plug Z change since last log [mm, negative = toward socket]
  DZ      : current plug-socket delta Z [mm] — decreasing = inserting
  VZ      : plug Z velocity [mm/s, negative = moving toward socket]
  STATUS  : INSERTING / BLOCKED / STALL / IDLE

Usage (same args as play.py):
    python scripts/environments/debug_policy_insertion.py \\
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-ROS-Inference-v0 \\
        --num_envs 1 \\
        --checkpoint logs/rsl_rl/dp_insertion_rizon4s/2026-06-11_22-49-29/model_100.pt \\
        --visualizer kit
"""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys

import gymnasium as gym
import torch
import warp as wp
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reinforcement_learning", "rsl_rl"))
import cli_args  # isort: skip  # noqa: E402

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Policy insertion debugger.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--log_interval", type=int, default=5,
                    help="Print diagnostics every N steps (default 5).")
parser.add_argument("--max_steps", type=int, default=500,
                    help="Max steps per episode (default 500).")
parser.add_argument("--num_episodes", type=int, default=3,
                    help="Number of episodes to run (default 3).")
parser.add_argument("--action_active_threshold", type=float, default=0.05,
                    help="Action norm below this → policy flagged IDLE (default 0.05).")
parser.add_argument("--block_threshold_mm", type=float, default=0.1,
                    help="ΔEEF_Z < -threshold but ΔPLUG_Z > -threshold → BLOCKED (default 0.1 mm).")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _eef_pos(env) -> torch.Tensor | None:
    robot = env.unwrapped.scene["robot"]
    try:
        idx = robot.find_bodies("flange")[0][0]
        return robot.data.body_pos_w[0, idx].cpu()
    except Exception:
        return None


def _plug_pos(env) -> torch.Tensor:
    return wp.to_torch(env.unwrapped.scene["dp_plug"].data.root_pos_w)[0].cpu()


def _plug_vel_z(env) -> float:
    return wp.to_torch(env.unwrapped.scene["dp_plug"].data.root_lin_vel_w)[0, 2].cpu().item()


def _socket_pos(env) -> torch.Tensor:
    return wp.to_torch(env.unwrapped.scene["dp_socket"].data.root_pos_w)[0].cpu()


def _classify(act_norm, d_eef_z, d_plug_z, threshold_mm, active_thresh) -> str:
    if act_norm < active_thresh:
        return "IDLE   "
    eef_descending = d_eef_z < -threshold_mm
    plug_descending = d_plug_z < -threshold_mm
    if eef_descending and plug_descending:
        return "INSERT "
    if eef_descending and not plug_descending:
        return "BLOCKED"  # arm moving but plug isn't → collision
    return "STALL  "


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # Force single env for readable per-env diagnostics
        env_cfg.scene.num_envs = 1

        task_name = args_cli.task.split(":")[-1].replace("-Play", "")
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        print(f"\n[DEBUG] Checkpoint: {resume_path}")

        env_cfg.log_dir = os.path.dirname(resume_path)
        env = gym.make(args_cli.task, cfg=env_cfg)

        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner: {agent_cfg.class_name}")

        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        hdr = f"{'STEP':>5} | {'ACT':>8} | {'joints (7)':^55} | {'dEEF_Z':>8} | {'dPLUG_Z':>8} | {'DZ':>8} | {'VZ':>8} | STATUS"
        print(f"\n{hdr}")
        print("-" * len(hdr))

        try:
            for episode in range(args_cli.num_episodes):
                print(f"\n{'='*60}  Episode {episode + 1}/{args_cli.num_episodes}  {'='*60}")

                obs = env.get_observations()
                prev_eef_z = None
                prev_plug_z = None

                for step in range(args_cli.max_steps):
                    with torch.inference_mode():
                        actions = policy(obs)
                        obs, _, dones, _ = env.step(actions)
                        if version.parse(installed_version) >= version.parse("4.0.0"):
                            policy.reset(dones)

                    if (step + 1) % args_cli.log_interval != 0:
                        # Still track positions every step for accurate deltas
                        raw = env.unwrapped
                        cur_eef = _eef_pos(raw)
                        cur_plug = _plug_pos(raw)
                        if prev_eef_z is None and cur_eef is not None:
                            prev_eef_z = cur_eef[2].item() * 1e3
                        if prev_plug_z is None:
                            prev_plug_z = cur_plug[2].item() * 1e3
                        continue

                    raw = env.unwrapped
                    act_cpu = actions[0].cpu()
                    act_norm = act_cpu.abs().sum().item()

                    cur_eef = _eef_pos(raw)
                    cur_plug = _plug_pos(raw)
                    cur_sock = _socket_pos(raw)
                    vz = _plug_vel_z(raw)

                    eef_z = cur_eef[2].item() * 1e3 if cur_eef is not None else float("nan")
                    plug_z = cur_plug[2].item() * 1e3
                    sock_z = cur_sock[2].item() * 1e3
                    dz = plug_z - sock_z

                    d_eef_z = (eef_z - prev_eef_z) if prev_eef_z is not None else 0.0
                    d_plug_z = (plug_z - prev_plug_z) if prev_plug_z is not None else 0.0
                    prev_eef_z = eef_z
                    prev_plug_z = plug_z

                    status = _classify(
                        act_norm, d_eef_z, d_plug_z,
                        args_cli.block_threshold_mm,
                        args_cli.action_active_threshold,
                    )

                    j_str = " ".join(f"{v:+.3f}" for v in act_cpu.tolist())
                    print(
                        f"  {step+1:4d} | {act_norm:8.4f} | [{j_str}] | "
                        f"{d_eef_z:+8.2f} | {d_plug_z:+8.2f} | {dz:8.2f} | {vz*1e3:+8.2f} | {status}"
                    )

                    # Annotate notable events
                    if act_norm < args_cli.action_active_threshold:
                        print(f"         *** POLICY IDLE — action norm {act_norm:.4f} ***")
                    if status == "BLOCKED":
                        print(f"         *** COLLISION — EEF moved {d_eef_z:+.2f} mm but plug moved {d_plug_z:+.2f} mm ***")

                    if dones.any():
                        print(f"  Episode terminated at step {step+1}")
                        break

        except KeyboardInterrupt:
            pass

        env.close()


if __name__ == "__main__":
    main()
