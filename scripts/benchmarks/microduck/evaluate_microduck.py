# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default-flat tracking and gait benchmark for MicroDuck MLP checkpoints.

The benchmark restores the action scale and policy architecture. It uses the default flat robot
and does not restore rough terrain, backlash presets, or other dynamics overrides.
"""

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import torch
import yaml
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from isaaclab.utils.io import load_yaml
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.microduck.mdp.symmetry import compute_symmetric_states
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    """Evaluate a checkpoint with fixed commands and report tracking, survival and gait metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=650)
    parser.add_argument("--num_envs", type=int, default=96)
    parser.add_argument("--action_scale", type=float)
    parser.add_argument("--asset_path")
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()
    torch.set_num_threads(4)
    labels = [
        "stand",
        "forward015",
        "forward030",
        "backward020",
        "left015",
        "right015",
        "turn_left",
        "turn_right",
        "forward_left",
        "forward_right",
        "back_left",
        "back_right",
    ]
    targets = torch.tensor(
        [
            [0, 0, 0],
            [0.15, 0, 0],
            [0.3, 0, 0],
            [-0.2, 0, 0],
            [0, 0.15, 0],
            [0, -0.15, 0],
            [0, 0, 0.8],
            [0, 0, -0.8],
            [0.2, 0, 0.8],
            [0.2, 0, -0.8],
            [-0.15, 0, 0.6],
            [-0.15, 0, -0.6],
        ],
        device="cuda:0",
    )
    if args.steps <= 50 or args.num_envs < 1 or args.num_envs % len(labels):
        parser.error("steps must exceed 50, and num_envs must be a positive multiple of 12")
    agent = load_yaml(str(args.checkpoint.parent / "params/agent.yaml"))
    saved_env = yaml.load((args.checkpoint.parent / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    if args.action_scale is None:
        args.action_scale = float(saved_env["actions"]["joint_pos"]["scale"])
    torch.manual_seed(args.seed)
    if args.action_scale <= 0:
        parser.error("action_scale must be positive")
    actor_cfg = dict(agent["actor"])
    actor_cfg.pop("class_name")
    actor = (
        MLPModel(
            TensorDict({"policy": torch.zeros(args.num_envs, 61, device="cuda:0")}, batch_size=[args.num_envs]),
            agent["obs_groups"],
            "actor",
            14,
            **actor_cfg,
        )
        .cuda()
        .eval()
    )
    actor.load_state_dict(torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)["actor_state_dict"])
    task = "IsaacContrib-Velocity-Flat-MicroDuck"
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=args.num_envs)
    cfg.play_mode()
    cfg.scene.num_envs = args.num_envs
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    cfg.seed = args.seed
    cfg.curriculum = None
    cfg.events.push_robot = None
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.resampling_time_range = (10000.0, 10000.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.head_pose.ranges = ((0.0, 0.0),) * 4
    cfg.commands.body_pose.ranges = ((0.0, 0.0),) * 6
    cfg.episode_length_s = args.steps * cfg.sim.dt * cfg.decimation + 1.0
    cfg.actions.joint_pos.scale = args.action_scale
    env = gym.make(task, cfg=cfg).unwrapped
    sensor = env.scene.sensors["contact_forces"]
    feet = [sensor.sensor_names.index(name) for name in ("ankle_left", "ankle_right")]
    groups = torch.arange(args.num_envs, device=env.device) % len(labels)
    commands = targets[groups]
    count = torch.zeros(len(labels), device=env.device)
    errors = torch.zeros(len(labels), 3, device=env.device)
    squared_errors = torch.zeros_like(errors)
    velocity_sum = torch.zeros_like(errors)
    stance = torch.zeros(len(labels), 2, device=env.device)
    stance_by_env = torch.zeros(args.num_envs, 2, device=env.device)
    count_by_env = torch.zeros(args.num_envs, device=env.device)
    touchdowns = torch.zeros_like(stance)
    swing = torch.zeros_like(stance)
    single_support = torch.zeros_like(count)
    mirror_error = torch.zeros_like(count)
    falls = torch.zeros(args.num_envs, device=env.device)
    alive = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
    with torch.inference_mode():
        obs, _ = env.reset()
        for step in range(args.steps):
            term = env.command_manager.get_term("base_velocity")
            term.vel_command_b.copy_(commands)
            term.is_standing_env[:] = False
            obs["policy"][:, 48:51] = commands
            td = TensorDict(obs, batch_size=[args.num_envs])
            action = actor(td, stochastic_output=False)
            mirrored_obs, mirrored_actions = compute_symmetric_states(None, obs=td, actions=action)
            disagreement = (
                (actor(mirrored_obs[args.num_envs :], stochastic_output=False) - mirrored_actions[args.num_envs :])
                .square()
                .mean(-1)
            )
            obs, _, terminated, truncated, _ = env.step(action)
            falls += terminated
            alive &= ~terminated
            data = env.scene["robot"].data
            v = quat_apply_inverse(yaw_quat(data.root_link_quat_w.torch), data.root_link_lin_vel_w.torch)
            actual = torch.cat((v[:, :2], data.root_link_ang_vel_w.torch[:, 2:3]), dim=-1)
            contact = sensor.data.current_contact_time.torch[:, feet] > 0
            touchdown = sensor.compute_first_contact(env.step_dt).torch[:, feet]
            last_air = sensor.data.last_air_time.torch[:, feet]
            valid = alive & (step >= 50) & ~truncated
            stance_by_env += contact * valid.unsqueeze(-1)
            count_by_env += valid
            for i in range(len(labels)):
                mask = (groups == i) & valid
                count[i] += mask.sum()
                err = actual[mask] - commands[mask]
                errors[i] += err.abs().sum(0)
                squared_errors[i] += err.square().sum(0)
                velocity_sum[i] += actual[mask].sum(0)
                stance[i] += contact[mask].sum(0)
                touchdowns[i] += touchdown[mask].sum(0)
                swing[i] += (last_air[mask] * touchdown[mask]).sum(0)
                single_support[i] += (contact[mask].sum(-1) == 1).sum()
                mirror_error[i] += disagreement[mask].sum()
        report = {
            "protocol": "microduck_fixed_commands_v1",
            "seed": args.seed,
            "steps": args.steps,
            "num_envs": args.num_envs,
            "step_dt": env.step_dt,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "action_scale": args.action_scale,
            "metrics": {},
        }
        for i, label in enumerate(labels):
            n = int(count[i])
            eligible = (args.steps - 50) * int((groups == i).sum())
            duty = (stance[i] / max(n, 1)).tolist()
            selected = (groups == i) & (count_by_env > 0)
            individual_duty = stance_by_env[selected] / count_by_env[selected].unsqueeze(-1)
            asymmetry = (individual_duty[:, 0] - individual_duty[:, 1]).abs()
            report["metrics"][label] = {
                "command": targets[i].tolist(),
                "samples": n,
                "sample_coverage": n / eligible,
                "mean_velocity": (velocity_sum[i] / n).tolist() if n else None,
                "mae": (errors[i] / n).tolist() if n else None,
                "rmse": (squared_errors[i] / n).sqrt().tolist() if n else None,
                "falls": int(falls[groups == i].sum()),
                "fraction_never_fell": float(alive[groups == i].float().mean()),
                "stance_duty": duty if n else None,
                "stance_asymmetry": float(asymmetry.mean()) if asymmetry.numel() else None,
                "stance_asymmetry_p90": float(torch.quantile(asymmetry, 0.9)) if asymmetry.numel() else None,
                "touchdowns": touchdowns[i].tolist(),
                "swing_duration": [
                    (float(swing[i, j] / touchdowns[i, j]) if touchdowns[i, j] > 0 else None) for j in range(2)
                ],
                "single_support_fraction": float(single_support[i] / n) if n else None,
                "policy_mirror_mse": float(mirror_error[i] / n) if n else None,
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("MICRODUCK_EVALUATION", json.dumps(report), flush=True)
    env.close()


if __name__ == "__main__":
    main()
