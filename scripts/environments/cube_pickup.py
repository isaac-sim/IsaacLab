# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate reset states, prove scripted pickup, or evaluate sparse pickup from ordinary table starts.

Train through the standard runner::

    uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Cube-Pickup-Franka --num_envs 1024

Evaluate a trained policy from home with held-out cube positions::

    uv run python scripts/environments/cube_pickup.py --mode evaluate --checkpoint model.pt \
        --num_envs 64 --episodes 512 --seed 10000 --output evaluation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.app.logging_utils import configure_console_logging
from isaaclab.utils import to_dict
from isaaclab.utils.math import combine_frame_transforms

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.cube_pickup import mdp
from isaaclab_tasks.utils import resolve_task_config

TASK_NAME = "IsaacContrib-Cube-Pickup-Franka"


def main() -> None:
    """Run the bounded generation, scripted-control, or policy-evaluation probe."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("generate", "scripted", "evaluate"), required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--bank_size", type=int, default=256)
    parser.add_argument("--bank_output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args, overrides = parser.parse_known_args()
    if args.mode == "evaluate" and args.checkpoint is None:
        parser.error("--checkpoint is required for policy evaluation")
    if min(args.num_envs, args.bank_size, args.episodes, args.steps) < 1:
        parser.error("environment, bank, episode and step counts must be positive")
    if args.mode == "generate":
        bank_path = args.bank_output or args.output.with_suffix(".pt")
        if bank_path.resolve() == args.output.resolve():
            parser.error("the reset-state bank and --output report must use different paths")
        if bank_path.resolve() == bank_path.with_suffix(".json").resolve():
            parser.error("the reset-state bank must use a different path from its .json metadata")
    elif args.mode == "evaluate" and args.checkpoint.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the policy checkpoint")

    configure_console_logging(vars(args))
    cfg, agent_cfg = resolve_task_config(TASK_NAME, "rsl_rl_cfg_entry_point", overrides=overrides)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.reset_buffer.size = args.bank_size
    cfg.reset_buffer.seed = args.seed
    cfg.reset_mode = "adaptive" if args.mode == "generate" else "table"
    if args.device:
        cfg.sim.device = args.device
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, args):
        env = gym.make(TASK_NAME, cfg=cfg)
        try:
            task = env.unwrapped
            if args.mode == "generate":
                validation = _validate_replay(env)
                task.save_reset_buffer(bank_path)
                report = {**task.generation_report, "bank_path": str(bank_path), "replay_validation": validation}
            else:
                report = _rollout(env, agent_cfg, args)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            if args.mode != "generate":
                if not report["all_finite"] or report["completed_episodes"] < args.episodes:
                    raise RuntimeError(
                        "The rollout did not finish the requested finite episodes; see the output report."
                    )
                if args.mode == "scripted" and report["success_rate"] < 0.9:
                    raise RuntimeError(
                        "Scripted pickup succeeded on fewer than 90 percent of episodes; see the report."
                    )
        finally:
            env.close()


def _validate_replay(env: gym.Env) -> dict[str, bool]:
    """Exercise actual partial bank replay, including isolation and post-reset steps."""
    task = env.unwrapped
    observations, _ = env.reset()
    if not torch.isfinite(observations["policy"]).all():
        raise RuntimeError("Reset produced non-finite observations.")
    success_cfg = task.termination_manager.get_term_cfg("success")
    if success_cfg.func(task, **success_cfg.params).any():
        raise RuntimeError("A reset state already satisfies the grasped-cube goal.")
    before = task.scene.get_state(is_relative=True)
    split = max(1, task.num_envs // 2)
    task.reset(env_ids=torch.arange(split, device=task.device))
    if success_cfg.func(task, **success_cfg.params).any():
        raise RuntimeError("Partial replay produced an already successful reset state.")
    after = task.scene.get_state(is_relative=True)
    for kind, assets in before.items():
        for name, fields in assets.items():
            for field, value in fields.items():
                torch.testing.assert_close(value[split:], after[kind][name][field][split:], rtol=1.0e-5, atol=1.0e-6)
    for _ in range(5):
        observations, reward, _, _, _ = env.step(torch.zeros((task.num_envs, 7), device=task.device))
        if not torch.isfinite(observations["policy"]).all() or not torch.isfinite(reward).all():
            raise RuntimeError("Bank replay produced non-finite observations or rewards after stepping.")
    return {"finite_reset_and_steps": True, "partial_reset_isolated": True, "no_initial_success": True}


def _rollout(env: gym.Env, agent_cfg, args: argparse.Namespace) -> dict:
    """Count completed full-start attempts without using the reset bank."""
    task = env.unwrapped
    if args.mode == "evaluate":
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        agent_cfg.device = task.device
        runner = OnPolicyRunner(wrapped, to_dict(agent_cfg), log_dir=None, device=task.device)
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=task.device)
        observations = wrapped.get_observations()
    else:
        env.reset()
        elapsed = torch.zeros(task.num_envs, device=task.device)
        cube_start = mdp.object_position(task).clone()
        orientation = cube_start.new_tensor((1.0, 0.0, 0.0, 0.0)).expand(task.num_envs, -1)

    counts = torch.zeros(2, dtype=torch.long, device=task.device)
    remaining = torch.full((task.num_envs,), args.episodes // task.num_envs, dtype=torch.long, device=task.device)
    remaining[: args.episodes % task.num_envs] += 1
    finite = torch.ones((), dtype=torch.bool, device=task.device)
    maximum_height = torch.zeros((), device=task.device)
    with torch.inference_mode():
        for step in range(args.steps):
            if args.mode == "evaluate":
                observations, reward, done, _ = wrapped.step(policy(observations))
                done = done.bool()
                policy.reset(done)
            else:
                target = torch.cat((cube_start, orientation), dim=-1)
                target[:, 2] += torch.where(elapsed < 5.0, 0.10, 0.0)
                goal = task.command_manager.get_command("object_pose")[:, :3]
                goal, _ = combine_frame_transforms(
                    task.scene["robot"].data.root_pos_w.torch,
                    task.scene["robot"].data.root_quat_w.torch,
                    goal,
                )
                goal -= task.scene.env_origins
                target[:, :3] = torch.where((elapsed >= 7.6)[:, None], goal, target[:, :3])
                action = task.scripted_action(target, close=False)
                action[:, 6] = torch.where(elapsed >= 7.0, -1.0, 1.0)
                observations, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated

            # Same-step auto-reset restores the cube before this read. The successful terminal
            # transition remains cached in the termination manager until the next step.
            finite &= torch.isfinite(observations["policy"]).all() & torch.isfinite(reward).all()
            counted = done & (remaining > 0)
            counts += torch.stack((counted.sum(), (task.termination_manager.get_term("success") & counted).sum()))
            remaining.sub_(counted.long())
            cube_position = mdp.object_position(task)
            maximum_height = torch.fmax(maximum_height, cube_position[:, 2].max())
            if args.mode == "scripted":
                elapsed += task.step_dt
                elapsed[done] = 0.0
                cube_start[done] = cube_position[done]
            if not (remaining.any() & finite):
                break
    completed, successes = counts.tolist()
    return {
        "mode": args.mode,
        "task": TASK_NAME,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "seed": args.seed,
        "reset_mode": "table",
        "goal_position_robot_frame_m": task.command_manager.get_command("object_pose")[0, :3].tolist(),
        "goal_threshold_m": task.cfg.terminations.success.params["threshold"],
        "num_envs": task.num_envs,
        "steps": step + 1,
        "completed_episodes": completed,
        "successful_episodes": successes,
        "success_rate": successes / completed if completed else 0.0,
        "all_finite": bool(finite),
        "maximum_observed_cube_height_m": float(maximum_height),
    }


if __name__ == "__main__":
    main()
