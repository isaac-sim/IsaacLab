# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Portions derived from SuperMimic (MIT), Copyright (c) 2026 ETH Zurich, Chong Zhang.
# See docs/licenses/dependencies/supermimic-license.txt.

"""Native SuperMimic policy playback and complete motion-set evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import update_from_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.g1_tracking.g1_tracking_env_cfg import TransitionRecorderCfg
from isaaclab_tasks.contrib.g1_tracking.tracking import MAX_REWARD_RATE, OBS_DIM
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from ...supermimic.networks import Policy
from ..common import (
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    create_isaaclab_env,
    pre_launch_video_config,
    scoped_torch_backend_flags,
)


@torch.no_grad()
def evaluate(env, policy: Policy, real_time: bool = False) -> dict:
    """Score each clip once, retaining failed tails in the full-reference-length denominator."""
    native_env = env.unwrapped
    command = native_env.command_manager.get_term("motion")
    capture = native_env.recorder_manager.get_term("transition")
    motion = command.motion
    clip_count = len(motion.names)
    returns = torch.zeros(clip_count, device=native_env.device)
    steps = torch.zeros(clip_count, dtype=torch.long, device=native_env.device)
    failed = torch.zeros(clip_count, dtype=torch.bool, device=native_env.device)
    for offset in range(0, clip_count, native_env.num_envs):
        count = min(native_env.num_envs, clip_count - offset)
        command.cfg.evaluation_clip_offset = offset
        observations, _ = env.reset()
        finished = torch.arange(native_env.num_envs, device=native_env.device) >= count
        while not bool(finished.all()):
            started = time.perf_counter()
            action = policy(observations["policy"], observations["goal"])
            observations, _, _, _, _ = env.step(action)
            transition = capture.latest
            alive = ~finished
            returns[offset : offset + count] += torch.where(alive[:count], transition["r"][:count], 0.0)
            steps[offset : offset + count] += alive[:count]
            failed[offset : offset + count] |= alive[:count] & transition["term"][:count]
            finished |= transition["done"]
            if real_time:
                time.sleep(max(0.0, native_env.step_dt - (time.perf_counter() - started)))
    maximum_step = MAX_REWARD_RATE * native_env.step_dt * native_env.reward_manager.get_term_cfg("tracking").weight
    denominators = (motion.clip_len - 1).float() * maximum_step
    normalized = returns / denominators
    return {
        "norm_return": normalized.mean().item(),
        "norm_return_framew": (returns.sum() / denominators.sum()).item(),
        "n_failed": int(failed.sum()),
        "failed": [motion.names[i] for i in failed.nonzero().flatten().tolist()],
        "per_clip": dict(zip(motion.names, normalized.tolist(), strict=True)),
        "per_clip_steps": dict(zip(motion.names, steps.tolist(), strict=True)),
    }


def run(argv: list[str]) -> None:
    """Restore a native checkpoint's task settings and evaluate its policy with physical simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_play_args(
        parser, agent_default="supermimic_cfg_entry_point", agent_help="SuperMimic agent configuration."
    )
    parser.add_argument("--checkpoint", required=True, help="Full or policy-only native SuperMimic checkpoint.")
    parser.add_argument("--output", help="Evaluation JSON; defaults to <checkpoint>_eval.json.")
    parser.add_argument("--motion_dir", help="Evaluate a different directory of reference motions.")
    parser.add_argument("--asset_path", help="Relocated USD asset path.")
    parser.add_argument("--kinematics_path", help="Relocated matching MJCF asset path.")
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, argv)
    if overrides or args.train_env_cfg or args.frontend != "torch":
        parser.error(
            "Evaluation restores the saved task and uses its play mode; use explicit relocation/device/seed options."
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    task = args.task or checkpoint["task"]
    if task != checkpoint["task"]:
        parser.error("The selected task must match the checkpoint.")
    env_cfg, _ = resolve_task_config(task, None, overrides=[])
    env_cfg.recorders.transition = TransitionRecorderCfg()
    update_from_dict(env_cfg, checkpoint["env_cfg"])
    apply_env_overrides(args, env_cfg)
    if args.motion_dir:
        env_cfg.commands.motion.motion_dir = args.motion_dir
    if args.asset_path:
        env_cfg.scene.robot.spawn.usd_path = args.asset_path
    if args.kinematics_path:
        env_cfg.commands.motion.kinematics_path = args.kinematics_path
    if args.seed is not None:
        env_cfg.seed = args.seed
    count = sum(path.suffix in (".csv", ".npy") for path in Path(env_cfg.commands.motion.motion_dir).rglob("*"))
    if count < 1 or env_cfg.scene.num_envs < 1:
        parser.error("Evaluation needs reference motions and a positive world count.")
    env_cfg.scene.num_envs = min(env_cfg.scene.num_envs, count)
    env_cfg.play_mode()
    output = Path(args.output or Path(args.checkpoint).with_suffix("").as_posix() + "_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    pre_launch_video_config(env_cfg, args)
    apply_video_recording(env_cfg, str(output.parent), args, subdir="play", checkpoint_path=args.checkpoint)
    torch.manual_seed(env_cfg.seed)
    with (
        scoped_torch_backend_flags(
            cuda_matmul_allow_tf32=False, cudnn_allow_tf32=True, cudnn_deterministic=False, cudnn_benchmark=False
        ),
        launch_simulation(env_cfg, args),
    ):
        env = create_isaaclab_env(task, env_cfg, args, convert_marl_to_single_agent=False)
        try:
            dtype = torch.bfloat16 if checkpoint["agent_cfg"]["amp"] == "bf16" else None
            policy = Policy(
                OBS_DIM,
                73 * env_cfg.commands.motion.goal_frames,
                hist=env_cfg.observations.policy.history_length,
                amp_dtype=dtype,
            ).to(env.unwrapped.device)
            policy.load_state_dict(checkpoint["pi"])
            policy.eval()
            result = evaluate(env, policy, args.real_time)
            result.update(
                checkpoint=str(Path(args.checkpoint).resolve()), iteration=checkpoint["iteration"], seed=env_cfg.seed
            )
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                f"Completed {len(result['per_clip'])} clips: {result['n_failed']} failures, "
                f"normalized return {result['norm_return']:.6f}; {output}"
            )
        finally:
            env.close()
