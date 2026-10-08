# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a conveyor checkpoint from reproducible, full moving-belt starts."""

from __future__ import annotations

import argparse
import json
from contextlib import closing
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import to_dict
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.entrypoints.common import resolve_published_checkpoint
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

_TASK = "IsaacContrib-Conveyor-Racetrack-Transfer-v0"


def main() -> None:
    """Run a bounded evaluation and report transfers and failures instead of training reward."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="pretrained")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    add_launcher_args(parser)
    parser.set_defaults(device="cpu", headless=True)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num_envs and --steps must be positive.")
    cfg = parse_env_cfg(_TASK, device=args.device, num_envs=args.num_envs)
    cfg.play_mode()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.events.reset_from_state_table.params["arm_joint_noise"] = 0.0
    cfg.sim.use_fabric = False
    cfg.sim.save_logs_to_file = False
    cfg.sim.physics.use_cuda_graph = torch.device(args.device).type == "cuda"
    agent = load_cfg_from_registry(_TASK, "rsl_rl_cfg_entry_point")
    agent.device = args.device
    checkpoint = (
        resolve_published_checkpoint("rsl_rl", _TASK, cfg)
        if args.checkpoint == "pretrained"
        else retrieve_file_path(args.checkpoint)
    )
    if checkpoint is None:
        raise FileNotFoundError("No published conveyor checkpoint was found; pass --checkpoint /path/to/model.pt.")
    with launch_simulation(cfg, args), closing(gym.make(_TASK, cfg=cfg).unwrapped) as env:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = OnPolicyRunner(wrapped, to_dict(agent), device=agent.device)
        runner.load(checkpoint, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        env.reset(seed=args.seed)
        observations = wrapped.get_observations()
        transfers = torch.zeros(2, device=env.device)
        failures = 0
        nonfinite_observations = 0
        with torch.inference_mode():
            for _ in range(args.steps):
                observations, _, dones, info = wrapped.step(policy(observations))
                policy.reset(dones)
                ended = int(dones.sum())
                failures += ended
                nonfinite_observations += int((~torch.isfinite(observations["policy"])).any(dim=1).sum())
                if ended:
                    for side, direction in enumerate(("left_to_right", "right_to_left")):
                        transfers[side] += info["log"][f"Metrics/transfer/{direction}_transfers"] * ended
        transfers += env.command_manager.get_term("transfer").direction_transfer_counts.sum(dim=0)
        seconds = args.steps * env.step_dt * args.num_envs
        result = {
            "task": _TASK,
            "checkpoint": checkpoint,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "simulated_environment_seconds": seconds,
            "completed_transfers": int(transfers.sum().round()),
            "left_to_right_transfers": int(transfers[0].round()),
            "right_to_left_transfers": int(transfers[1].round()),
            "failures": failures,
            "nonfinite_observations": nonfinite_observations,
            "transfers_per_environment_minute": float(transfers.sum()) * 60.0 / seconds,
        }
    report = json.dumps(result, indent=2)
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")


if __name__ == "__main__":
    main()
