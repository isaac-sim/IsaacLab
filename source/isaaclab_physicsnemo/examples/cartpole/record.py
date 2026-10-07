# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record task-independent Cartpole episodes stepped by MuJoCo Warp as a PhysicsNeMo trajectory dataset."""

import argparse
from pathlib import Path

import torch
from isaaclab_physicsnemo import SmoothRandomActions, TrajectoryDatasetRecorder
from isaaclab_physicsnemo.tasks import CARTPOLE_REMOVE_ROOT_TRANSLATION, CartpoleExplorationEnvCfg

from isaaclab_tasks.core.cartpole.cartpole_direct_env import CartpoleEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Dataset directory.")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200, help="Environment steps per environment.")
    parser.add_argument(
        "--action-scale",
        type=float,
        nargs=2,
        default=(0.1, 2.0),
        metavar=("LOW", "HIGH"),
        help="Range of the per-episode action standard deviation; 2.0 saturates the 400 N cart actuator.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replace", action="store_true", help="Replace an existing dataset.")
    args = parser.parse_args()

    cfg = CartpoleExplorationEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = CartpoleEnv(cfg=cfg)
    try:
        env.reset(seed=args.seed)
        actions = SmoothRandomActions(
            env.num_envs, cfg.action_space, scale=tuple(args.action_scale), device=env.device, seed=args.seed
        )
        dones = None
        with TrajectoryDatasetRecorder(
            env,
            args.output,
            scenario_id="isaaclab-cartpole-mjwarp",
            remove_root_translation=CARTPOLE_REMOVE_ROOT_TRANSLATION,
            seed=args.seed,
            replace=args.replace,
        ) as recorder:
            with torch.inference_mode():
                for _ in range(args.steps):
                    _, _, terminated, truncated, _ = env.step(actions(dones))
                    dones = terminated | truncated
                    recorder.record(dones)
        print(f"Wrote {recorder.episodes_written} episodes to {args.output.resolve()}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
