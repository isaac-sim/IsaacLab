# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record ANYmal-D flat-ground episodes stepped by MuJoCo Warp as a PhysicsNeMo trajectory dataset."""

import argparse
from pathlib import Path

import torch
from actions import make_action_source
from isaaclab_physicsnemo import TrajectoryDatasetRecorder
from isaaclab_physicsnemo.tasks import ANYMAL_D_REMOVE_ROOT_TRANSLATION, AnymalDFlatTeacherEnvCfg

from isaaclab.envs import ManagerBasedRLEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Dataset directory.")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1500, help="Environment steps per environment.")
    parser.add_argument("--policy", type=Path, default=None, help="Optional exported TorchScript policy.")
    parser.add_argument("--action-scale", type=float, default=0.5, help="Standard deviation of the exploration.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replace", action="store_true", help="Replace an existing dataset.")
    args = parser.parse_args()

    cfg = AnymalDFlatTeacherEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        env.reset(seed=args.seed)
        actions = make_action_source(env, policy_path=args.policy, scale=args.action_scale, seed=args.seed)
        dones = None
        with TrajectoryDatasetRecorder(
            env,
            args.output,
            scenario_id="isaaclab-anymal-d-flat-mjwarp",
            remove_root_translation=ANYMAL_D_REMOVE_ROOT_TRANSLATION,
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
