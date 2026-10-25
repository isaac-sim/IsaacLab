# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect a registered task's native NeRD transitions into an HDF5 dataset."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.managers import DatasetExportMode
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--action_scale", type=float, default=1.5, help="Uniform normalized action bound.")
    parser.add_argument("--seed", type=int, default=42)
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser)
    if args.num_envs < 1 or args.steps < 1 or args.action_scale <= 0:
        parser.error("World/step counts and action scale must be positive.")
    path = args.output.resolve()
    if path.suffix != ".hdf5":
        parser.error("--output must end in .hdf5.")
    if path.exists() or path.with_suffix(".json").exists():
        parser.error("The output dataset already exists; choose another path.")
    cfg, _ = resolve_task_config(args.task, "", overrides=overrides)
    if cfg.recorders.transition is None:
        parser.error("Choose a task with a configured NeRD transition recorder, such as a -Explore task.")
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.recorders.transition.record_to_episodes = True
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    cfg.recorders.dataset_export_dir_path = str(path.parent)
    cfg.recorders.dataset_filename = path.stem
    cfg.recorders.export_in_close = True
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(str(path.with_suffix(".yaml")), cfg)
    torch.manual_seed(args.seed)
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg).unwrapped
        try:
            env.reset()
            capture = env.recorder_manager.get_term("transition")
            metadata = {
                "task": args.task,
                "seed": args.seed,
                "action_scale": args.action_scale,
                "dynamics": capture.dynamics.descriptor,
            }
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            with torch.inference_mode():
                for _ in range(args.steps):
                    actions = torch.empty(env.num_envs, env.action_manager.total_action_dim, device=env.device)
                    env.step(actions.uniform_(-args.action_scale, args.action_scale))
        finally:
            env.close()
    print(f"Recorded native transitions to {path}")


if __name__ == "__main__":
    main()
