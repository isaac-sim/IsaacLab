# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a task's declared exploration environment into a native NeRD dataset.

The task's model definition names the contact input and the labels that the data must hold; ``model.<field>``
overrides configure it. The dataset describes the recorded robot, and ``train.py`` builds the model on it.
"""

# ruff: noqa: E402
# Configure Warp before importing simulation modules.

import argparse
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.io import dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--num_frames", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, argv)
    # The task resolver composes the model definition in its agent slot.
    overrides = [
        f"agent.{token.removeprefix('model.')}" if token.startswith("model.") else token for token in overrides
    ]
    env_cfg, model_cfg = resolve_task_config(
        args.task,
        "nerd_cfg_entry_point",
        overrides=overrides,
        env_cfg_entry_point="nerd_recording_cfg_entry_point",
    )
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    recorders = env_cfg.recorders
    recorders.dataset_export_dir_path, recorders.dataset_filename = str(args.dataset.parent), args.dataset.name
    recorders.dynamics.contact_bodies, recorders.dynamics.prediction = model_cfg.contact_bodies, model_cfg.prediction
    with launch_simulation(env_cfg, args):
        env = ManagerBasedRLEnv(env_cfg)
        capture = env.recorder_manager.get_term("dynamics")
        try:
            action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            env.reset()
            for _ in range(args.num_frames):
                env.step(action)
            capture.close()
            dump_yaml(str(args.dataset / "params" / "env.yaml"), env_cfg)
        except BaseException:
            capture.close(complete=False)
            raise
        finally:
            env.close()


if __name__ == "__main__":
    main()
