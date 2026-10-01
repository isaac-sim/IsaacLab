# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch the contributed H1 tablecloth task with its scripted expert.

The task owns scene construction, physics, actions, observations, resets,
rewards, and terminations. Its ``expert`` module owns the calibrated bimanual
Warp controller; this script only launches the task and runs that controller.

.. code-block:: bash

    uv run --extra importers isaaclab example tablecloth-h1
    uv run --extra importers python examples/tablecloth_h1.py \
        --visualizer none --max_steps 312

"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Run the scripted H1 tablecloth expert.")
parser.add_argument("--task", type=str, default="IsaacContrib-Tablecloth-H1", help="Task to run.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
parser.add_argument("--pull_speed", type=float, default=2.0, help="Peak X withdrawal speed [m/s].")
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()

import gymnasium as gym
import torch

from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.version import standalone_importers_available

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.tablecloth.expert import H1TableclothStateMachine
from isaaclab_tasks.contrib.tablecloth.mdp._metrics import cloth_pull_distance
from isaaclab_tasks.utils import parse_env_cfg


def _torso_pose_in_robot_frame(env) -> torch.Tensor:
    robot = env.scene["robot"]
    torso_ids, _ = robot.find_bodies("torso_link")
    torso_pos, torso_quat = subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        robot.data.body_pos_w.torch[:, torso_ids[0]],
        robot.data.body_quat_w.torch[:, torso_ids[0]],
    )
    return torch.cat((torso_pos, torso_quat), dim=-1)


def main() -> None:
    """Launch the task and run its scripted expert."""
    args_cli.require_kit = not standalone_importers_available()
    if not math.isfinite(args_cli.pull_speed) or args_cli.pull_speed <= 0.0:
        raise ValueError("--pull_speed must be finite and positive")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    success_term = env_cfg.terminations.success
    # The expert rollout should remain visible after task completion rather than auto-resetting.
    env_cfg.terminations.success = None
    env_cfg.terminations.tableware_fallen = None
    env_cfg.rewards.success = None

    with launch_simulation(cfg=env_cfg, launcher_args=args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            env.reset()
            state_machine = H1TableclothStateMachine(
                env.unwrapped.step_dt,
                _torso_pose_in_robot_frame(env.unwrapped),
                env.unwrapped.action_manager.total_action_dim,
                args_cli.pull_speed,
            )
            print("[INFO]: Setup complete. H1 tablecloth expert is ready.", flush=True)
            step = 0
            while env.unwrapped.sim.is_running() and (args_cli.max_steps < 0 or step < args_cli.max_steps):
                with torch.inference_mode():
                    _, _, terminated, truncated, _ = env.step(state_machine.compute())
                    dones = terminated | truncated
                    if dones.any():
                        state_machine.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                step += 1
            success = success_term.func(env.unwrapped, **success_term.params)
            displacements = {
                name: torch.linalg.vector_norm(
                    env.unwrapped.scene[name].data.root_pos_w.torch[:, :3]
                    - env.unwrapped.scene[name].data.default_root_pose.torch[:, :3]
                    - env.unwrapped.scene.env_origins,
                    dim=1,
                )
                for name in success_term.params["asset_names"]
            }
            maximum_displacement = torch.stack(list(displacements.values())).amax(dim=0)
            displacement_report = {name: value.tolist() for name, value in displacements.items()}
            print(
                f"[INFO]: Tablecloth trick success: {success.tolist()}, cloth travel [m]: "
                f"{cloth_pull_distance(env.unwrapped).tolist()}, "
                f"maximum tableware displacement [m]: {maximum_displacement.tolist()}, "
                f"by object: {displacement_report}",
                flush=True,
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
