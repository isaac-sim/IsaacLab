# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an interactive H1 locomotion policy on rough terrain.

.. code-block:: bash

    # Usage
    uvx isaaclab demo h1-locomotion

"""

import argparse
import math
from importlib import metadata

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args

parser = argparse.ArgumentParser(
    description="This script demonstrates an interactive demo with the H1 rough terrain environment."
)
cli_args.add_rsl_rl_args(parser)
parser.add_argument("--num_envs", type=int, default=9, help="Number of H1 robots to spawn.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
parser.add_argument(
    "--physics", default="newton_mjwarp", choices=["isaacsim_physx", "newton_mjwarp"], help="Physics backend."
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_gl"])
args_cli = parser.parse_args()
if args_cli.num_envs < 1:
    parser.error("--num_envs must be at least 1.")
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")

import torch
from isaaclab_visualizers.newton import NewtonGLVisualizer, NewtonRTXVisualizer
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import (
    get_pretrained_checkpoint_backend_names,
    get_published_pretrained_checkpoint,
)

from isaaclab_tasks.utils import resolve_task_config

TASK = "Isaac-Velocity-Rough-H1"
RL_LIBRARY = "rsl_rl"
FORWARD_SPEED = 1.0
"""Forward velocity command while a drive key is held [m/s]."""
YAW_RATE = 0.5
"""Yaw velocity command while a turn key is held [rad/s]."""


class H1RoughDemo:
    """Provide keyboard control for H1 robots running a locomotion policy.

    It loads a pre-trained checkpoint for the Isaac-Velocity-Rough-H1 task, trained with RSL RL
    and defines a set of keyboard commands for directing motion of a selected robot. The Newton
    viewer reserves WASD, the arrow keys, Q, and E for flying its camera, so the robot is driven
    with the following keys in the viewer window:

    * N: select the next robot, cycling back to no selection
    * I: go forward
    * J: turn left
    * L: turn right
    * K: stop
    * C: toggle a third-person camera that follows the selected robot

    Unselected robots keep following random velocity commands.
    """

    def __init__(self, env_cfg: ManagerBasedRLEnvCfg) -> None:
        """Initialize the environment, policy, and keyboard controls.

        Args:
            env_cfg: Resolved H1 rough-terrain environment configuration.

        Raises:
            FileNotFoundError: If no published checkpoint matches the configured backends.
        """
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        backend_names = get_pretrained_checkpoint_backend_names(env_cfg)
        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK, *backend_names)
        if checkpoint is None:
            raise FileNotFoundError("No published checkpoint is available for the H1 locomotion demo.")
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(checkpoint)
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        self._viewers = [
            visualizer
            for visualizer in self.env.unwrapped.sim.visualizers
            if isinstance(visualizer, (NewtonGLVisualizer, NewtonRTXVisualizer))
        ]
        self._key_to_control = {
            "i": torch.tensor([FORWARD_SPEED, 0.0, 0.0], device=self.device),
            "k": torch.tensor([0.0, 0.0, 0.0], device=self.device),
            "j": torch.tensor([FORWARD_SPEED, 0.0, YAW_RATE], device=self.device),
            "l": torch.tensor([FORWARD_SPEED, 0.0, -YAW_RATE], device=self.device),
        }
        self._keys_down: set[str] = set()
        self._manual_command = torch.zeros(3, device=self.device)
        self._selected_id: int | None = None
        self._follow_camera = False
        # Follow-camera offset in the robot base frame [m].
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)
        self._frame_robots()

    def update_controls(self) -> None:
        """Poll the viewer keyboard and update the selection, command, and camera."""
        if self._key_pressed("n"):
            previous_id = self._selected_id
            next_id = 0 if previous_id is None else previous_id + 1
            self._selected_id = next_id if next_id < self.env.num_envs else None
            if previous_id is not None:
                self.env.unwrapped.command_manager.reset([previous_id])
            print(f"[INFO]: Selected robot: {self._selected_id if self._selected_id is not None else 'none'}")
        if self._key_pressed("c"):
            self._follow_camera = not self._follow_camera
            if not self._follow_camera:
                self._frame_robots()

        self._manual_command.zero_()
        for key, command in self._key_to_control.items():
            if any(viewer.is_key_down(key) for viewer in self._viewers):
                self._manual_command.copy_(command)
        if self._follow_camera:
            self._update_camera()

    def apply_commands(self) -> TensorDict:
        """Apply interactive commands and return observations containing them."""
        command = self.env.unwrapped.command_manager.get_command("base_velocity")
        if self._selected_id is not None:
            command[self._selected_id].copy_(self._manual_command)
        observations = self.env.unwrapped.observation_manager.compute()
        return TensorDict(observations, batch_size=[self.env.num_envs])

    def _key_pressed(self, key: str) -> bool:
        """Return whether a key went down since the previous poll.

        Args:
            key: Key name understood by the Newton viewer.

        Returns:
            True on the first poll after the key is pressed, False otherwise.
        """
        is_down = any(viewer.is_key_down(key) for viewer in self._viewers)
        was_down = key in self._keys_down
        if is_down:
            self._keys_down.add(key)
        else:
            self._keys_down.discard(key)
        return is_down and not was_down

    def _frame_robots(self) -> None:
        """Point the viewer camera at the whole group of robots."""
        center = self.env.unwrapped.scene.env_origins.mean(dim=0).tolist()
        scale = 2.5 * math.ceil(math.sqrt(self.env.num_envs))
        eye = (center[0] + scale, center[1] - scale, center[2] + 0.6 * scale)
        self.env.unwrapped.sim.set_camera_view(eye=eye, target=(center[0], center[1], center[2] + 0.5))

    def _update_camera(self) -> None:
        """Move the viewer camera to follow the selected robot."""
        if self._selected_id is None:
            return
        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w.torch[self._selected_id]
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w.torch[self._selected_id, :]
        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos
        target = base_pos + torch.tensor([0.0, 0.0, 0.6], device=self.device)
        self.env.unwrapped.sim.set_camera_view(eye=camera_pos.tolist(), target=target.tolist())


def main() -> None:
    """Run interactive H1 policy inference."""
    env_cfg, _ = resolve_task_config(TASK, "", play_mode=True, overrides=(f"physics={args_cli.physics}",))
    env_cfg.scene.num_envs = args_cli.num_envs
    # Place the robots in a compact grid so they share one view instead of scattering across terrain tiles.
    env_cfg.scene.terrain.use_terrain_origins = False
    env_cfg.episode_length_s = 1000000
    env_cfg.curriculum = None
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # The physics preset is already selected above; forwarding --physics would replace the
    # task's tuned solver settings with backend defaults the policy was not trained on.
    with launch_simulation(env_cfg, vars(args_cli) | {"physics": None}):
        demo_h1 = H1RoughDemo(env_cfg)
        demo_h1.env.reset()
        sim = demo_h1.env.unwrapped.sim
        step_count = 0
        try:
            while sim.is_headless_or_exist_active_visualizer() and (
                args_cli.max_steps < 0 or step_count < args_cli.max_steps
            ):
                demo_h1.update_controls()
                with torch.inference_mode():
                    obs = demo_h1.apply_commands()
                    action = demo_h1.policy(obs)
                    demo_h1.env.step(action)
                step_count += 1
        finally:
            demo_h1.env.close()


if __name__ == "__main__":
    main()
