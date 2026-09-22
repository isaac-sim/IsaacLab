# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an interactive H1 locomotion policy on rough terrain.

.. code-block:: bash

    # Usage
    uvx --from 'isaaclab[isaacsim]' isaaclab demo h1-locomotion

"""

import argparse
from importlib import metadata

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import (
    get_pretrained_checkpoint_backend_names,
    get_published_pretrained_checkpoint,
)

from isaaclab_tasks.utils import resolve_task_config

parser = argparse.ArgumentParser(
    description="This script demonstrates an interactive demo with the H1 rough terrain environment."
)
cli_args.add_rsl_rl_args(parser)
parser.add_argument("--num_envs", type=int, default=9, help="Number of H1 robots to spawn.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
parser.add_argument(
    "--physics",
    default="isaacsim_physx",
    choices=["isaacsim_physx"],
    help="Physics backend.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()
if args_cli.num_envs < 1:
    parser.error("--num_envs must be at least 1.")
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Kit modules are available only after AppLauncher starts the application.
import carb
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf

from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply

TASK = "Isaac-Velocity-Rough-H1"
RL_LIBRARY = "rsl_rl"


class H1RoughDemo:
    """Provide keyboard control for H1 robots running a locomotion policy.

    It loads a pre-trained checkpoint for the Isaac-Velocity-Rough-H1 task, trained with RSL RL
    and defines a set of keyboard commands for directing motion of selected robots.

    A robot can be selected from the scene through a mouse click. Once selected, the following
    keyboard controls can be used to control the robot:

    * UP: go forward
    * LEFT: turn left
    * RIGHT: turn right
    * DOWN: stop
    * C: switch between third-person and perspective views
    * ESC: exit current third-person view
    """

    def __init__(self) -> None:
        """Initialize the environment, policy, camera, and keyboard controls."""
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        env_cfg, _ = resolve_task_config(TASK, "", play_mode=True, overrides=(f"physics={args_cli.physics}",))
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.episode_length_s = 1000000
        env_cfg.curriculum = None
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        backend_names = get_pretrained_checkpoint_backend_names(env_cfg)
        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK, *backend_names)
        if checkpoint is None:
            raise FileNotFoundError("No published checkpoint is available for the H1 locomotion demo.")
        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        ppo_runner.load(checkpoint)
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        self.create_camera()
        self._manual_command = torch.zeros(3, device=self.device)
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id: int | None = None
        self._previous_selected_id: int | None = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)

    def create_camera(self) -> None:
        """Create the third-person camera."""
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def set_up_keyboard(self) -> None:
        """Register keyboard controls."""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        forward_speed = 1.0
        yaw_rate = 0.5
        self._key_to_control = {
            "UP": torch.tensor([forward_speed, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([0.0, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([forward_speed, 0.0, -yaw_rate], device=self.device),
            "RIGHT": torch.tensor([forward_speed, 0.0, yaw_rate], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0], device=self.device),
        }

    def _on_keyboard_event(self, event: carb.input.KeyboardEvent) -> bool:
        """Update the selected robot's command from a keyboard event."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._key_to_control:
                if self._selected_id is not None:
                    self._manual_command.copy_(self._key_to_control[event.input.name])
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if self._selected_id is not None:
                self._manual_command.zero_()
        return True

    def update_selected_object(self) -> None:
        """Update the selected robot and its camera.

        For valid robots, we enter the third-person view for that robot.
        When a new robot is selected, we reset the command of the previously selected
        to continue random commands."""

        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_path_parts = selected_prim_paths[0].split("/")
            env_name = prim_path_parts[3] if len(prim_path_parts) >= 4 else ""
            env_index = env_name.removeprefix("env_")
            if env_name.startswith("env_") and env_index.isdigit() and int(env_index) < self.env.num_envs:
                self._selected_id = int(env_index)
                if self._previous_selected_id != self._selected_id:
                    self._manual_command.zero_()
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a H1 robot")

        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])

    def apply_commands(self) -> TensorDict:
        """Apply interactive commands and return observations containing them."""
        command = self.env.unwrapped.command_manager.get_command("base_velocity")
        if self._selected_id is not None:
            command[self._selected_id].copy_(self._manual_command)
        observations = self.env.unwrapped.observation_manager.compute()
        return TensorDict(observations, batch_size=[self.env.num_envs])

    def _update_camera(self) -> None:
        """Move the third-person camera to follow the selected robot."""

        if self._selected_id is None:
            return
        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w.torch[self._selected_id]
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w.torch[self._selected_id, :]

        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)


def main() -> None:
    """Run interactive H1 policy inference."""
    demo_h1 = H1RoughDemo()
    demo_h1.env.reset()
    step_count = 0
    try:
        while simulation_app.is_running() and (args_cli.max_steps < 0 or step_count < args_cli.max_steps):
            demo_h1.update_selected_object()
            with torch.inference_mode():
                obs = demo_h1.apply_commands()
                action = demo_h1.policy(obs)
                demo_h1.env.step(action)
            step_count += 1
    finally:
        demo_h1.env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
