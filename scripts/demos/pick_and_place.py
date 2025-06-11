# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard control for Isaac Lab Pick and Place.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from collections.abc import Sequence

import carb
import omni

from isaaclab_assets.robots.pick_and_place import PICK_AND_PLACE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    RigidObject,
    RigidObjectCfg,
    SurfaceGripper,
    SurfaceGripperCfg,
)
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import SPHERE_MARKER_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


@configclass
class PickAndPlaceEnvCfg(DirectRLEnvCfg):
    """Example configuration for a PickAndPlace robot using suction-cups.

    This example follows what would be typically done in a DirectRL pipeline.
    """

    # env
    decimation = 4
    episode_length_s = 240.0
    action_space = 4
    observation_space = 6
    state_space = 0
    device = "cpu"

    # Simulation cfg. Note that we are forcing the simulation to run on CPU.
    # This is because the surface gripper API is only supported on CPU backend for now.
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation, device="cpu")
    debug_vis = True

    # robot
    robot_cfg: ArticulationCfg = PICK_AND_PLACE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    x_dof_name = "x_axis"
    y_dof_name = "y_axis"
    z_dof_name = "z_axis"

    # We add a cube to pick-up
    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    # Surface Gripper, the prim_expr need to point to a unique surface gripper per environment.
    gripper = SurfaceGripperCfg(
        prim_expr="/World/envs/env_.*/Robot/picker_head/SurfaceGripper",
        max_grip_distance=0.1,
        shear_force_limit=500.0,
        coaxial_force_limit=500.0,
        retry_interval=0.2,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=12.0, replicate_physics=True)

    # reset logic
    # Initial position of the robot
    initial_x_pos_range = [-2.0, 2.0]
    initial_y_pos_range = [-2.0, 2.0]
    initial_z_pos_range = [0.0, 0.5]

    # Initial position of the cube
    initial_object_x_pos_range = [-2.0, 2.0]
    initial_object_y_pos_range = [-2.0, -0.5]
    initial_object_z_pos = 0.2

    # Target position of the cube
    target_x_pos_range = [-2.0, 2.0]
    target_y_pos_range = [2.0, 0.5]
    target_z_pos = 0.2


class PickAndPlaceEnv(DirectRLEnv):
    """Example environment for a PickAndPlace robot using suction-cups.

    This example follows what would be typically done in a DirectRL pipeline.
    Here we substitute the policy by keyboard inputs.
    """

    cfg: PickAndPlaceEnvCfg

    def __init__(self, cfg: PickAndPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Indices used to control the different axes of the gantry
        self._x_dof_idx, _ = self.pick_and_place.find_joints(self.cfg.x_dof_name)
        self._y_dof_idx, _ = self.pick_and_place.find_joints(self.cfg.y_dof_name)
        self._z_dof_idx, _ = self.pick_and_place.find_joints(self.cfg.z_dof_name)

        # joints info
        self.joint_pos = self.pick_and_place.data.joint_pos
        self.joint_vel = self.pick_and_place.data.joint_vel

        # Buffers
        self.go_to_cube = False
        self.go_to_target = False
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.instant_controls = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.permanent_controls = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)

        # Visual marker for the target
        self.set_debug_vis(self.cfg.debug_vis)

        # Sets up the keyboard callback and settings
        self.set_up_keyboard()

    def set_up_keyboard(self):
        """Sets up interface for keyboard input and registers the desired keys for control."""
        # Acquire keyboard interface
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        # Open / Close / Idle commands for gripper
        self._instant_key_controls = {
            "Q": torch.tensor([0, 0, -1]),
            "E": torch.tensor([0, 0, 1]),
            "ZEROS": torch.tensor([0, 0, 0]),
        }
        # Move up or down
        self._permanent_key_controls = {
            "W": torch.tensor([-200.0], device=self.device),
            "S": torch.tensor([100.0], device=self.device),
        }
        # Aiming manually is painful we can automate this.
        self._auto_aim_cube = "A"
        self._auto_aim_target = "D"

        # Task description:
        print("Keyboard set up!")
        print("The simulation is ready for you to try it out!")
        print("Your goal is pick up the purple cube and to drop it on the red sphere!")
        print("Use the following controls to interact with the simulation:")
        print("Press the 'A' key to have the gripper track the cube position.")
        print("Press the 'D' key to have the gripper track the target position")
        print("Press the 'W' or 'S' keys to move the gantry UP or DOWN respectively")
        print("Press 'Q' or 'E' to OPEN or CLOSE the gripper respectively")

    def _on_keyboard_event(self, event):
        """Checks for a keyboard event and assign the corresponding command control depending on key pressed."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Logic on key press
            if event.input.name == self._auto_aim_target:
                self.go_to_target = True
                self.go_to_cube = False
            if event.input.name == self._auto_aim_cube:
                self.go_to_cube = True
                self.go_to_target = False
            if event.input.name in self._instant_key_controls:
                self.go_to_cube = False
                self.go_to_target = False
                self.instant_controls[0] = self._instant_key_controls[event.input.name]
            if event.input.name in self._permanent_key_controls:
                self.go_to_cube = False
                self.go_to_target = False
                self.permanent_controls[0] = self._permanent_key_controls[event.input.name]
        # On key release, the robot stops moving
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.go_to_cube = False
            self.go_to_target = False
            self.instant_controls[0] = self._instant_key_controls["ZEROS"]

    def _setup_scene(self):
        self.pick_and_place = Articulation(self.cfg.robot_cfg)
        self.cube = RigidObject(self.cfg.cube_cfg)
        self.gripper = SurfaceGripper(self.cfg.gripper)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["pick_and_place"] = self.pick_and_place
        self.scene.rigid_objects["cube"] = self.cube
        self.scene.surface_grippers["gripper"] = self.gripper
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Store the actions
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # We use the keyboard outputs as an action.
        if self.go_to_cube:
            # Effort based proportional controller to track the cube position
            head_pos_x = self.pick_and_place.data.joint_pos[:, self._x_dof_idx[0]]
            head_pos_y = self.pick_and_place.data.joint_pos[:, self._y_dof_idx[0]]
            cube_pos_x = self.cube.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]
            cube_pos_y = self.cube.data.root_pos_w[:, 1] - self.scene.env_origins[:, 1]
            d_cube_robot_x = cube_pos_x - head_pos_x
            d_cube_robot_y = cube_pos_y - head_pos_y
            self.instant_controls[0] = torch.tensor(
                [d_cube_robot_x * 5.0, d_cube_robot_y * 5.0, 0.0], device=self.device
            )
        elif self.go_to_target:
            # Effort based proportional controller to track the target position
            head_pos_x = self.pick_and_place.data.joint_pos[:, self._x_dof_idx[0]]
            head_pos_y = self.pick_and_place.data.joint_pos[:, self._y_dof_idx[0]]
            target_pos_x = self.target_pos[:, 0]
            target_pos_y = self.target_pos[:, 1]
            d_target_robot_x = target_pos_x - head_pos_x
            d_target_robot_y = target_pos_y - head_pos_y
            self.instant_controls[0] = torch.tensor(
                [d_target_robot_x * 5.0, d_target_robot_y * 5.0, 0.0], device=self.device
            )
        # Set the joint effort targets for the picker
        self.pick_and_place.set_joint_effort_target(
            self.instant_controls[:, 0].unsqueeze(dim=1), joint_ids=self._x_dof_idx
        )
        self.pick_and_place.set_joint_effort_target(
            self.instant_controls[:, 1].unsqueeze(dim=1), joint_ids=self._y_dof_idx
        )
        self.pick_and_place.set_joint_effort_target(
            self.permanent_controls[:, 0].unsqueeze(dim=1), joint_ids=self._z_dof_idx
        )
        # Set the gripper command
        self.gripper.set_grippers_command(self.instant_controls[:, 2].unsqueeze(dim=1))

    def _get_observations(self) -> dict:
        # Get the observations
        gripper_state = self.gripper.state.clone()
        obs = torch.cat(
            (
                self.joint_pos[:, self._x_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._x_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._y_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._y_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._z_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._z_dof_idx[0]].unsqueeze(dim=1),
                self.target_pos[:, 0].unsqueeze(dim=1),
                self.target_pos[:, 1].unsqueeze(dim=1),
                gripper_state.unsqueeze(dim=1),
            ),
            dim=-1,
        )

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros_like(self.reset_terminated, dtype=torch.float32)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Dones
        self.joint_pos = self.pick_and_place.data.joint_pos
        self.joint_vel = self.pick_and_place.data.joint_vel
        # Check for time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Check if the cube reached the target
        cube_to_target_x_dist = self.cube.data.root_pos_w[:, 0] - self.target_pos[:, 0] - self.scene.env_origins[:, 0]
        cube_to_target_y_dist = self.cube.data.root_pos_w[:, 1] - self.target_pos[:, 1] - self.scene.env_origins[:, 1]
        cube_to_target_z_dist = self.cube.data.root_pos_w[:, 2] - self.target_pos[:, 2] - self.scene.env_origins[:, 2]
        cube_to_target_distance = torch.norm(
            torch.stack((cube_to_target_x_dist, cube_to_target_y_dist, cube_to_target_z_dist), dim=1), dim=1
        )
        self.target_reached = cube_to_target_distance < 0.3
        # Check if the cube is out of bounds (that is outside of the picking area)
        cube_to_origin_xy_diff = self.cube.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        cube_to_origin_x_dist = torch.abs(cube_to_origin_xy_diff[:, 0])
        cube_to_origin_y_dist = torch.abs(cube_to_origin_xy_diff[:, 1])
        self.cube_out_of_bounds = (cube_to_origin_x_dist > 2.5) | (cube_to_origin_y_dist > 2.5)

        time_out = time_out | self.target_reached
        return self.cube_out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.pick_and_place._ALL_INDICES
        # Reset the environment, this must be done first! As it releases the objects held by the grippers.
        # (And that's an operation that should be done before the gripper or the gripped objects are moved)
        super()._reset_idx(env_ids)
        num_resets = len(env_ids)

        # Set a target position for the cube
        self.target_pos[env_ids, 0] = sample_uniform(
            self.cfg.target_x_pos_range[0],
            self.cfg.target_x_pos_range[1],
            num_resets,
            self.device,
        )
        self.target_pos[env_ids, 1] = sample_uniform(
            self.cfg.target_y_pos_range[0],
            self.cfg.target_y_pos_range[1],
            num_resets,
            self.device,
        )
        self.target_pos[env_ids, 2] = self.cfg.target_z_pos

        # Set the initial position of the cube
        cube_pos = self.cube.data.default_root_state[env_ids, :7]
        cube_pos[:, 0] = sample_uniform(
            self.cfg.initial_object_x_pos_range[0],
            self.cfg.initial_object_x_pos_range[1],
            cube_pos[:, 0].shape,
            self.device,
        )
        cube_pos[:, 1] = sample_uniform(
            self.cfg.initial_object_y_pos_range[0],
            self.cfg.initial_object_y_pos_range[1],
            cube_pos[:, 1].shape,
            self.device,
        )
        cube_pos[:, 2] = self.cfg.initial_object_z_pos
        cube_pos[:, :3] += self.scene.env_origins[env_ids]
        self.cube.write_root_pose_to_sim(cube_pos, env_ids)

        # Set the initial position of the robot
        joint_pos = self.pick_and_place.data.default_joint_pos[env_ids]
        joint_pos[:, self._x_dof_idx] += sample_uniform(
            self.cfg.initial_x_pos_range[0],
            self.cfg.initial_x_pos_range[1],
            joint_pos[:, self._x_dof_idx].shape,
            self.device,
        )
        joint_pos[:, self._y_dof_idx] += sample_uniform(
            self.cfg.initial_y_pos_range[0],
            self.cfg.initial_y_pos_range[1],
            joint_pos[:, self._y_dof_idx].shape,
            self.device,
        )
        joint_pos[:, self._z_dof_idx] += sample_uniform(
            self.cfg.initial_z_pos_range[0],
            self.cfg.initial_z_pos_range[1],
            joint_pos[:, self._z_dof_idx].shape,
            self.device,
        )
        joint_vel = self.pick_and_place.data.default_joint_vel[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.pick_and_place.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.markers["sphere"].radius = 0.25
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self.target_pos + self.scene.env_origins)


def main():
    """Main function."""
    # create environment
    pick_and_place = PickAndPlaceEnv(PickAndPlaceEnvCfg())
    obs, _ = pick_and_place.reset()
    while simulation_app.is_running():
        # check for selected robots
        with torch.inference_mode():
            actions = torch.zeros((pick_and_place.num_envs, 4), device=pick_and_place.device, dtype=torch.float32)
            pick_and_place.step(actions)


if __name__ == "__main__":
    main()
    simulation_app.close()
