# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactively pick up a cube and place it on a target."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
import warp as wp

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Keyboard control for Isaac Lab Pick and Place.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to spawn.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
parser.add_argument(
    "--physics",
    default="isaacsim_physx",
    choices=["isaacsim_physx"],
    help="Physics backend.",
)
add_launcher_args(parser)
# surface grippers only run on CPU, and the launcher applies --device to the environment
parser.set_defaults(visualizer=["kit"], device="cpu")
args_cli = parser.parse_args()
if args_cli.num_envs < 1:
    parser.error("--num_envs must be at least 1.")
if args_cli.max_steps == 0 or args_cli.max_steps < -1:
    parser.error("--max_steps must be positive or -1.")

from isaaclab_physx.assets import SurfaceGripperCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices import Se3KeyboardCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import SPHERE_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, index_fill_, instantiate, replace
from isaaclab.utils.math import sample_uniform

from isaaclab_assets.robots.pick_and_place import PICK_AND_PLACE_CFG


@configclass
class PickAndPlaceSceneCfg(InteractiveSceneCfg):
    """Assets for the pick-and-place example."""

    robot: ArticulationCfg = replace(PICK_AND_PLACE_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.4),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    # Surface Gripper, the prim_expr need to point to a unique surface gripper per environment.
    gripper = SurfaceGripperCfg(
        prim_path="{ENV_REGEX_NS}/Robot/picker_head/SurfaceGripper",
        max_grip_distance=0.1,
        shear_force_limit=500.0,
        coaxial_force_limit=500.0,
        retry_interval=0.2,
    )
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/ground", collision_group=-1, spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )
    goal_position = replace(SPHERE_MARKER_CFG, prim_path="/Visuals/Command/goal_position")
    goal_position.markers["sphere"].radius = 0.25


@configclass
class PickAndPlaceEnvCfg(DirectRLEnvCfg):
    """Example configuration for a PickAndPlace robot using suction-cups.

    This example follows what would be typically done in a DirectRL pipeline.
    """

    # env
    decimation = 4
    episode_length_s = 240.0
    action_space = 4
    observation_space = 9
    state_space = 0

    # Simulation cfg. Surface grippers are currently only supported on CPU.
    # Surface grippers also require scene query support to function.
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        device="cpu",
        render_interval=decimation,
        use_fabric=True,
        enable_scene_query_support=True,
    )
    debug_vis = True

    scene: PickAndPlaceSceneCfg = PickAndPlaceSceneCfg(num_envs=1, env_spacing=12.0, replicate_physics=True)

    x_dof_name = "x_axis"
    y_dof_name = "y_axis"
    z_dof_name = "z_axis"

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
    target_y_pos_range = [0.5, 2.0]
    target_z_pos = 0.2


class PickAndPlaceEnv(DirectRLEnv):
    """Example environment for a PickAndPlace robot using suction-cups.

    This example follows what would be typically done in a DirectRL pipeline.
    Here we substitute the policy by keyboard inputs. The 4-D action holds the x and y efforts,
    the z effort and the gripper command (-1 open, 1 close, 0 idle).
    """

    cfg: PickAndPlaceEnvCfg

    def __init__(self, cfg: PickAndPlaceEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        self.pick_and_place, self.cube, self.gripper, self.goal_pos_visualizer = [
            self.scene[name] for name in ("robot", "cube", "gripper", "goal_position")
        ]

        # Indices used to control the different axes of the gantry
        self._x_dof_idx, _ = self.pick_and_place.find_joints(self.cfg.x_dof_name)
        self._y_dof_idx, _ = self.pick_and_place.find_joints(self.cfg.y_dof_name)
        self._z_dof_idx, _ = self.pick_and_place.find_joints(self.cfg.z_dof_name)

        # joints info
        self.joint_pos = self.pick_and_place.data.joint_pos.torch
        self.joint_vel = self.pick_and_place.data.joint_vel.torch

        # Buffers
        self.go_to_cube = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.go_to_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

        # Visual marker for the target
        self.set_debug_vis(self.cfg.debug_vis)

    def auto_aim(self, cube: bool) -> None:
        """Make all grippers track the cube (``cube=True``) or the target position."""
        self.go_to_cube[:] = cube
        self.go_to_target[:] = not cube

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Store the actions
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        xy_efforts = self.actions[:, :2].clone()
        # Manual x/y input cancels auto-aim
        manual = xy_efforts.any(dim=1)
        self.go_to_cube &= ~manual
        self.go_to_target &= ~manual
        # Effort based proportional controller to track the cube or the target position
        head_pos_xy = self.pick_and_place.data.joint_pos.torch[:, [self._x_dof_idx[0], self._y_dof_idx[0]]]
        cube_pos_xy = self.cube.data.root_pos_w.torch[:, :2] - self.scene.env_origins[:, :2]
        for mask, goal_xy in ((self.go_to_cube, cube_pos_xy), (self.go_to_target, self.target_pos[:, :2])):
            xy_efforts[mask] = (goal_xy[mask] - head_pos_xy[mask]) * 5.0

        # Set the joint effort targets for the picker
        self.pick_and_place.set_joint_effort_target_index(target=xy_efforts[:, 0:1], joint_ids=self._x_dof_idx)
        self.pick_and_place.set_joint_effort_target_index(target=xy_efforts[:, 1:2], joint_ids=self._y_dof_idx)
        self.pick_and_place.set_joint_effort_target_index(target=self.actions[:, 2:3], joint_ids=self._z_dof_idx)
        # Set the gripper command
        self.gripper.set_grippers_command(self.actions[:, 3])

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Get the observations
        gripper_state = wp.to_torch(self.gripper.state).clone()
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

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros_like(self.reset_terminated, dtype=torch.float32)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Dones
        self.joint_pos = self.pick_and_place.data.joint_pos.torch
        self.joint_vel = self.pick_and_place.data.joint_vel.torch
        # Check for time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Check if the cube reached the target
        cube_root_pos_w = self.cube.data.root_pos_w.torch
        cube_to_target_x_dist = cube_root_pos_w[:, 0] - self.target_pos[:, 0] - self.scene.env_origins[:, 0]
        cube_to_target_y_dist = cube_root_pos_w[:, 1] - self.target_pos[:, 1] - self.scene.env_origins[:, 1]
        cube_to_target_z_dist = cube_root_pos_w[:, 2] - self.target_pos[:, 2] - self.scene.env_origins[:, 2]
        cube_to_target_distance = torch.linalg.norm(
            torch.stack((cube_to_target_x_dist, cube_to_target_y_dist, cube_to_target_z_dist), dim=1), dim=1
        )
        self.target_reached = cube_to_target_distance < 0.3
        # Check if the cube is out of bounds (that is outside of the picking area)
        cube_to_origin_xy_diff = cube_root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        cube_to_origin_x_dist = torch.abs(cube_to_origin_xy_diff[:, 0])
        cube_to_origin_y_dist = torch.abs(cube_to_origin_xy_diff[:, 1])
        self.cube_out_of_bounds = (cube_to_origin_x_dist > 2.5) | (cube_to_origin_y_dist > 2.5)

        time_out = time_out | self.target_reached
        return self.cube_out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
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
        index_fill_(self.target_pos[:, 2], env_ids, self.cfg.target_z_pos)

        # Set the initial position of the cube
        cube_pos = self.cube.data.default_root_pose.torch[env_ids]
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
        self.cube.write_root_pose_to_sim_index(root_pose=cube_pos, env_ids=env_ids)

        # Set the initial position of the robot
        joint_pos = self.pick_and_place.data.default_joint_pos.torch[env_ids]
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
        joint_vel = self.pick_and_place.data.default_joint_vel.torch[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.pick_and_place.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.pick_and_place.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        self.goal_pos_visualizer.set_visibility(debug_vis)

    def _debug_vis_callback(self, event) -> None:
        """Update the target marker after a simulation step."""
        self.goal_pos_visualizer.visualize(self.target_pos + self.scene.env_origins)


def main() -> None:
    """Run the interactive surface-gripper demo."""
    env_cfg = PickAndPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    with launch_simulation(env_cfg, args_cli):
        pick_and_place = PickAndPlaceEnv(env_cfg)
        pick_and_place.reset()
        actions = torch.zeros((pick_and_place.num_envs, 4), device=pick_and_place.device)
        teleop = None
        if pick_and_place.sim.has_gui:
            teleop_cfg = Se3KeyboardCfg(pos_sensitivity=10.0, sim_device=pick_and_place.device)
            teleop = instantiate(teleop_cfg)
            teleop.add_callback("N", lambda: pick_and_place.auto_aim(cube=True))
            teleop.add_callback("M", lambda: pick_and_place.auto_aim(cube=False))
            print(teleop)
            print("Pick up the purple cube and drop it on the red sphere, in ALL environments at once.")
            print("\tW/S and A/D move the gantries, Q/E latch them UP/DOWN, K toggles the grippers.")
            print("\tN/M make the grippers track the cube/target position.")
        step_count = 0
        try:
            while pick_and_place.sim.is_running() and (args_cli.max_steps < 0 or step_count < args_cli.max_steps):
                if teleop is not None:
                    cmd = teleop.advance()
                    actions[:, :2] = cmd[:2]
                    # Latch the gantry height effort; the z joint moves up for negative effort
                    if cmd[2] != 0:
                        actions[:, 2] = -200.0 if cmd[2] > 0 else 100.0
                    # Se3Keyboard reports +1 for open, the surface gripper uses -1 for open
                    actions[:, 3] = -cmd[6]
                with torch.inference_mode():
                    pick_and_place.step(actions)
                step_count += 1
        finally:
            pick_and_place.close()


if __name__ == "__main__":
    main()
