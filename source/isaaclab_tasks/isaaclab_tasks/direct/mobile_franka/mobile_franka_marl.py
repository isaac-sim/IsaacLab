# Isaac Lab 2.0.1
from __future__ import annotations

from isaaclab.envs import DirectMARLEnv
import numpy as np
import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, saturate
from .mobile_franka_marl_cfg import MobileFrankaMARLCfg

class MobileFrankaEnv(DirectMARLEnv):
    cfg: MobileFrankaMARLCfg

    def __init__(self, cfg: MobileFrankaMARLCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.action_scale = 7.5
        # self.start_position_noise = 0.0
        # self.start_rotation_noise = 0.0
        # self.num_props = 4
        self.dof_vel_scale = 0.1
        self.dist_reward_scale = 2.0
        self.rot_reward_scale = 0.5
        self.around_handle_reward_scale = 10.0
        self.open_reward_scale = 7.5
        self.finger_dist_reward_scale = 100.0
        self.action_penalty_scale = 0.01
        self.finger_close_reward_scale = 10.0
        
        # self.distX_offset = 0.04
        # self.control_frequency = 120.0/2
        # self.dt=1/self.control_frequency
        self.num_franka_dofs = self.mobilefranka.num_joints
        self._num_actions = 10
        
        # buffers for franka targets
        self.franka_dof_targets = torch.zeros(
            (self.num_envs, self.num_franka_dofs), dtype=torch.float, device=self.device
        )
        self.franka_prev_targets = torch.zeros(
            (self.num_envs, self.num_franka_dofs), dtype=torch.float, device=self.device
        )
        self.franka_curr_targets = torch.zeros(
            (self.num_envs, self.num_franka_dofs), dtype=torch.float, device=self.device
        )

        # list of actuated joints 
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.mobilefranka.joint_names.index(joint_name))
        
        # list of mobile base joints
        self.actuated_mov_indices = list()
        for joint_name in cfg.mobile_base_names:
            self.actuated_mov_indices.append(self.mobilefranka.joint_names.index(joint_name))
        
        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.finger_body_names:
            self.finger_bodies.append(self.mobilefranka.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_finger = len(self.finger_bodies)

        # xy base joints
        self.xy_base_indices = list()
        for joint_name in cfg.xy_base_names:
            self.xy_base_indices.append(self.mobilefranka.joint_names.index(joint_name))

        # set the ranges for the target randomization
        self.x_lim = [-3, 3]
        self.y_lim = [-3, 3]
        self.z_lim = [0.2, 1.2]

        # joint limits
        joint_pos_limits = self.mobilefranka.root_physx_view.get_dof_limits().to(self.device)
        self.lower_limits = joint_pos_limits[..., 0]
        self.upper_limits = joint_pos_limits[..., 1]
        # print("lower_limits", self.lower_limits[1,self.actuated_mov_indices], "upper_limits", self.upper_limits[1,:])
        
        self.target_positions = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.target_positions[:, :] = torch.tensor([2.0, 0.0, 0.5], device=self.device)
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0

        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)
        # Set the default joint positions for the mobile franka
        self.mobilefranka.data.default_joint_pos[:,:] = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, -0.7856, 0.0, -2.356, 0.0, 1.572, 0.7854, 0.035, 0.035], device=self.device
        ) # base_x, base_y, base_z, joint1-7, finger1-2
        self.mobilefranka.data.default_joint_vel[:,:] = torch.tensor(
            [0.0]*self.num_franka_dofs, device=self.device
        )# base_x, base_y, base_z, joint1-7, finger1-2 (12)
        self.default_joint_pos = self.mobilefranka.data.default_joint_pos
        self.default_joint_vel = self.mobilefranka.data.default_joint_vel

    def _setup_scene(self):
        # add MobileFranka and goal object
        self.mobilefranka = Articulation(self.cfg.mobile_franka_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["mobilefranka"] = self.mobilefranka
        # self.scene.rigid_objects["target_cube"] = self.target_cube
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        # print(f"Action franka shape: {self.actions['franka'].shape}, base shape: {self.actions['base'].shape}")
        # joints
        self.franka_curr_targets[:, self.actuated_dof_indices] = scale(
            self.actions["franka"],
            self.lower_limits[:, self.actuated_dof_indices],
            self.upper_limits[:, self.actuated_dof_indices],
        )
        self.franka_curr_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.franka_curr_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.franka_prev_targets[:, self.actuated_dof_indices]
        )
        self.franka_curr_targets[:, self.actuated_dof_indices] = saturate(
            self.franka_curr_targets[:, self.actuated_dof_indices],
            self.lower_limits[:, self.actuated_dof_indices],
            self.upper_limits[:, self.actuated_dof_indices],
        )

        # Last 2 values for mobile base (x, y position)
        self.franka_curr_targets[:, self.actuated_mov_indices] = scale(
            self.actions["base"],
            self.lower_limits[:,self.actuated_mov_indices],
            self.upper_limits[:,self.actuated_mov_indices],
        )
        self.franka_curr_targets[:, self.actuated_mov_indices] = (
            self.cfg.act_moving_average * self.franka_curr_targets[:, self.actuated_mov_indices]
            + (1.0 - self.cfg.act_moving_average) * self.franka_prev_targets[:, self.actuated_mov_indices]
        )
        self.franka_curr_targets[:, self.actuated_mov_indices] = saturate(
            self.franka_curr_targets[:, self.actuated_mov_indices],
            self.lower_limits[:, self.actuated_mov_indices],
            self.upper_limits[:, self.actuated_mov_indices],
        )
        
        # save current targets
        self.franka_curr_targets[:, self.actuated_dof_indices] = self.franka_curr_targets[
            :, self.actuated_dof_indices
        ]
        self.franka_curr_targets[:, self.actuated_mov_indices] = self.franka_curr_targets[
            :, self.actuated_mov_indices
        ]
        
        # set targets
        self.mobilefranka.set_joint_position_target(
            self.franka_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        self.mobilefranka.set_joint_position_target(
            self.franka_curr_targets[:, self.actuated_mov_indices], joint_ids=self.actuated_mov_indices
        )
        
    def _get_observations(self) -> dict[str, torch.Tensor]:
        # print("joint position", self.mobilefranka.data.joint_pos)
        observations = {
            "franka": torch.cat(
                (   
                    #-------arm--------
                    # DOF positions (12)
                    unscale(self.joint_pos, self.lower_limits, self.upper_limits),
                    # DOF velocities (12)
                    self.dof_vel_scale * self.joint_vel,
                    # finger positions (3*2)
                    self.finger_body_pos.view(self.num_envs, self.num_finger*3),
                    # actions (7)
                    self.actions["franka"],
                    # actions (3)
                    self.actions["base"],
                    # positions (3)
                    self.target_positions,
                ),
                dim=-1,
            ),
            "base": torch.cat(
                (
                    #-------base--------
                    # DOF positions (3)
                    unscale(self.joint_pos, self.lower_limits, self.upper_limits),
                    # DOF velocities (3)
                    self.dof_vel_scale * self.joint_vel,
                    # finger positions (3*2)
                    self.finger_body_pos.view(self.num_envs, self.num_finger*3),
                    # actions (7)
                    self.actions["franka"],
                    # actions (3)
                    self.actions["base"],
                    # positions (3)
                    self.target_positions,
                ),
                dim=-1,
            ),
        }
        return observations

    def _get_states(self) -> torch.Tensor:
        states = torch.cat(
            (
                # DOF positions (12)
                unscale(self.joint_pos, self.lower_limits, self.upper_limits),
                # DOF velocities (12)
                self.dof_vel_scale * self.joint_vel,
                # finger positions (3*2)
                self.finger_body_pos.view(self.num_envs, self.num_finger*3),
                # actions (7)
                self.actions["franka"],
                # actions (3)
                self.actions["base"],
                # positions (3)
                self.target_positions,
            ),
            dim=-1,
        )
        return states
    
    def _get_rewards(self) -> dict[str, torch.Tensor]:
         # Calculate distance from each finger to the target separately
        finger_to_target_dists = torch.zeros((self.num_envs, self.num_finger), device=self.device)
        
        # For each finger, calculate its distance to target
        for i in range(self.num_finger):
            finger_to_target_dists[:, i] = torch.norm(
                self.finger_body_pos[:, i] - self.target_positions, p=2, dim=-1
            )
        
        # Mean distance across all fingers to target
        goal_dist = torch.mean(finger_to_target_dists, dim=1)

        rew_dist = 5 * torch.exp(-self.cfg.dist_reward_scale * goal_dist)
        
        # log reward components
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["dist_reward"] = rew_dist.mean()
        self.extras["log"]["dist_goal"] = goal_dist.mean()

        return {"franka": rew_dist, "base": rew_dist}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.joint_pos = self.mobilefranka.data.joint_pos
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self.xy_base_indices]) > self.cfg.max_base_pos, dim=1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = {agent: out_of_bounds for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.mobilefranka._ALL_INDICES
        # reset articulation and rigid body attributes
        super()._reset_idx(env_ids)
        
        # reset goals
        self._reset_target_pose(env_ids)

        # reset franka
        # delta_max = self.upper_limits[env_ids] - self.default_joint_pos[env_ids]
        # delta_min = self.lower_limits[env_ids] - self.default_joint_pos[env_ids]

        # dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_franka_dofs), device=self.device)
        # rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        # dof_pos = self.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        # dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_franka_dofs), device=self.device)
        # dof_vel = self.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        # # print("dof_pos", dof_pos[env_ids,0:3])
        # dof_pos[env_ids,0:3]=torch.tensor([0.0, 0.0, 0.0], device=self.device)
        # dof_vel[env_ids,0:3]=torch.tensor([0.0, 0.0, 0.0], device=self.device)

        # Reset franka - get default joint positions
        dof_pos = self.default_joint_pos[env_ids]
        self.franka_prev_targets[env_ids] = dof_pos
        self.franka_curr_targets[env_ids] = dof_pos
        self.franka_dof_targets[env_ids] = dof_pos
        
        # Get default root state and modify it
        default_root_state = self.mobilefranka.data.default_root_state[env_ids].clone()
        
        # Add environment origins for proper placement in each env
        default_root_state[:, :2] += self.scene.env_origins[env_ids, :2]  # Only x,y
        
        # Important: Set Z position to proper ground contact height
        # This depends on your robot's geometry - adjust as needed
        # robot_base_height = 0.05  # Height of robot base from ground
        # default_root_state[:, 2] = robot_base_height  # Set appropriate Z height
        
        # Write the corrected pose and velocity
        self.mobilefranka.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.mobilefranka.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        
        # Write joint states
        self.mobilefranka.write_joint_state_to_sim(
            self.default_joint_vel[env_ids], 
            self.default_joint_pos[env_ids], 
            env_ids=env_ids
        )
        self.mobilefranka.reset(env_ids)
        
        # No need for long sleep - robots should be stable on ground
        # Compute intermediate values for observation
        self._compute_intermediate_values()
        
    def _reset_target_pose(self, env_ids):
        # Reset goal position
        rand_pos = sample_uniform(0.0, 1.0, (len(env_ids), 3), device=self.device)
        pos = torch.zeros((len(env_ids), 3), device=self.device)
        pos[:, 0] = rand_pos[:, 0] * (self.x_lim[1] - self.x_lim[0]) + self.x_lim[0]
        pos[:, 1] = rand_pos[:, 1] * (self.y_lim[1] - self.y_lim[0]) + self.y_lim[0]
        pos[:, 2] = rand_pos[:, 2] * (self.z_lim[1] - self.z_lim[0]) + self.z_lim[0]
        self.target_positions[env_ids] = pos

        # Reset goal rotation
        rot = torch.zeros((len(env_ids), 4), dtype=torch.float, device=self.device)
        rot[:, 0] = 1.0
        self.goal_rot[env_ids] = rot
        goal_pos = self.target_positions + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)
    
    def _compute_intermediate_values(self):
        self.finger_body_pos = self.mobilefranka.data.body_pos_w[:, self.finger_bodies]
        self.finger_body_pos -= self.scene.env_origins.repeat((1, self.num_finger)).reshape(
            self.num_envs, self.num_finger, 3
        )

        self.joint_pos = self.mobilefranka.data.joint_pos
        self.joint_vel = self.mobilefranka.data.joint_vel


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)