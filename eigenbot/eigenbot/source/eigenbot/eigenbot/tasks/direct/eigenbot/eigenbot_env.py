# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Eigenbot hexapod locomotion environment for Isaac Lab.

Ported from legged_gym (Isaac Gym) LeggedRobot / EigenbotRoughCfg.
"""

from __future__ import annotations

import math
import os
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply, wrap_to_pi

from .eigenbot_env_cfg import (
    EigenbotEnvCfg,
    FEET_BODIES,
    PENALIZE_CONTACT_BODIES,
    TERMINATE_CONTACT_BODIES,
    N_PROPRIO,
    N_SCAN,
    HISTORY_LEN,
)

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets")
_USDZ_PATH = os.path.join(_ASSETS_DIR, "eigenbot_new.usdz")


def _euler_from_quat_wxyz(quat: torch.Tensor):
    """Convert (w,x,y,z) quaternion to (roll, pitch, yaw) Euler angles."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)
    t2 = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)
    return roll, pitch, yaw


class EigenbotEnv(DirectRLEnv):
    """Full RL environment for Eigenbot hexapod locomotion.
    """

    cfg: EigenbotEnvCfg

    def __init__(self, cfg: EigenbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Cache frequently used config values
        self._dt = self.cfg.sim.dt * self.cfg.decimation  # policy dt
        self._action_scale = self.cfg.action_scale
        self._p_gain = 20.0
        self._d_gain = 0.5

        # Resolve body indices for contacts
        self._resolve_body_indices()

        # Initialize all state buffers
        self._init_buffers()

        # Prepare reward function dispatch
        self._prepare_reward_functions()

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # Terrain (flat plane by default, or generated rough terrain)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)

        # Height scanner (132-ray grid for terrain observation)
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Body index resolution
    # ------------------------------------------------------------------
    def _resolve_body_indices(self):
        """Resolve body names to indices for contact force lookups."""
        # Feet indices (order matches FEET_BODIES list)
        self.feet_indices, _ = self.robot.find_bodies(FEET_BODIES)
        self.feet_indices = torch.tensor(self.feet_indices, dtype=torch.long, device=self.device)

        # Penalized contact body indices
        self.penalised_contact_indices, _ = self.robot.find_bodies(PENALIZE_CONTACT_BODIES)
        self.penalised_contact_indices = torch.tensor(
            self.penalised_contact_indices, dtype=torch.long, device=self.device
        )

        # Termination contact body indices
        self.termination_contact_indices, _ = self.robot.find_bodies(TERMINATE_CONTACT_BODIES)
        self.termination_contact_indices = torch.tensor(
            self.termination_contact_indices, dtype=torch.long, device=self.device
        )

        self.num_bodies = self.robot.num_bodies

    # ------------------------------------------------------------------
    # Buffer initialization
    # ------------------------------------------------------------------
    def _init_buffers(self):
        """Initialize all torch buffers for tracking state."""
        ne = self.num_envs
        nj = self.cfg.action_space
        nf = len(FEET_BODIES)
        dev = self.device

        # Joint state caches
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.num_joints = nj

        # Default joint positions (from articulation data)
        self.default_dof_pos = self.robot.data.default_joint_pos  # (ne, nj)

        # Actions
        self.actions = torch.zeros(ne, nj, device=dev)
        self.last_actions = torch.zeros(ne, nj, device=dev)
        self.last_dof_vel = torch.zeros(ne, nj, device=dev)

        # Action delay ring buffer (simulates actuator latency)
        if self.cfg.domain_rand.action_delay:
            buf_len = self.cfg.domain_rand.action_buf_len
            self.action_history_buf = torch.zeros(ne, buf_len, nj, device=dev)
            self.action_delay_idx = 0

        # Commands: [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]
        self.commands = torch.zeros(ne, self.cfg.commands.num_commands, device=dev)

        # Command scale for observation normalization
        self.commands_scale = torch.tensor(
            [self.cfg.normalization.obs_scales.lin_vel], device=dev
        )

        # Contact tracking
        self.last_contacts = torch.zeros(ne, nf, dtype=torch.bool, device=dev)
        self.contact_filt = torch.zeros(ne, nf, dtype=torch.bool, device=dev)
        self.feet_air_time = torch.zeros(ne, nf, device=dev)

        # Gait pattern tracking for rule_3
        n_pairs = nf // 3  # = 2 (left/right pairs)
        self.middle_liftoff_time_lpsi = torch.zeros(ne, n_pairs, device=dev)
        self.middle_liftoff_time_contra = torch.zeros(ne, n_pairs, device=dev)
        self.hind_liftoff_time_lpsi = torch.zeros(ne, n_pairs, device=dev)
        self.hind_liftoff_time_contra = torch.zeros(ne, n_pairs, device=dev)
        self.front_liftoff_time_contra = torch.zeros(ne, n_pairs, device=dev)
        self.contact_flag = torch.zeros(ne, nf, dtype=torch.bool, device=dev)

        # Observation history buffer
        self.obs_history_buf = torch.zeros(ne, HISTORY_LEN, N_PROPRIO, device=dev)

        # Computed torques (for reward computation)
        self.torques = torch.zeros(ne, nj, device=dev)

        # Forward vector for heading computation
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(ne, 1)

        # Gravity vector
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=dev).repeat(ne, 1)

        # Flat terrain flag (will be computed from height measurements)
        self.flat_tensor = torch.ones(ne, 1, dtype=torch.bool, device=dev)

        # Measured heights (flat terrain: zeros)
        self.measured_heights = torch.zeros(ne, N_SCAN, device=dev)

        # Domain randomization buffers
        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (
            (str_rng[1] - str_rng[0])
            * torch.rand(2, ne, nj, device=dev)
            + str_rng[0]
        )

        # Mass / COM randomization params (for privileged obs)
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            rand_mass = (rng[1] - rng[0]) * torch.rand(ne, 1, device=dev) + rng[0]
        else:
            rand_mass = torch.zeros(ne, 1, device=dev)

        if self.cfg.domain_rand.randomize_base_com:
            rng = self.cfg.domain_rand.added_com_range
            rand_com = (rng[1] - rng[0]) * torch.rand(ne, 3, device=dev) + rng[0]
        else:
            rand_com = torch.zeros(ne, 3, device=dev)

        self.mass_params_tensor = torch.cat([rand_mass, rand_com], dim=-1)  # (ne, 4)

        # Friction coefficients (for privileged obs)
        if self.cfg.domain_rand.randomize_friction:
            f_rng = self.cfg.domain_rand.friction_range
            self.friction_coeffs_tensor = (
                (f_rng[1] - f_rng[0]) * torch.rand(ne, 1, device=dev) + f_rng[0]
            )
        else:
            self.friction_coeffs_tensor = torch.ones(ne, 1, device=dev)

        # DOF position limits (from URDF: ±π/2 for bendy joints)
        soft = self.cfg.rewards.soft_dof_pos_limit
        lower = self.robot.data.soft_joint_pos_limits[0, :, 0]  # (nj,)
        upper = self.robot.data.soft_joint_pos_limits[0, :, 1]  # (nj,)
        mid = (lower + upper) / 2
        half_range = (upper - lower) / 2
        self.dof_pos_limits = torch.stack(
            [mid - soft * half_range, mid + soft * half_range], dim=-1
        )  # (nj, 2)

        # Push interval in steps
        self.push_interval = int(self.cfg.domain_rand.push_interval_s / self._dt)

        # Common step counter
        self.common_step_counter = 0

    # ------------------------------------------------------------------
    # Reward function dispatch
    # ------------------------------------------------------------------
    def _prepare_reward_functions(self):
        """Build list of (name, scale, function) for non-zero reward terms."""
        scales_cfg = self.cfg.rewards.scales
        self.reward_functions = []
        self.reward_names = []
        self.reward_scales = {}
        self.episode_sums = {}

        # Iterate over all fields in RewardScalesCfg
        for name in vars(scales_cfg):
            if name.startswith("_"):
                continue
            scale = getattr(scales_cfg, name)
            if scale == 0.0:
                continue
            # Scale by dt (matches legacy behavior)
            scaled = scale * self._dt
            self.reward_scales[name] = scaled
            self.episode_sums[name] = torch.zeros(
                self.num_envs, device=self.device
            )
            if name == "termination":
                continue  # handled separately after clipping
            fn_name = f"_reward_{name}"
            if hasattr(self, fn_name):
                self.reward_functions.append(getattr(self, fn_name))
                self.reward_names.append(name)
            else:
                print(f"[WARN] Reward function {fn_name} not found, skipping")

    # ------------------------------------------------------------------
    # Core env methods
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        clip_val = self.cfg.normalization.clip_actions
        self.actions = torch.clamp(actions, -clip_val, clip_val)

        if self.cfg.domain_rand.action_delay:
            buf_len = self.cfg.domain_rand.action_buf_len
            self.action_history_buf[:, self.action_delay_idx % buf_len] = self.actions
            self.action_delay_idx += 1

    def _apply_action(self) -> None:
        if self.cfg.domain_rand.action_delay:
            buf_len = self.cfg.domain_rand.action_buf_len
            oldest = self.action_delay_idx % buf_len
            delayed_actions = self.action_history_buf[:, oldest]
            targets = self._action_scale * delayed_actions + self.default_dof_pos
        else:
            targets = self._action_scale * self.actions + self.default_dof_pos
        self.robot.set_joint_position_target(targets)

        self.torques = (
            self._p_gain * (targets - self.robot.data.joint_pos)
            - self._d_gain * self.robot.data.joint_vel
        )
        torque_limit = self.cfg.rewards.torque_limit_hard
        self.torques = torch.clamp(self.torques, -torque_limit, torque_limit)

    def _get_observations(self) -> dict:
        self._post_physics_update()
        obs = self._compute_observations()
        clip_val = self.cfg.normalization.clip_observations
        obs = torch.clamp(obs, -clip_val, clip_val)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return self._compute_rewards()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._check_termination()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset DOF state with randomization
        default_pos = self.robot.data.default_joint_pos[env_ids]
        rand_scale = 0.5 + torch.rand_like(default_pos)  # [0.5, 1.5]
        joint_pos = default_pos * rand_scale
        joint_vel = torch.zeros_like(joint_pos)

        # Reset root state
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # Random base velocity [-0.5, 0.5] m/s and rad/s
        default_root_state[:, 7:13] = (
            torch.rand(len(env_ids), 6, device=self.device) - 0.5
        )

        # Write to sim
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.obs_history_buf[env_ids] = 0.0
        if self.cfg.domain_rand.action_delay:
            self.action_history_buf[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.middle_liftoff_time_lpsi[env_ids] = 0.0
        self.middle_liftoff_time_contra[env_ids] = 0.0
        self.hind_liftoff_time_lpsi[env_ids] = 0.0
        self.hind_liftoff_time_contra[env_ids] = 0.0
        self.front_liftoff_time_contra[env_ids] = 0.0

        # Resample commands
        self._resample_commands(env_ids)

        # Re-randomize motor strength on reset
        if self.cfg.domain_rand.randomize_motor:
            rng = self.cfg.domain_rand.motor_strength_range
            self.motor_strength[:, env_ids] = (
                (rng[1] - rng[0]) * torch.rand(2, len(env_ids), self.num_joints, device=self.device) + rng[0]
            )

    # ------------------------------------------------------------------
    # Post-physics update (called before observations)
    # ------------------------------------------------------------------
    def _post_physics_update(self):
        """Compute derived quantities after physics step."""
        self.common_step_counter += 1

        # Update contact filtering
        contact_forces = self.contact_sensor.data.net_forces_w  # (ne, nb, 3)
        contact = torch.norm(contact_forces[:, self.feet_indices], dim=-1) > 2.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        # Resample commands periodically
        resample_steps = int(self.cfg.commands.resampling_time / self._dt)
        if resample_steps > 0:
            env_ids = (self.episode_length_buf % resample_steps == 0).nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self._resample_commands(env_ids)

        # Heading-based angular velocity command
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.robot.data.root_quat_w, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clamp(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
            )

        # Compute height measurements (flat terrain: constant)
        self._update_height_measurements()

        # Push robots periodically
        if (
            self.cfg.domain_rand.push_robots
            and self.push_interval > 0
            and self.common_step_counter % self.push_interval == 0
        ):
            self._push_robots()

        # Store last values for next step
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.robot.data.joint_vel

    # ------------------------------------------------------------------
    # Observation computation
    # ------------------------------------------------------------------
    def _compute_observations(self) -> torch.Tensor:
        """Build the full observation vector (974 dims)."""
        obs_scales = self.cfg.normalization.obs_scales
        root_quat = self.robot.data.root_quat_w  # (ne, 4) wxyz
        base_ang_vel = self.robot.data.root_ang_vel_b  # body frame
        projected_gravity = self.robot.data.projected_gravity_b
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel

        # IMU: roll, pitch from quaternion
        roll, pitch, _ = _euler_from_quat_wxyz(root_quat)
        imu_obs = torch.stack([roll, pitch], dim=1)  # (ne, 2)

        # Delta yaw: heading error
        current_heading = self._compute_yaw()
        commanded_heading = self.commands[:, 3]
        delta_yaw = wrap_to_pi(commanded_heading - current_heading)

        # Flat terrain flag (from height measurements)
        not_flat_tensor = (torch.abs(torch.mean(self.measured_heights, dim=1)) > 0.05).unsqueeze(1)
        self.flat_tensor = ~not_flat_tensor

        # Build proprioceptive observation (72 dims)
        obs_proprio = torch.cat([
            base_ang_vel * obs_scales.ang_vel,                         # 3
            imu_obs,                                                    # 2
            delta_yaw.unsqueeze(1),                                     # 1
            projected_gravity,                                          # 3
            self.commands[:, 0:1] * self.commands_scale,                # 1
            (dof_pos - self.default_dof_pos) * obs_scales.dof_pos,      # 18
            dof_vel * obs_scales.dof_vel,                               # 18
            self.flat_tensor.float(),                                   # 1
            not_flat_tensor.float(),                                    # 1
            self.actions,                                               # 18
            self.contact_filt.float() - 0.5,                            # 6
        ], dim=-1)  # = 72

        # Height measurements (132 dims)
        root_z = self.robot.data.root_pos_w[:, 2]
        heights = torch.clamp(
            root_z.unsqueeze(1) - 0.3 - self.measured_heights, -1.0, 1.0
        )

        # Privileged explicit (9 dims)
        base_lin_vel = self.robot.data.root_lin_vel_b
        priv_explicit = torch.cat([
            base_lin_vel * obs_scales.lin_vel,
            torch.zeros_like(base_lin_vel),
            torch.zeros_like(base_lin_vel),
        ], dim=-1)

        # Privileged latent (41 dims)
        priv_latent = torch.cat([
            self.mass_params_tensor,                    # 4
            self.friction_coeffs_tensor,                # 1
            self.motor_strength[0] - 1.0,               # 18
            self.motor_strength[1] - 1.0,               # 18
        ], dim=-1)

        # Update observation history
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            obs_proprio.unsqueeze(1).expand(-1, HISTORY_LEN, -1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_proprio.unsqueeze(1),
            ], dim=1),
        )

        # Full observation
        obs = torch.cat([
            obs_proprio,                                                    # 72
            heights,                                                        # 132
            priv_explicit,                                                  # 9
            priv_latent,                                                    # 41
            self.obs_history_buf.view(self.num_envs, -1),                   # 720
        ], dim=-1)  # = 974

        return obs

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------
    def _compute_rewards(self) -> torch.Tensor:
        """Compute total reward from all active reward terms."""
        total_reward = torch.zeros(self.num_envs, device=self.device)

        for i, fn in enumerate(self.reward_functions):
            name = self.reward_names[i]
            rew = fn() * self.reward_scales[name]
            total_reward += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            total_reward = torch.clamp(total_reward, min=0.0)

        # Terminal reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            total_reward += rew
            self.episode_sums["termination"] += rew

        return total_reward

    # ------------------------------------------------------------------
    # Termination check
    # ------------------------------------------------------------------
    def _check_termination(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check for episode termination (contact-based and timeout)."""
        contact_forces = self.contact_sensor.data.net_forces_w

        # Terminate if base_link has significant contact
        terminated = torch.any(
            torch.norm(
                contact_forces[:, self.termination_contact_indices, :], dim=-1
            ) > 1.0,
            dim=1,
        )

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ------------------------------------------------------------------
    # Command generation
    # ------------------------------------------------------------------
    def _resample_commands(self, env_ids):
        """Resample velocity and heading commands for given environments."""
        if len(env_ids) == 0:
            return

        cmd_ranges = self.cfg.commands.ranges

        # Heading: offset from current heading within ±π/3
        current_heading = wrap_to_pi(self._compute_yaw())
        if self.cfg.commands.rand_heading:
            max_delta = torch.minimum(
                torch.full_like(current_heading[env_ids], math.pi) - current_heading[env_ids],
                torch.tensor(cmd_ranges.heading[1], device=self.device),
            )
            min_delta = torch.maximum(
                torch.full_like(current_heading[env_ids], -math.pi) - current_heading[env_ids],
                torch.tensor(cmd_ranges.heading[0], device=self.device),
            )
            delta = min_delta + (max_delta - min_delta) * torch.rand(
                len(env_ids), device=self.device
            )
            self.commands[env_ids, 3] = wrap_to_pi(current_heading[env_ids] + delta)
        else:
            self.commands[env_ids, 3] = current_heading[env_ids]

        # Linear velocity magnitude
        self.commands[env_ids, 0] = (
            (cmd_ranges.lin_vel_x[1] - cmd_ranges.lin_vel_x[0])
            * torch.rand(len(env_ids), device=self.device)
            + cmd_ranges.lin_vel_x[0]
        )

        # Zero out small commands
        self.commands[env_ids, :2] *= (
            torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip
        )

    def _compute_yaw(self) -> torch.Tensor:
        """Compute robot heading angle from forward vector."""
        forward = quat_apply(self.robot.data.root_quat_w, self.forward_vec)
        return torch.atan2(forward[:, 1], forward[:, 0])

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------
    def _push_robots(self):
        """Apply random velocity impulse to all robots."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        rand_vel = (torch.rand(self.num_envs, 2, device=self.device) * 2 - 1) * max_vel
        # Build 6-dim velocity tensor: [lin_vel(3), ang_vel(3)]
        lin_vel = self.robot.data.root_lin_vel_w.clone()
        ang_vel = self.robot.data.root_ang_vel_w.clone()
        lin_vel[:, 0:2] = rand_vel
        velocities = torch.cat([lin_vel, ang_vel], dim=-1)
        self.robot.write_root_velocity_to_sim(velocities)

    # ------------------------------------------------------------------
    # Height measurements
    # ------------------------------------------------------------------
    def _update_height_measurements(self):
        """Update measured heights from RayCaster sensor."""
        self.measured_heights = self._height_scanner.data.ray_hits_w[..., 2]

    # ------------------------------------------------------------------
    # USDZ visual attachment
    # ------------------------------------------------------------------
    def _attach_usdz_visual_asset(self):
        """Attach the USDZ robot mesh as a visual overlay."""
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None or not os.path.isfile(_USDZ_PATH):
            return

        robot_root = "/World/envs/env_0/Robot"
        robot_prim = stage.GetPrimAtPath(robot_root)
        if not robot_prim.IsValid():
            return

        visual_root = robot_root + "/usdz_visuals"
        stage.DefinePrim(visual_root, "Xform")
        visuals_prim = stage.GetPrimAtPath(visual_root)
        visuals_prim.GetReferences().AddReference(_USDZ_PATH)

        from pxr import UsdGeom, Gf, Usd

        xformable = UsdGeom.Xformable(visuals_prim)
        xformable.ClearXformOpOrder()
        rot_op = xformable.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ)
        rot_op.Set(Gf.Vec3f(180.0, 0.0, 0.0))
        scale_op = xformable.AddXformOp(UsdGeom.XformOp.TypeScale)
        scale_op.Set(Gf.Vec3f(0.001, 0.001, 0.001))

        for prim in Usd.PrimRange(visuals_prim):
            if prim.HasAPI("UsdPhysicsCollisionAPI"):
                prim.RemoveAPI("UsdPhysicsCollisionAPI")

    # ==================================================================
    # REWARD FUNCTIONS
    # Each returns a per-env tensor (unscaled). Scale is applied externally.
    # ==================================================================

    def _reward_tracking_goal_vel(self):
        """Reward tracking commanded velocity along the heading direction."""
        # Target heading unit vector
        heading_angle = self.commands[:, 3]
        target_x = torch.cos(heading_angle)
        target_y = torch.sin(heading_angle)
        target_vec = torch.stack([target_x, target_y], dim=1)
        norm = torch.norm(target_vec, dim=-1, keepdim=True)
        target_vec_norm = target_vec / (norm + 1e-5)

        # Current world-frame velocity
        cur_vel = self.robot.data.root_lin_vel_w[:, :2]
        vel_along_heading = torch.sum(target_vec_norm * cur_vel, dim=-1)

        lin_vel_error = torch.square(vel_along_heading - self.commands[:, 0])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_delta_yaw(self):
        """Reward for heading alignment with commanded heading."""
        current_heading = self._compute_yaw()
        return torch.exp(-torch.abs(wrap_to_pi(self.commands[:, 3] - current_heading)))

    def _reward_lin_vel_z(self):
        """Penalize vertical base velocity."""
        return torch.square(self.robot.data.root_lin_vel_b[:, 2])

    def _reward_ang_vel_xy(self):
        """Penalize roll/pitch angular velocity."""
        return torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)

    def _reward_orientation(self):
        """Penalize non-flat base orientation (on flat terrain only)."""
        gravity = self.robot.data.projected_gravity_b
        return torch.sum(torch.square(gravity[:, :2] * self.flat_tensor), dim=1)

    def _reward_torques(self):
        """Penalize joint torques."""
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """Penalize joint velocities."""
        return torch.sum(torch.square(self.robot.data.joint_vel), dim=1)

    def _reward_dof_acc(self):
        """Penalize joint accelerations."""
        dof_acc = (self.last_dof_vel - self.robot.data.joint_vel) / self._dt
        return torch.sum(torch.square(dof_acc), dim=1)

    def _reward_action_rate(self):
        """Penalize changes in actions."""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        """Penalize collisions on penalized bodies."""
        contact_forces = self.contact_sensor.data.net_forces_w
        return torch.sum(
            (torch.norm(contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1).float(),
            dim=1,
        )

    def _reward_termination(self):
        """Terminal penalty (only on actual termination, not timeout)."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.any(
            torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.termination_contact_indices, :],
                dim=-1,
            ) > 1.0,
            dim=1,
        )
        return terminated.float() * (~time_out).float()

    def _reward_dof_pos_limits(self):
        """Penalize DOF positions outside soft limits."""
        dof_pos = self.robot.data.joint_pos
        out_of_limits = -(dof_pos - self.dof_pos_limits[:, 0]).clamp(max=0.0)
        out_of_limits += (dof_pos - self.dof_pos_limits[:, 1]).clamp(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_air_time(self):
        """Reward long steps (swing time)."""
        contact = (
            self.contact_sensor.data.net_forces_w[:, self.feet_indices, 2]
            > self.cfg.rewards.contact_tresh
        )
        contact_filt = torch.logical_or(contact, self.last_contacts)

        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self._dt
        rew = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        # No reward for zero command
        rew *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew

    def _reward_stumble(self):
        """Penalize feet hitting vertical surfaces."""
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.feet_indices]
        lateral = torch.norm(contact_forces[:, :, :2], dim=2)
        vertical = torch.abs(contact_forces[:, :, 2])
        return torch.any(lateral > 5.0 * vertical, dim=1).float()

    def _reward_stand_still(self):
        """Penalize motion at zero commands."""
        dof_pos = self.robot.data.joint_pos
        return (
            torch.sum(torch.abs(dof_pos - self.default_dof_pos), dim=1)
            * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        )

    def _reward_rule_1(self):
        """Hexapod gait Rule 1: Prevent simultaneous ipsilateral liftoff.

        Rewards tripod-like coordination:
        - Middle leg up while front leg down
        - Hind leg up while middle leg down
        """
        contact = (
            self.contact_sensor.data.net_forces_w[:, self.feet_indices, 2]
            > self.cfg.rewards.contact_tresh
        )
        contact_filt = torch.logical_or(contact, self.last_contacts)

        middle = contact_filt[:, [1, 4]]
        front = contact_filt[:, [0, 3]]
        hind = contact_filt[:, [2, 5]]

        middle_up = ~middle
        rule_1_a = torch.any(middle_up & front, dim=1)

        hind_up = ~hind
        rule_1_b = torch.any(hind_up & middle, dim=1)

        return rule_1_a.float() + rule_1_b.float()

    def _reward_rule_3(self):
        """Hexapod gait Rule 3: Encourage swing initiation timing.

        Tracks liftoff timing for ipsilateral and contralateral coordination
        with exponentially decaying reward.
        """
        contact = (
            self.contact_sensor.data.net_forces_w[:, self.feet_indices, 2]
            > self.cfg.rewards.contact_tresh
        )
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.contact_flag = contact_filt

        middle = contact_filt[:, [1, 4]]
        front = contact_filt[:, [0, 3]]
        hind = contact_filt[:, [2, 5]]

        dt = self._dt

        # Ipsilateral timing
        self.middle_liftoff_time_lpsi += dt
        self.middle_liftoff_time_lpsi *= (~middle) * (~front)

        self.hind_liftoff_time_lpsi += dt
        self.hind_liftoff_time_lpsi *= (~hind) * (~middle)

        # Contralateral timing
        self.middle_liftoff_time_contra += dt
        self.middle_liftoff_time_contra *= (~middle[:, [0, 1]]) * (~middle[:, [1, 0]])

        self.front_liftoff_time_contra += dt
        self.front_liftoff_time_contra *= (~front[:, [0, 1]]) * (~front[:, [1, 0]])

        self.hind_liftoff_time_contra += dt
        self.hind_liftoff_time_contra *= (~hind[:, [0, 1]]) * (~hind[:, [1, 0]])

        c = self.cfg.rewards.exp_coeff_rule3

        hind_rew = torch.sum(
            torch.exp(c * self.hind_liftoff_time_lpsi)
            + torch.exp(c * self.hind_liftoff_time_contra),
            dim=1,
        )
        middle_rew = torch.sum(
            torch.exp(c * self.middle_liftoff_time_lpsi)
            + torch.exp(c * self.middle_liftoff_time_contra),
            dim=1,
        )
        front_rew = torch.sum(
            torch.exp(c * self.front_liftoff_time_contra),
            dim=1,
        )

        return hind_rew + middle_rew + front_rew
