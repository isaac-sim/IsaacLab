# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class-based reward terms for the gear assembly manipulation environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_mul

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer

    from .events import randomize_gear_type


class keypoint_command_error(ManagerTermBase):
    """Compute keypoint distance between current and desired poses from command.

    This class-based term uses _compute_keypoint_distance internally.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the keypoint command error term.

        Args:
            cfg: Reward term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset configuration
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("ee_frame"))
        self.command_name: str = cfg.params.get("command_name", "ee_pose")

        # Create keypoint distance computer
        self.keypoint_computer = _compute_keypoint_distance(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        keypoint_scale: float = 1.0,
        add_cube_center_kp: bool = True,
    ) -> torch.Tensor:
        """Compute keypoint distance error.

        Args:
            env: Environment instance
            command_name: Name of the command containing desired pose
            asset_cfg: Configuration of the asset to track
            keypoint_scale: Scale factor for keypoint offsets
            add_cube_center_kp: Whether to include center keypoint

        Returns:
            Mean keypoint distance tensor of shape (num_envs,)
        """
        # Extract frame transformer sensor
        asset: FrameTransformer = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)

        # Get desired pose from command
        des_pos_w = command[:, :3]
        des_quat_w = command[:, 3:7]

        # Get current pose from frame transformer
        curr_pos_w = asset.data.target_pos_source.torch[:, 0]
        curr_quat_w = asset.data.target_quat_source.torch[:, 0]

        # Compute keypoint distance
        keypoint_dist_sep = self.keypoint_computer.compute(
            current_pos=curr_pos_w,
            current_quat=curr_quat_w,
            target_pos=des_pos_w,
            target_quat=des_quat_w,
            keypoint_scale=keypoint_scale,
        )

        return keypoint_dist_sep.mean(-1)


class keypoint_command_error_exp(ManagerTermBase):
    """Compute exponential keypoint reward between current and desired poses from command.

    This class-based term uses _compute_keypoint_distance internally and applies
    exponential reward transformation.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the keypoint command error exponential term.

        Args:
            cfg: Reward term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset configuration
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("ee_frame"))
        self.command_name: str = cfg.params.get("command_name", "ee_pose")

        # Create keypoint distance computer
        self.keypoint_computer = _compute_keypoint_distance(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        kp_exp_coeffs: list[tuple[float, float]] = [(1.0, 0.1)],
        kp_use_sum_of_exps: bool = True,
        keypoint_scale: float = 1.0,
        add_cube_center_kp: bool = True,
    ) -> torch.Tensor:
        """Compute exponential keypoint reward.

        Args:
            env: Environment instance
            command_name: Name of the command containing desired pose
            asset_cfg: Configuration of the asset to track
            kp_exp_coeffs: List of (a, b) coefficient pairs for exponential reward
            kp_use_sum_of_exps: Whether to use sum of exponentials
            keypoint_scale: Scale factor for keypoint offsets
            add_cube_center_kp: Whether to include center keypoint

        Returns:
            Exponential keypoint reward tensor of shape (num_envs,)
        """
        # Extract frame transformer sensor
        asset: FrameTransformer = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)

        # Get desired pose from command
        des_pos_w = command[:, :3]
        des_quat_w = command[:, 3:7]

        # Get current pose from frame transformer
        curr_pos_w = asset.data.target_pos_source.torch[:, 0]
        curr_quat_w = asset.data.target_quat_source.torch[:, 0]

        # Compute keypoint distance
        keypoint_dist_sep = self.keypoint_computer.compute(
            current_pos=curr_pos_w,
            current_quat=curr_quat_w,
            target_pos=des_pos_w,
            target_quat=des_quat_w,
            keypoint_scale=keypoint_scale,
        )

        # Compute exponential reward
        keypoint_reward_exp = torch.zeros_like(keypoint_dist_sep[:, 0])

        if kp_use_sum_of_exps:
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += (
                    1.0 / (torch.exp(a * keypoint_dist_sep) + b + torch.exp(-a * keypoint_dist_sep))
                ).mean(-1)
        else:
            keypoint_dist = keypoint_dist_sep.mean(-1)
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += 1.0 / (torch.exp(a * keypoint_dist) + b + torch.exp(-a * keypoint_dist))

        return keypoint_reward_exp


class keypoint_entity_error(ManagerTermBase):
    """Compute keypoint distance between a RigidObject and the dynamically selected gear.

    This class-based term pre-caches gear type mapping and asset references.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the keypoint entity error term.

        Args:
            cfg: Reward term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset configuration
        self.asset_cfg_1: SceneEntityCfg = cfg.params.get("asset_cfg_1", SceneEntityCfg("factory_gear_base"))
        self.asset_1 = env.scene[self.asset_cfg_1.name]

        self._init_gear_selection(env)

        # Create keypoint distance computer
        self.keypoint_computer = _compute_keypoint_distance(cfg, env)

    def _init_gear_selection(self, env: ManagerBasedRLEnv) -> None:
        """Pre-allocate gear type mapping, index tensors, and cache gear scene assets."""
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.env_indices = torch.arange(env.num_envs, device=env.device)

        self.gear_assets = {
            "gear_small": env.scene["factory_gear_small"],
            "gear_medium": env.scene["factory_gear_medium"],
            "gear_large": env.scene["factory_gear_large"],
        }

    def _get_selected_gear_poses(self, env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve world-frame position and quaternion of the active gear per environment.

        Returns:
            Tuple of (gear_pos, gear_quat) with shapes (num_envs, 3) and (num_envs, 4).
        """
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this reward term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        self.gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        all_gear_pos = torch.stack(
            [
                self.gear_assets["gear_small"].data.body_pos_w.torch[:, 0],
                self.gear_assets["gear_medium"].data.body_pos_w.torch[:, 0],
                self.gear_assets["gear_large"].data.body_pos_w.torch[:, 0],
            ],
            dim=1,
        )

        all_gear_quat = torch.stack(
            [
                self.gear_assets["gear_small"].data.body_quat_w.torch[:, 0],
                self.gear_assets["gear_medium"].data.body_quat_w.torch[:, 0],
                self.gear_assets["gear_large"].data.body_quat_w.torch[:, 0],
            ],
            dim=1,
        )

        gear_pos = all_gear_pos[self.env_indices, self.gear_type_indices]
        gear_quat = all_gear_quat[self.env_indices, self.gear_type_indices]

        return gear_pos, gear_quat

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg_1: SceneEntityCfg,
        keypoint_scale: float = 1.0,
        add_cube_center_kp: bool = True,
    ) -> torch.Tensor:
        """Compute keypoint distance error.

        Args:
            env: Environment instance
            asset_cfg_1: Configuration of the first asset (RigidObject)
            keypoint_scale: Scale factor for keypoint offsets
            add_cube_center_kp: Whether to include center keypoint

        Returns:
            Mean keypoint distance tensor of shape (num_envs,)
        """
        # Get current pose of asset_1 (RigidObject)
        curr_pos_1 = self.asset_1.data.body_pos_w.torch[:, 0]
        curr_quat_1 = self.asset_1.data.body_quat_w.torch[:, 0]

        # Get selected gear pose
        curr_pos_2, curr_quat_2 = self._get_selected_gear_poses(env)

        # Compute keypoint distance
        keypoint_dist_sep = self.keypoint_computer.compute(
            current_pos=curr_pos_1,
            current_quat=curr_quat_1,
            target_pos=curr_pos_2,
            target_quat=curr_quat_2,
            keypoint_scale=keypoint_scale,
        )

        return keypoint_dist_sep.mean(-1)


class keypoint_entity_error_exp(keypoint_entity_error):
    """Compute exponential keypoint reward between a RigidObject and the dynamically selected gear.

    Inherits gear selection and initialization from :class:`keypoint_entity_error`
    and applies an exponential reward transformation to the keypoint distances.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg_1: SceneEntityCfg,
        kp_exp_coeffs: list[tuple[float, float]] = [(1.0, 0.1)],
        kp_use_sum_of_exps: bool = True,
        keypoint_scale: float = 1.0,
        add_cube_center_kp: bool = True,
    ) -> torch.Tensor:
        """Compute exponential keypoint reward.

        Args:
            env: Environment instance
            asset_cfg_1: Configuration of the first asset (RigidObject)
            kp_exp_coeffs: List of (a, b) coefficient pairs for exponential reward
            kp_use_sum_of_exps: Whether to use sum of exponentials
            keypoint_scale: Scale factor for keypoint offsets
            add_cube_center_kp: Whether to include center keypoint

        Returns:
            Exponential keypoint reward tensor of shape (num_envs,)
        """
        # Get current pose of asset_1 (RigidObject)
        curr_pos_1 = self.asset_1.data.body_pos_w.torch[:, 0]
        curr_quat_1 = self.asset_1.data.body_quat_w.torch[:, 0]

        # Get selected gear pose
        curr_pos_2, curr_quat_2 = self._get_selected_gear_poses(env)

        # Compute keypoint distance
        keypoint_dist_sep = self.keypoint_computer.compute(
            current_pos=curr_pos_1,
            current_quat=curr_quat_1,
            target_pos=curr_pos_2,
            target_quat=curr_quat_2,
            keypoint_scale=keypoint_scale,
        )

        # Compute exponential reward
        keypoint_reward_exp = torch.zeros_like(keypoint_dist_sep[:, 0])

        if kp_use_sum_of_exps:
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += (
                    1.0 / (torch.exp(a * keypoint_dist_sep) + b + torch.exp(-a * keypoint_dist_sep))
                ).mean(-1)
        else:
            keypoint_dist = keypoint_dist_sep.mean(-1)
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += 1.0 / (torch.exp(a * keypoint_dist) + b + torch.exp(-a * keypoint_dist))

        return keypoint_reward_exp


class keypoint_ee_grasp_error(keypoint_entity_error):
    """Compute keypoint distance between the robot end effector and the gear's grasp-corrected pose.

    Transforms the gear's actual world pose into the expected EE position/orientation
    using grasp offsets, so that the distance is ~0 when properly holding the gear
    and increases when the gripper drifts away.

    The penalty is gated by ``ee_grasp_threshold``: It only activates when the mean
    keypoint error exceeds the threshold, i.e., when the EE has drifted away from the
    expected grasp pose. With threshold=0.0, the penalty is effectively always active.

    Supports linear weight ramp-up: The returned reward is scaled by a factor that
    linearly increases from ``weight_ramp_start`` to 1.0 over ``weight_ramp_steps``
    env steps, allowing the reward to grow in importance as training progresses.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        ManagerTermBase.__init__(self, cfg, env)

        self._init_gear_selection(env)
        self.keypoint_computer = _compute_keypoint_distance(cfg, env)

        self.robot_asset_cfg: SceneEntityCfg = cfg.params.get("robot_asset_cfg", SceneEntityCfg("robot"))
        self.robot_asset: Articulation = env.scene[self.robot_asset_cfg.name]

        self.end_effector_body_name: str = cfg.params["end_effector_body_name"]
        grasp_rot_offset = cfg.params["grasp_rot_offset"]
        self.grasp_rot_offset_tensor = (
            torch.tensor(grasp_rot_offset, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
        )

        gear_offsets_grasp = cfg.params["gear_offsets_grasp"]
        self.gear_grasp_offsets_stacked = torch.stack(
            [
                torch.tensor(gear_offsets_grasp["gear_small"], device=env.device, dtype=torch.float32),
                torch.tensor(gear_offsets_grasp["gear_medium"], device=env.device, dtype=torch.float32),
                torch.tensor(gear_offsets_grasp["gear_large"], device=env.device, dtype=torch.float32),
            ],
            dim=0,
        )

        self.weight_ramp_start: float = cfg.params.get("weight_ramp_start", 0.0)
        self.weight_ramp_steps: int = cfg.params.get("weight_ramp_steps", 1)
        self.ee_grasp_threshold: float = cfg.params.get("ee_grasp_threshold", 0.0)

        eef_indices, _ = self.robot_asset.find_bodies([self.end_effector_body_name])
        self.eef_idx = eef_indices[0] if len(eef_indices) > 0 else None
        self._step_count = 0

    def _get_weight_scale(self, env: ManagerBasedRLEnv) -> float:
        progress = min(env.common_step_counter / max(self.weight_ramp_steps, 1), 1.0)
        return self.weight_ramp_start + (1.0 - self.weight_ramp_start) * progress

    def _get_grasp_corrected_target(
        self, env: ManagerBasedRLEnv
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute EE pose and grasp-corrected target pose.

        Returns:
            Tuple of (eef_pos, eef_quat, gear_grasp_pos, gear_quat_grasp).
        """
        eef_pos = self.robot_asset.data.body_link_pos_w.torch[:, self.eef_idx]
        eef_quat = self.robot_asset.data.body_link_quat_w.torch[:, self.eef_idx]

        gear_pos, gear_quat = self._get_selected_gear_poses(env)

        gear_quat_grasp = quat_mul(gear_quat, self.grasp_rot_offset_tensor)
        grasp_offsets = self.gear_grasp_offsets_stacked[self.gear_type_indices]
        gear_grasp_pos = gear_pos + quat_apply(gear_quat_grasp, grasp_offsets)

        return eef_pos, eef_quat, gear_grasp_pos, gear_quat_grasp

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        end_effector_body_name: str = "",
        grasp_rot_offset: list | None = None,
        gear_offsets_grasp: dict | None = None,
        keypoint_scale: float = 1.0,
        add_cube_center_kp: bool = True,
        weight_ramp_start: float = 0.0,
        weight_ramp_steps: int = 1,
        ee_grasp_threshold: float = 0.0,
    ) -> torch.Tensor:
        if self.eef_idx is None:
            return torch.zeros(env.num_envs, device=env.device)

        eef_pos, eef_quat, gear_grasp_pos, gear_quat_grasp = self._get_grasp_corrected_target(env)

        keypoint_dist_sep = self.keypoint_computer.compute(
            current_pos=eef_pos,
            current_quat=eef_quat,
            target_pos=gear_grasp_pos,
            target_quat=gear_quat_grasp,
            keypoint_scale=keypoint_scale,
        )

        mean_kp_error = keypoint_dist_sep.mean(-1)

        is_active = (mean_kp_error > self.ee_grasp_threshold).float()

        weight_scale = self._get_weight_scale(env)
        scaled_reward = mean_kp_error * weight_scale * is_active

        mean_error_scalar = mean_kp_error.mean().item()
        pct_active = is_active.mean().item()

        if not hasattr(env, "extras"):
            env.extras = {}
        if "log" not in env.extras:
            env.extras["log"] = {}
        env.extras["log"]["ee_grasp_kp_error/mean_keypoint_dist"] = mean_error_scalar
        env.extras["log"]["ee_grasp_kp_error/pct_envs_active"] = pct_active
        env.extras["log"]["ee_grasp_kp_error/weight_scale"] = weight_scale

        self._step_count += 1
        import carb

        carb.log_info(
            f"[ee_grasp_kp_error] step={self._step_count}"
            f" | mean_kp_error={mean_error_scalar:.5f}"
            f" | pct_active={pct_active:.3f}"
            f" | weight_scale={weight_scale:.4f}"
        )

        return scaled_reward


class keypoint_ee_grasp_error_exp(keypoint_ee_grasp_error):
    """Compute exponential keypoint reward between the robot end effector and the gear's grasp-corrected pose.

    Transforms the gear's actual world pose into the expected EE position/orientation
    using grasp offsets, so that the reward is high (~1) when properly holding the gear
    and drops sharply when the gripper drifts away.

    The reward is gated by ``ee_grasp_threshold``: It only activates when the mean
    keypoint error exceeds the threshold, i.e. when the EE has drifted away from the
    expected grasp pose. With threshold=0.0 the reward is effectively always active.

    Supports linear weight ramp-up: The returned reward is scaled by a factor that
    linearly increases from ``weight_ramp_start`` to 1.0 over ``weight_ramp_steps``
    env steps, allowing the reward to grow in importance as training progresses.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        end_effector_body_name: str = "",
        grasp_rot_offset: list | None = None,
        gear_offsets_grasp: dict | None = None,
        kp_exp_coeffs: list[tuple[float, float]] = [(1.0, 0.1)],
        kp_use_sum_of_exps: bool = True,
        keypoint_scale: float = 1.0,
        add_cube_center_kp: bool = True,
        weight_ramp_start: float = 0.0,
        weight_ramp_steps: int = 1,
        ee_grasp_threshold: float = 0.0,
    ) -> torch.Tensor:
        if self.eef_idx is None:
            return torch.zeros(env.num_envs, device=env.device)

        eef_pos, eef_quat, gear_grasp_pos, gear_quat_grasp = self._get_grasp_corrected_target(env)

        keypoint_dist_sep = self.keypoint_computer.compute(
            current_pos=eef_pos,
            current_quat=eef_quat,
            target_pos=gear_grasp_pos,
            target_quat=gear_quat_grasp,
            keypoint_scale=keypoint_scale,
        )

        mean_kp_error = keypoint_dist_sep.mean(-1)

        is_active = (mean_kp_error > self.ee_grasp_threshold).float()

        keypoint_reward_exp = torch.zeros_like(keypoint_dist_sep[:, 0])
        if kp_use_sum_of_exps:
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += (
                    1.0 / (torch.exp(a * keypoint_dist_sep) + b + torch.exp(-a * keypoint_dist_sep))
                ).mean(-1)
        else:
            kp_dist_mean = keypoint_dist_sep.mean(-1)
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += 1.0 / (torch.exp(a * kp_dist_mean) + b + torch.exp(-a * kp_dist_mean))

        weight_scale = self._get_weight_scale(env)
        scaled_reward = keypoint_reward_exp * weight_scale * is_active

        mean_error_scalar = mean_kp_error.mean().item()
        mean_reward_scalar = keypoint_reward_exp.mean().item()
        pct_active = is_active.mean().item()

        if not hasattr(env, "extras"):
            env.extras = {}
        if "log" not in env.extras:
            env.extras["log"] = {}
        env.extras["log"]["ee_grasp_kp_error_exp/mean_keypoint_dist"] = mean_error_scalar
        env.extras["log"]["ee_grasp_kp_error_exp/mean_exp_reward"] = mean_reward_scalar
        env.extras["log"]["ee_grasp_kp_error_exp/pct_envs_active"] = pct_active
        env.extras["log"]["ee_grasp_kp_error_exp/weight_scale"] = weight_scale

        self._step_count += 1
        import carb

        carb.log_info(
            f"[ee_grasp_kp_error_exp] step={self._step_count}"
            f" | mean_kp_error={mean_error_scalar:.5f}"
            f" | pct_active={pct_active:.3f}"
            f" | weight_scale={weight_scale:.4f}"
            f" | mean_exp_reward={mean_reward_scalar:.5f}"
        )

        return scaled_reward


##
# Helper functions and classes
##


def _get_keypoint_offsets_full_6d(add_cube_center_kp: bool = False, device: torch.device | None = None) -> torch.Tensor:
    """Get keypoints for pose alignment comparison. Pose is aligned if all axis are aligned.

    Args:
        add_cube_center_kp: Whether to include the center keypoint (0, 0, 0)
        device: Device to create the tensor on

    Returns:
        Keypoint offsets tensor of shape (num_keypoints, 3)
    """
    if add_cube_center_kp:
        keypoint_corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        keypoint_corners = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    keypoint_corners = torch.tensor(keypoint_corners, device=device, dtype=torch.float32)
    keypoint_corners = torch.cat((keypoint_corners, -keypoint_corners[-3:]), dim=0)

    return keypoint_corners


class _compute_keypoint_distance:
    """Compute keypoint distance between current and target poses.

    This helper class pre-caches keypoint offsets and identity quaternions
    to avoid repeated allocations during reward computation.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the compute keypoint distance helper.

        Args:
            cfg: Reward term configuration
            env: Environment instance
        """
        # Get keypoint configuration
        add_cube_center_kp = cfg.params.get("add_cube_center_kp", True)

        # Pre-compute base keypoint offsets (unscaled)
        self.keypoint_offsets_base = _get_keypoint_offsets_full_6d(
            add_cube_center_kp=add_cube_center_kp, device=env.device
        )
        self.num_keypoints = self.keypoint_offsets_base.shape[0]

        # Pre-allocate identity quaternion for keypoint transforms
        self.identity_quat_keypoints = (
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device, dtype=torch.float32)
            .repeat(env.num_envs * self.num_keypoints, 1)
            .contiguous()
        )

        # Pre-allocate buffer for batched keypoint offsets
        self.keypoint_offsets_buffer = torch.zeros(
            env.num_envs, self.num_keypoints, 3, device=env.device, dtype=torch.float32
        )

    def compute(
        self,
        current_pos: torch.Tensor,
        current_quat: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        keypoint_scale: float = 1.0,
    ) -> torch.Tensor:
        """Compute keypoint distance between current and target poses.

        Args:
            current_pos: Current position tensor of shape (num_envs, 3)
            current_quat: Current quaternion tensor of shape (num_envs, 4)
            target_pos: Target position tensor of shape (num_envs, 3)
            target_quat: Target quaternion tensor of shape (num_envs, 4)
            keypoint_scale: Scale factor for keypoint offsets

        Returns:
            Keypoint distance tensor of shape (num_envs, num_keypoints)
        """
        num_envs = current_pos.shape[0]

        # Scale keypoint offsets
        keypoint_offsets = self.keypoint_offsets_base * keypoint_scale

        # Create batched keypoints (in-place operation)
        self.keypoint_offsets_buffer[:num_envs] = keypoint_offsets.unsqueeze(0)

        # Flatten for batch processing
        keypoint_offsets_flat = self.keypoint_offsets_buffer[:num_envs].reshape(-1, 3)
        identity_quat = self.identity_quat_keypoints[: num_envs * self.num_keypoints]

        # Expand quaternions and positions for all keypoints
        current_quat_expanded = current_quat.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 4)
        current_pos_expanded = current_pos.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 3)
        target_quat_expanded = target_quat.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 4)
        target_pos_expanded = target_pos.unsqueeze(1).expand(-1, self.num_keypoints, -1).reshape(-1, 3)

        # Transform all keypoints at once
        keypoints_current_flat, _ = combine_frame_transforms(
            current_pos_expanded, current_quat_expanded, keypoint_offsets_flat, identity_quat
        )
        keypoints_target_flat, _ = combine_frame_transforms(
            target_pos_expanded, target_quat_expanded, keypoint_offsets_flat, identity_quat
        )

        # Reshape back
        keypoints_current = keypoints_current_flat.reshape(num_envs, self.num_keypoints, 3)
        keypoints_target = keypoints_target_flat.reshape(num_envs, self.num_keypoints, 3)

        # Calculate L2 norm distance
        keypoint_dist_sep = torch.linalg.norm(keypoints_target - keypoints_current, ord=2, dim=-1)

        return keypoint_dist_sep
