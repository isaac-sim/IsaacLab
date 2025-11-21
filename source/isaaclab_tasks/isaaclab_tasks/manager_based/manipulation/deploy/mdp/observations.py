# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class-based observation terms for the gear assembly manipulation environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaacsim.core.utils.torch.transformations import tf_combine

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .events import RandomizeGearType


class GearShaftPosW(ManagerTermBase):
    """Gear shaft position in world frame with offset applied.

    This class-based term caches gear offset tensors and identity quaternions.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the gear shaft position observation term.

        Args:
            cfg: Observation term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("factory_gear_base"))
        self.asset: RigidObject = env.scene[self.asset_cfg.name]

        # Pre-cache gear offset tensors (required parameter)
        if "gear_offsets" not in cfg.params:
            raise ValueError(
                "'gear_offsets' parameter is required in GearShaftPosW configuration. "
                "It should be a dict with keys 'gear_small', 'gear_medium', 'gear_large' mapping to [x, y, z] offsets."
            )
        gear_offsets = cfg.params["gear_offsets"]
        if not isinstance(gear_offsets, dict):
            raise TypeError(
                f"'gear_offsets' parameter must be a dict, got {type(gear_offsets).__name__}. "
                "It should have keys 'gear_small', 'gear_medium', 'gear_large' mapping to [x, y, z] offsets."
            )

        self.gear_offset_tensors = {}
        for gear_type in ["gear_small", "gear_medium", "gear_large"]:
            if gear_type not in gear_offsets:
                raise ValueError(
                    f"'{gear_type}' offset is required in 'gear_offsets' parameter. "
                    f"Found keys: {list(gear_offsets.keys())}"
                )
            self.gear_offset_tensors[gear_type] = torch.tensor(
                gear_offsets[gear_type], device=env.device, dtype=torch.float32
            )

        # Pre-allocate buffers
        self.offsets_buffer = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
        self.identity_quat = (
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device, dtype=torch.float32)
            .repeat(env.num_envs, 1)
            .contiguous()
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("factory_gear_base"),
        gear_offsets: dict | None = None,
    ) -> torch.Tensor:
        """Compute gear shaft position in world frame.

        Args:
            env: Environment instance
            asset_cfg: Configuration of the gear base asset (unused, kept for compatibility)

        Returns:
            Gear shaft position tensor of shape (num_envs, 3)
        """
        # Check if gear type manager exists
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure RandomizeGearType event is configured "
                "in your environment's event configuration before this observation term is used."
            )

        gear_type_manager: RandomizeGearType = env._gear_type_manager
        current_gear_types = gear_type_manager.get_all_gear_types()

        # Get base gear position and orientation
        base_pos = self.asset.data.root_pos_w
        base_quat = self.asset.data.root_quat_w

        # Update offsets using cached gear offset tensors
        for i in range(env.num_envs):
            self.offsets_buffer[i] = self.gear_offset_tensors[current_gear_types[i]]

        # Transform offsets
        _, shaft_pos = tf_combine(base_quat, base_pos, self.identity_quat, self.offsets_buffer)

        return shaft_pos - env.scene.env_origins


class GearShaftQuatW(ManagerTermBase):
    """Gear shaft orientation in world frame.

    This class-based term caches the asset reference.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the gear shaft orientation observation term.

        Args:
            cfg: Observation term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Cache asset
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("factory_gear_base"))
        self.asset: RigidObject = env.scene[self.asset_cfg.name]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("factory_gear_base"),
    ) -> torch.Tensor:
        """Compute gear shaft orientation in world frame.

        Args:
            env: Environment instance
            asset_cfg: Configuration of the gear base asset (unused, kept for compatibility)

        Returns:
            Gear shaft orientation tensor of shape (num_envs, 4)
        """
        # Get base quaternion
        base_quat = self.asset.data.root_quat_w

        # Ensure w component is positive
        w_negative = base_quat[:, 0] < 0
        positive_quat = base_quat.clone()
        positive_quat[w_negative] = -base_quat[w_negative]

        return positive_quat


class GearPosW(ManagerTermBase):
    """Gear position in world frame.

    This class-based term caches gear type mapping and index tensors.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the gear position observation term.

        Args:
            cfg: Observation term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Pre-allocate gear type mapping and indices
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.env_indices = torch.arange(env.num_envs, device=env.device)

        # Cache gear assets
        self.gear_assets = {
            "gear_small": env.scene["factory_gear_small"],
            "gear_medium": env.scene["factory_gear_medium"],
            "gear_large": env.scene["factory_gear_large"],
        }

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute gear position in world frame.

        Args:
            env: Environment instance

        Returns:
            Gear position tensor of shape (num_envs, 3)
        """
        # Check if gear type manager exists
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure RandomizeGearType event is configured "
                "in your environment's event configuration before this observation term is used."
            )

        gear_type_manager: RandomizeGearType = env._gear_type_manager
        current_gear_types = gear_type_manager.get_all_gear_types()

        # Stack all gear positions
        all_gear_positions = torch.stack(
            [
                self.gear_assets["gear_small"].data.root_pos_w,
                self.gear_assets["gear_medium"].data.root_pos_w,
                self.gear_assets["gear_large"].data.root_pos_w,
            ],
            dim=1,
        )

        # Update gear_type_indices
        for i in range(env.num_envs):
            self.gear_type_indices[i] = self.gear_type_map[current_gear_types[i]]

        # Select gear positions using advanced indexing
        gear_positions = all_gear_positions[self.env_indices, self.gear_type_indices]

        return gear_positions - env.scene.env_origins


class GearQuatW(ManagerTermBase):
    """Gear orientation in world frame.

    This class-based term caches gear type mapping and index tensors.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the gear orientation observation term.

        Args:
            cfg: Observation term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Pre-allocate gear type mapping and indices
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.env_indices = torch.arange(env.num_envs, device=env.device)

        # Cache gear assets
        self.gear_assets = {
            "gear_small": env.scene["factory_gear_small"],
            "gear_medium": env.scene["factory_gear_medium"],
            "gear_large": env.scene["factory_gear_large"],
        }

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute gear orientation in world frame.

        Args:
            env: Environment instance

        Returns:
            Gear orientation tensor of shape (num_envs, 4)
        """
        # Check if gear type manager exists
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure RandomizeGearType event is configured "
                "in your environment's event configuration before this observation term is used."
            )

        gear_type_manager: RandomizeGearType = env._gear_type_manager
        current_gear_types = gear_type_manager.get_all_gear_types()

        # Stack all gear quaternions
        all_gear_quat = torch.stack(
            [
                self.gear_assets["gear_small"].data.root_quat_w,
                self.gear_assets["gear_medium"].data.root_quat_w,
                self.gear_assets["gear_large"].data.root_quat_w,
            ],
            dim=1,
        )

        # Update gear_type_indices
        for i in range(env.num_envs):
            self.gear_type_indices[i] = self.gear_type_map[current_gear_types[i]]

        # Select gear quaternions using advanced indexing
        gear_quat = all_gear_quat[self.env_indices, self.gear_type_indices]

        # Ensure w component is positive
        w_negative = gear_quat[:, 0] < 0
        gear_positive_quat = gear_quat.clone()
        gear_positive_quat[w_negative] = -gear_quat[w_negative]

        return gear_positive_quat
