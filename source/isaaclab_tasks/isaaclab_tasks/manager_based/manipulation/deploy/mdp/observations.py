# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class-based observation terms for the gear assembly manipulation environment.

Migrated from PhysX to Newton. Key changes:
- Data access returns Warp arrays -> ``wp.to_torch()`` for torch operations
- Quaternion w-component is at index 3 (XYZW), not index 0 (WXYZ)
- Identity quaternion is ``[0, 0, 0, 1]`` in XYZW (not ``[1, 0, 0, 0]``)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import warp as wp

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .events import randomize_gear_type


class gear_shaft_pos_w(ManagerTermBase):
    """Gear shaft position in world frame with offset applied.

    This class-based term caches gear offset tensors and identity quaternions for efficient computation
    across all environments. It transforms the gear base position by the appropriate offset based on the
    active gear type in each environment.

    Args:
        asset_cfg: The asset configuration for the gear base. Defaults to SceneEntityCfg("factory_gear_base").
        gear_offsets: A dictionary mapping gear type names to their shaft offsets in the gear base frame.

    Returns:
        Gear shaft position tensor in the environment frame with shape (num_envs, 3).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the gear shaft position observation term."""
        super().__init__(cfg, env)

        # Cache asset
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("factory_gear_base"))
        self.asset: RigidObject = env.scene[self.asset_cfg.name]

        # Pre-cache gear offset tensors (required parameter)
        if "gear_offsets" not in cfg.params:
            raise ValueError(
                "'gear_offsets' parameter is required in gear_shaft_pos_w configuration. "
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

        # Stack offset tensors for vectorized indexing (shape: 3, 3)
        # Index 0=small, 1=medium, 2=large
        self.gear_offsets_stacked = torch.stack(
            [
                self.gear_offset_tensors["gear_small"],
                self.gear_offset_tensors["gear_medium"],
                self.gear_offset_tensors["gear_large"],
            ],
            dim=0,
        )

        # Pre-allocate buffers
        self.offsets_buffer = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
        self.env_indices = torch.arange(env.num_envs, device=env.device)
        # XYZW identity quaternion: [0, 0, 0, 1]
        self.identity_quat = (
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device, dtype=torch.float32)
            .repeat(env.num_envs, 1)
            .contiguous()
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("factory_gear_base"),
        gear_offsets: dict | None = None,
    ) -> torch.Tensor:
        """Compute gear shaft position in world frame."""
        # Check if gear type manager exists
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this observation term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Get base gear position and orientation (convert from Warp)
        base_pos = wp.to_torch(self.asset.data.root_link_pos_w)
        base_quat = wp.to_torch(self.asset.data.root_link_quat_w)

        # Update offsets using vectorized indexing
        self.offsets_buffer = self.gear_offsets_stacked[gear_type_indices]

        # Transform offsets (combine_frame_transforms uses XYZW natively)
        shaft_pos, _ = combine_frame_transforms(base_pos, base_quat, self.offsets_buffer, self.identity_quat)

        return shaft_pos - env.scene.env_origins


class gear_shaft_quat_w(ManagerTermBase):
    """Gear shaft orientation in world frame.

    Returns the orientation of the gear base. The quaternion is canonicalized to
    ensure the w component is positive (w is at index 3 in XYZW convention).

    Returns:
        Gear shaft orientation tensor as a quaternion (x, y, z, w) with shape (num_envs, 4).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the gear shaft orientation observation term."""
        super().__init__(cfg, env)

        # Cache asset
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("factory_gear_base"))
        self.asset: RigidObject = env.scene[self.asset_cfg.name]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("factory_gear_base"),
    ) -> torch.Tensor:
        """Compute gear shaft orientation in world frame."""
        # Get base quaternion (convert from Warp)
        base_quat = wp.to_torch(self.asset.data.root_link_quat_w)

        # Ensure w component is positive (w is at index 3 in XYZW)
        w_negative = base_quat[:, 3] < 0
        positive_quat = base_quat.clone()
        positive_quat[w_negative] = -base_quat[w_negative]

        return positive_quat


class gear_pos_w(ManagerTermBase):
    """Gear position in world frame.

    Returns the position of the active gear in each environment using vectorized indexing.

    Returns:
        Gear position tensor in the environment frame with shape (num_envs, 3).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the gear position observation term."""
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
        """Compute gear position in world frame."""
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this observation term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        self.gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Stack all gear positions (convert from Warp)
        all_gear_positions = torch.stack(
            [
                wp.to_torch(self.gear_assets["gear_small"].data.root_link_pos_w),
                wp.to_torch(self.gear_assets["gear_medium"].data.root_link_pos_w),
                wp.to_torch(self.gear_assets["gear_large"].data.root_link_pos_w),
            ],
            dim=1,
        )

        # Select gear positions using advanced indexing
        gear_positions = all_gear_positions[self.env_indices, self.gear_type_indices]

        return gear_positions - env.scene.env_origins


class gear_quat_w(ManagerTermBase):
    """Gear orientation in world frame.

    Returns the orientation of the active gear in each environment. The quaternion is
    canonicalized to ensure the w component is positive (w is at index 3 in XYZW).

    Returns:
        Gear orientation tensor as a quaternion (x, y, z, w) with shape (num_envs, 4).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the gear orientation observation term."""
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
        """Compute gear orientation in world frame."""
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this observation term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        self.gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Stack all gear quaternions (convert from Warp)
        all_gear_quat = torch.stack(
            [
                wp.to_torch(self.gear_assets["gear_small"].data.root_link_quat_w),
                wp.to_torch(self.gear_assets["gear_medium"].data.root_link_quat_w),
                wp.to_torch(self.gear_assets["gear_large"].data.root_link_quat_w),
            ],
            dim=1,
        )

        # Select gear quaternions using advanced indexing
        gear_quat = all_gear_quat[self.env_indices, self.gear_type_indices]

        # Ensure w component is positive (w is at index 3 in XYZW)
        w_negative = gear_quat[:, 3] < 0
        gear_positive_quat = gear_quat.clone()
        gear_positive_quat[w_negative] = -gear_quat[w_negative]

        return gear_positive_quat
