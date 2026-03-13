# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class-based observation terms for the gear assembly manipulation environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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
            Required keys are "gear_small", "gear_medium", and "gear_large", each mapping to a 3D offset
            list [x, y, z]. This parameter is required and must be provided in the configuration.

    Returns:
        Gear shaft position tensor in the environment frame with shape (num_envs, 3).

    Raises:
        ValueError: If the 'gear_offsets' parameter is not provided in the configuration.
        TypeError: If the 'gear_offsets' parameter is not a dictionary.
        ValueError: If any of the required gear type keys are missing from 'gear_offsets'.
        RuntimeError: If the gear type manager is not initialized in the environment.
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
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this observation term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        # Get gear type indices directly as tensor (no Python loops!)
        gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Get base gear position and orientation
        base_pos = self.asset.data.root_pos_w
        base_quat = self.asset.data.root_quat_w

        # Update offsets using vectorized indexing
        self.offsets_buffer = self.gear_offsets_stacked[gear_type_indices]

        # Transform offsets
        shaft_pos, _ = combine_frame_transforms(base_pos, base_quat, self.offsets_buffer, self.identity_quat)

        return shaft_pos - env.scene.env_origins


class gear_shaft_quat_w(ManagerTermBase):
    """Gear shaft orientation in world frame.

    This class-based term returns the orientation of the gear base (which is the same as the gear shaft
    orientation). The quaternion is canonicalized to ensure the w component is positive, reducing
    observation variation for the policy.

    Args:
        asset_cfg: The asset configuration for the gear base. Defaults to SceneEntityCfg("factory_gear_base").

    Returns:
        Gear shaft orientation tensor as a quaternion (w, x, y, z) with shape (num_envs, 4).
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

        # Ensure w component is positive (q and -q represent the same rotation)
        # Pick one canonical form to reduce observation variation seen by the policy
        w_negative = base_quat[:, 0] < 0
        positive_quat = base_quat.clone()
        positive_quat[w_negative] = -base_quat[w_negative]

        return positive_quat


class gear_pos_w(ManagerTermBase):
    """Gear position in world frame.

    This class-based term returns the position of the active gear in each environment. It uses
    vectorized indexing to efficiently select the correct gear position based on the gear type
    (small, medium, or large) active in each environment.

    Returns:
        Gear position tensor in the environment frame with shape (num_envs, 3).

    Raises:
        RuntimeError: If the gear type manager is not initialized in the environment.
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
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this observation term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        # Get gear type indices directly as tensor (no Python loops!)
        self.gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Stack all gear positions
        all_gear_positions = torch.stack(
            [
                self.gear_assets["gear_small"].data.root_pos_w,
                self.gear_assets["gear_medium"].data.root_pos_w,
                self.gear_assets["gear_large"].data.root_pos_w,
            ],
            dim=1,
        )

        # Select gear positions using advanced indexing
        gear_positions = all_gear_positions[self.env_indices, self.gear_type_indices]

        return gear_positions - env.scene.env_origins


class gear_quat_w(ManagerTermBase):
    """Gear orientation in world frame.

    This class-based term returns the orientation of the active gear in each environment. It uses
    vectorized indexing to efficiently select the correct gear orientation based on the gear type
    (small, medium, or large) active in each environment. The quaternion is canonicalized to ensure
    the w component is positive, reducing observation variation for the policy.

    Returns:
        Gear orientation tensor as a quaternion (w, x, y, z) with shape (num_envs, 4).

    Raises:
        RuntimeError: If the gear type manager is not initialized in the environment.
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
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this observation term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        # Get gear type indices directly as tensor (no Python loops!)
        self.gear_type_indices = gear_type_manager.get_all_gear_type_indices()

        # Stack all gear quaternions
        all_gear_quat = torch.stack(
            [
                self.gear_assets["gear_small"].data.root_quat_w,
                self.gear_assets["gear_medium"].data.root_quat_w,
                self.gear_assets["gear_large"].data.root_quat_w,
            ],
            dim=1,
        )

        # Select gear quaternions using advanced indexing
        gear_quat = all_gear_quat[self.env_indices, self.gear_type_indices]

        # Ensure w component is positive (q and -q represent the same rotation)
        # Pick one canonical form to reduce observation variation seen by the policy
        w_negative = gear_quat[:, 0] < 0
        gear_positive_quat = gear_quat.clone()
        gear_positive_quat[w_negative] = -gear_quat[w_negative]

        return gear_positive_quat
