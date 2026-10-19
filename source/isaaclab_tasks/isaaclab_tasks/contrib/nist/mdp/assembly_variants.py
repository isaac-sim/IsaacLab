# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime state for reset-selectable assembly variants."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from ..assembly_profile import AssemblyProfile
from ..assembly_variants import ASSEMBLY_VARIANT_NAMES, ASSEMBLY_VARIANTS

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from ..utils.pose_offset import Offset


_OFFSET_NAMES = ("board", "fixed_tip", "held_align", "held_grasp_point", "held_grasp_middle", "assembled")
_RANGE_NAMES = ("grasped", "grasped_centered")
_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


class AssemblyVariantContext(ManagerTermBase):
    """Own the per-environment assembly index and its packed task geometry."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        names: tuple[str, ...] = cfg.params["variant_names"]
        if names != ASSEMBLY_VARIANT_NAMES:
            raise ValueError("Scene and assembly variant order differ.")

        self.variant_names = names
        self.fixed_asset: RigidObject = env.scene[cfg.params["fixed_asset_cfg"].name]
        self.held_asset: RigidObject = env.scene[cfg.params["held_asset_cfg"].name]
        self.num_variants = len(ASSEMBLY_VARIANTS)
        for name, asset in (("fixed", self.fixed_asset), ("held", self.held_asset)):
            if asset.num_mesh_variants != self.num_variants:
                raise ValueError(f"{name!s} mesh bank does not match the assembly catalog.")

        offsets = []
        for variant in ASSEMBLY_VARIANTS:
            offsets.append(
                [
                    variant.board_offset.pose,
                    variant.fixed_tip.pose,
                    variant.held_align.pose,
                    variant.held_grasp_point.pose,
                    variant.held_grasp_middle.pose,
                    AssemblyProfile(variant.profile).assembled_offset.pose,
                ]
            )
        self._offsets = torch.tensor(offsets, device=env.device)
        self._offsets_warp: wp.array(dtype=wp.transformf, ndim=2) | None = None
        pos, quat = self._offsets[..., :3], self._offsets[..., 3:]
        inv_quat = math_utils.quat_inv(quat.reshape(-1, 4)).view_as(quat)
        inv_pos = -math_utils.quat_apply(inv_quat.reshape(-1, 4), pos.reshape(-1, 3)).view_as(pos)
        self._inverse_offsets = torch.cat((inv_pos, inv_quat), dim=-1)

        self._ranges = (
            torch.tensor(
                [
                    [[variant.grasped_pose_range[axis], variant.grasped_pose_range_centered[axis]] for axis in _AXES]
                    for variant in ASSEMBLY_VARIANTS
                ],
                device=env.device,
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        self._grasp_diameters = torch.tensor(
            [variant.held_grasp_diameter for variant in ASSEMBLY_VARIANTS], device=env.device
        )
        self._profiles = tuple(AssemblyProfile(variant.profile) for variant in ASSEMBLY_VARIANTS)
        self._next_variant_ids = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    @property
    def variant_ids(self) -> torch.Tensor:
        """Current assembly index for every environment."""
        return self.fixed_asset.mesh_variant_ids.torch

    @property
    def variant_ids_warp(self) -> wp.array(dtype=wp.int32):
        """Current assembly indices in Newton-native storage."""
        return self.fixed_asset.mesh_variant_ids.warp

    def offset_warp(self, offset: Offset | str) -> wp.array(dtype=wp.transformf):
        """Frame poses, with position [m] and quaternion xyzw, in mesh-variant order."""
        if not isinstance(offset, str):
            return wp.full(
                self.num_variants, wp.transformf(*offset.pose), dtype=wp.transformf, device=self.fixed_asset.device
            )
        if self._offsets_warp is None:
            self._offsets_warp = wp.from_torch(self._offsets, dtype=wp.transformf)
        return self._offsets_warp[:, _OFFSET_NAMES.index(offset)]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        fixed_asset_cfg: SceneEntityCfg,
        held_asset_cfg: SceneEntityCfg,
        variant_names: tuple[str, ...],
    ) -> None:
        self.select(env_ids)

    def select(self, env_ids: torch.Tensor | None) -> None:
        """Select one coherent fixed/held pair for each requested environment."""
        device = self.fixed_asset.device
        if env_ids is None:
            env_ids = torch.arange(self.fixed_asset.num_instances, device=device)
        variant_ids = self._next_variant_ids[env_ids]
        self.fixed_asset.write_mesh_variant_to_sim(variant_ids, env_ids)
        self.held_asset.write_mesh_variant_to_sim(variant_ids, env_ids)

    def prepare(self, variant_ids: torch.Tensor) -> None:
        """Set the assembly indices used by selection."""
        self._next_variant_ids.copy_(variant_ids)

    def _rows(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        ids = self.variant_ids if env_ids is None else self.variant_ids[env_ids]
        return ids.long()

    def combine(
        self, name: str, pos: torch.Tensor, quat: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose a variant offset onto parent frames."""
        offset = self._offsets[self._rows(env_ids), _OFFSET_NAMES.index(name)]
        return math_utils.combine_frame_transforms(pos, quat, offset[:, :3], offset[:, 3:])

    def subtract(
        self, name: str, pos: torch.Tensor, quat: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove a variant offset from target frames."""
        offset = self._inverse_offsets[self._rows(env_ids), _OFFSET_NAMES.index(name)]
        return math_utils.combine_frame_transforms(pos, quat, offset[:, :3], offset[:, 3:])

    def apply(
        self, name: str, asset: RigidObject | Articulation, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform a variant offset through an asset root pose."""
        pos, quat = asset.data.root_pos_w.torch, asset.data.root_quat_w.torch
        if env_ids is not None:
            pos, quat = pos[env_ids], quat[env_ids]
        return self.combine(name, pos, quat, env_ids)

    def pose_range(self, name: str, env_ids: torch.Tensor) -> torch.Tensor:
        """Return per-environment pose ranges in xyz/rpy order."""
        return self._ranges[self._rows(env_ids), _RANGE_NAMES.index(name)]

    def grasp_diameter(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return held-asset grasp diameters [m]."""
        return self._grasp_diameters[self._rows(env_ids)]

    def sample_profile(
        self, fraction_range: tuple[float, float], env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample every selected environment from its assembly profile."""
        ids = self._rows(env_ids)
        pos = torch.empty((len(env_ids), 3), device=env_ids.device)
        quat = torch.empty((len(env_ids), 4), device=env_ids.device)
        for index, profile in enumerate(self._profiles):
            rows = (ids == index).nonzero().flatten()
            if rows.numel() == 0:
                continue
            pos[rows], quat[rows] = profile.sample(fraction_range, rows.numel(), env_ids.device)
        return pos, quat

    def one_hot(self) -> torch.Tensor:
        """Return the current assembly identity as a one-hot tensor."""
        return F.one_hot(self.variant_ids.long(), num_classes=self.num_variants).float()


def assembly_variant_context(env: ManagerBasedRLEnv, name: str = "assembly_variants") -> AssemblyVariantContext:
    """Return the configured assembly context."""
    return env.event_manager.get_term_cfg(name).func


def select_assembly_variant(env: ManagerBasedRLEnv, env_ids: torch.Tensor, context: str = "assembly_variants") -> None:
    """Select matching fixed and held mesh variants."""
    assembly_variant_context(env, context).select(env_ids)


def assembly_variant_one_hot(env: ManagerBasedRLEnv, context: str = "assembly_variants") -> torch.Tensor:
    """Observe the active assembly pair."""
    return assembly_variant_context(env, context).one_hot()
