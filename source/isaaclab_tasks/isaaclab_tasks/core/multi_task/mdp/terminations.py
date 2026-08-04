# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-agnostic termination terms shared across terrain and factory tasks.

Three terms and one base cfg live here:

- :func:`abnormal_robot_state` — joint-velocity limit watchdog. Fires when any
  joint of the asset exceeds twice its declared joint-vel limit. Indicates
  unstable physics from extreme actions and applies equally to manipulators
  and legged robots.
- :func:`out_of_bound` — env-origin-relative AABB containment check on a rigid
  asset's root position. Replaces the absolute-z ``root_height_below_minimum``
  used by terrain (which doesn't generalize to non-zero spawn heights) and
  generalizes the manipulation-side held-asset bounds check.
- :class:`illegal_contact_ratio` — contact-impact watchdog. Fires when a
  contact-sensor body's force exceeds ``ratio × total_bodyweight``. The
  threshold is computed at init from the articulation's per-body mass, so
  the same cfg works across robots without per-robot tuning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import TerminationTermCfg as DoneTermCfg

from ..assembly_keypoints import Offset
from ..assembly_profile import AssemblyProfile
from ..assembly_profile_cfg import AssemblyProfileCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Fire when any joint speed exceeds twice its declared limit.

    Catches unstable physics from extreme actions — applies to any articulated
    asset (manipulator arm, legged base, …).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return (wp.to_torch(robot.data.joint_vel).abs() > (wp.to_torch(robot.data.joint_vel_limits) * 2)).any(dim=1)


def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    in_bound_range: dict[str, tuple[float, float]] = {},
) -> torch.Tensor:
    """Fire when the asset's env-relative root position leaves the AABB.

    Args:
        env: The environment.
        asset_cfg: The asset to track. Defaults to the ``"robot"`` scene entity.
        in_bound_range: Per-axis ``(min, max)`` bounds in env-local frame. Axes
            absent from the dict default to ``(0.0, 0.0)`` — i.e. nothing
            allowed — so callers should specify every axis they care about.

    Note: env-origin-relative, not absolute-world. For terrain envs whose
    spawn z varies with the terrain mesh, this remains correct because the
    env origin tracks the spawn cell.
    """
    object: RigidObject = env.scene[asset_cfg.name]
    range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)

    object_pos_local = wp.to_torch(object.data.root_pos_w) - env.scene.env_origins
    return ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)


class illegal_contact_ratio(ManagerTermBase):
    """Terminate when contact force exceeds ``threshold_ratio × total_bodyweight``.

    The threshold is resolved at construction from the articulation's
    per-body mass — ``ratio × Σ mᵢ × g`` — so the same cfg works across
    robots of different sizes without per-robot threshold presets.

    ``threshold_ratio = 3`` is the natural starting point: routine static
    contact (lying, kneeling, climbing) tops out around 1× bodyweight while
    shock impacts easily exceed 5-10×, so the middle band cleanly separates
    them.

    Domain-agnostic: usable by any task whose contact sensor's body subset
    should be impact-gated (locomotion non-foot bodies, manipulation tool
    shanks, …).

    Args (passed via :attr:`isaaclab.managers.TerminationTermCfg.params`):
        threshold_ratio: Multiple of total bodyweight that constitutes an
            impact.
        sensor_cfg: Contact sensor + body subset to monitor.
        asset_cfg: Articulation whose total mass defines bodyweight.
            Defaults to ``SceneEntityCfg("robot")``.

    Note: the per-env threshold is fixed at construction. Per-episode mass
    randomisation events (e.g. ``add_base_mass``) shift the true bodyweight
    by a few percent, well below the static-vs-impact margin, so the cached
    threshold remains a valid impact gate.
    """

    def __init__(self, cfg: DoneTerm, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        threshold_ratio = float(cfg.params["threshold_ratio"])
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._sensor = env.scene.sensors[sensor_cfg.name]
        self._body_ids = sensor_cfg.body_ids
        asset: Articulation = env.scene[asset_cfg.name]
        # [num_envs, 1] for broadcast against per-body force [num_envs, n_bodies].
        total_mass = wp.to_torch(asset.data.body_mass).sum(dim=-1)
        self._threshold = (threshold_ratio * total_mass * 9.81).unsqueeze(-1)
        # Manager will not pass kwargs back to ``__call__`` if cfg.params is empty.
        cfg.params = {}

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        net_forces = wp.to_torch(self._sensor.data.net_forces_w_history)
        max_force = torch.max(torch.linalg.norm(net_forces[:, :, self._body_ids], dim=-1), dim=1)[0]
        return torch.any(max_force > self._threshold, dim=1)


class progress_context(ManagerTermBase):
    def __init__(self, cfg: DoneTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.held_asset: Articulation | RigidObject = env.scene[cfg.params.get("held_asset_cfg").name]  # type: ignore
        self.fixed_asset: Articulation | RigidObject = env.scene[cfg.params.get("fixed_asset_cfg").name]  # type: ignore
        self.held_asset_offset: Offset = cfg.params.get("held_asset_offset")  # type: ignore
        profile_cfg: AssemblyProfileCfg = cfg.params.get("assembly_profile")  # type: ignore
        self.profile: AssemblyProfile = profile_cfg.class_type(profile_cfg)
        self.success_threshold: float = cfg.params.get("success_threshold")  # type: ignore

        self.orientation_aligned = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.position_centered = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.z_distance_reached = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.is_success = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)
        self.euler_xy_diff = torch.zeros((env.num_envs), device=env.device)
        self.xy_distance = torch.zeros((env.num_envs), device=env.device)
        self.z_distance = torch.zeros((env.num_envs), device=env.device)
        self.dummy_false_tensor = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        success_threshold: float,
        held_asset_cfg: SceneEntityCfg,
        fixed_asset_cfg: SceneEntityCfg,
        held_asset_offset: Offset,
        assembly_profile: AssemblyProfileCfg,
    ) -> torch.Tensor:
        held_asset_alignment_pos_w, held_asset_alignment_quat_w = self.held_asset_offset.apply(self.held_asset)
        fixed_asset_alignment_pos_w, fixed_asset_alignment_quat_w = self.profile.assembled_offset.apply(
            self.fixed_asset
        )
        held_asset_in_fixed_asset_frame_pos, held_asset_in_fixed_asset_frame_quat = (
            math_utils.subtract_frame_transforms(
                fixed_asset_alignment_pos_w,
                fixed_asset_alignment_quat_w,
                held_asset_alignment_pos_w,
                held_asset_alignment_quat_w,
            )
        )

        e_x, e_y, _ = math_utils.euler_xyz_from_quat(held_asset_in_fixed_asset_frame_quat)
        self.euler_xy_diff[:] = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
        self.xy_distance[:] = torch.norm(held_asset_in_fixed_asset_frame_pos[:, 0:2], dim=1)
        self.z_distance[:] = held_asset_in_fixed_asset_frame_pos[:, 2]

        self.orientation_aligned[:] = self.euler_xy_diff < 0.025
        self.position_centered[:] = self.xy_distance < 0.0025
        self.z_distance_reached[:] = self.z_distance < self.success_threshold
        self.is_success[:] = self.orientation_aligned & self.position_centered & self.z_distance_reached
        env.extras["successes"] = self.is_success

        return self.dummy_false_tensor


def success_termination(env: ManagerBasedRLEnv, context: str = "progress_context") -> torch.Tensor:
    return env.termination_manager.get_term_cfg(context).func.is_success
