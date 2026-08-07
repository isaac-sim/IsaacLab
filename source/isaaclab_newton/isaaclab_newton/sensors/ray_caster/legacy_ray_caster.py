# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy Newton adapters for Warp-mesh ray-caster implementations."""

from __future__ import annotations

# pyright: reportInvalidTypeForm=none, reportPrivateUsage=none
import warnings
from typing import Any

import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster import BaseMultiMeshRayCaster
from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster_camera import BaseMultiMeshRayCasterCamera
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster.base_ray_caster_camera import BaseRayCasterCamera
from isaaclab.sensors.ray_caster.kernels import copy_mesh_poses_to_table_kernel

from isaaclab_newton.physics import NewtonManager

from .newton_raycast_sensor import _newton_body_pattern, _NewtonRayCasterPoseMixin


def _has_rigid_body_api(prim) -> bool:
    """Return whether a USD prim has rigid-body physics applied."""
    return bool(prim.HasAPI(UsdPhysics.RigidBodyAPI))


class _LegacyNewtonRayCasterMixin(_NewtonRayCasterPoseMixin):
    """Add explicit mesh-target tracking required by legacy ray casters."""

    def __init__(self: Any, cfg):
        super().__init__(cfg)
        self._tracked_site_labels_by_target: dict[tuple[str, ...], list[str]] = {}
        for target_cfg in getattr(self, "_raycast_targets_cfg", []):
            if target_cfg.track_mesh_transforms:
                owner_exprs = self._resolve_target_owner_exprs(target_cfg.prim_expr)
                labels = self._register_target_sites_for_exprs(owner_exprs)
                self._tracked_site_labels_by_target[tuple(owner_exprs)] = labels

    def _resolve_target_owner_exprs(self, prim_expr: str) -> list[str]:
        """Resolve mesh target expressions to owning rigid-body expressions."""
        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        resolved = cloner.query.path_to_source(plan, prim_expr) if plan is not None else None
        if resolved is not None:
            source_path, dest_glob, asset_suffix = resolved
            walk_root = source_path + asset_suffix
            source_prims = sim_utils.find_matching_prims(walk_root)
            if not source_prims:
                raise RuntimeError(f"No ClonePlan source prims matched '{walk_root}'.")
            owner_exprs: list[str] = []
            for source_prim in source_prims:
                body = sim_utils.get_first_matching_ancestor_prim(source_prim.GetPath(), predicate=_has_rigid_body_api)
                if body is None:
                    raise RuntimeError(
                        f"Cannot track non-physics ray-cast target '{prim_expr}' with Newton. "
                        "Set track_mesh_transforms=False for static targets, or apply RigidBodyAPI "
                        "to dynamic targets."
                    )
                owner_prim_path = str(body.GetPath())
                owner_exprs.append(dest_glob + owner_prim_path[len(source_path) :])
            return list(dict.fromkeys(owner_exprs))

        prims = sim_utils.find_matching_prims(prim_expr)
        if not prims:
            return [_newton_body_pattern(prim_expr)]
        owner_exprs = []
        for prim in prims:
            body = sim_utils.get_first_matching_ancestor_prim(prim.GetPath(), predicate=_has_rigid_body_api)
            if body is None:
                raise RuntimeError(
                    f"Cannot track non-physics ray-cast target '{prim_expr}' with Newton. "
                    "Set track_mesh_transforms=False for static targets, or apply RigidBodyAPI "
                    "to dynamic targets."
                )
            owner_exprs.append(_newton_body_pattern(str(body.GetPath())))
        return list(dict.fromkeys(owner_exprs))

    def _register_target_sites_for_exprs(self, owner_exprs: list[str]) -> list[str]:
        """Register identity-pose Newton sites on target owner bodies."""
        identity = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))
        labels = [NewtonManager.cl_register_site(owner_expr, identity) for owner_expr in owner_exprs]
        return list(dict.fromkeys(labels))

    def _create_tracked_target_view(self: Any, target_prim_path: str | list[str]) -> wp.array:
        """Resolve dynamic multi-mesh target sites to Newton site indices."""
        target_exprs = target_prim_path if isinstance(target_prim_path, list) else [target_prim_path]
        labels = self._tracked_site_labels_by_target[tuple(target_exprs)]
        site_indices = self._resolve_site_indices(labels, str(target_prim_path), self._num_envs)
        return wp.array(site_indices, dtype=wp.int32, device=self._device)

    def _update_mesh_transforms(self: Any) -> None:
        """Refresh dynamic multi-mesh targets from Newton sites."""
        if not hasattr(self, "_mesh_views"):
            return
        mesh_index = 0
        for site_indices, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_index += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            site_count = site_indices.shape[0]
            pos_buf = wp.empty(site_count, dtype=wp.vec3f, device=self._device)
            quat_buf = wp.empty(site_count, dtype=wp.quatf, device=self._device)
            pose_buf = wp.empty(site_count, dtype=wp.transformf, device=self._device)
            self._update_newton_site_transforms(site_indices, pose_buf, pos_buf, quat_buf)
            meshes_per_env = site_count if site_count == 1 else site_count // self._num_envs

            wp.launch(
                copy_mesh_poses_to_table_kernel,
                dim=(self._num_envs, meshes_per_env),
                inputs=[
                    pos_buf,
                    quat_buf,
                    int(meshes_per_env),
                    int(mesh_index),
                    bool(site_count == 1),
                    self._mesh_positions_w,
                    self._mesh_orientations_w,
                ],
                device=self._device,
            )
            mesh_index += self._num_meshes_per_env[target_cfg.prim_expr]


class LegacyRayCaster(_LegacyNewtonRayCasterMixin, BaseRayCaster):
    """Legacy Newton ray caster that queries one configured Warp mesh."""


class LegacyRayCasterCamera(_LegacyNewtonRayCasterMixin, BaseRayCasterCamera):
    """Legacy Newton ray-caster camera backed by configured Warp meshes."""


class LegacyMultiMeshRayCaster(_LegacyNewtonRayCasterMixin, BaseMultiMeshRayCaster):
    """Legacy Newton ray caster for configured static and dynamic Warp meshes."""


class LegacyMultiMeshRayCasterCamera(_LegacyNewtonRayCasterMixin, BaseMultiMeshRayCasterCamera):
    """Legacy Newton ray-caster camera for configured Warp meshes."""


def _warn_legacy_alias(old_name: str, new_name: str) -> None:
    """Warn when a pre-rename Newton backend class is constructed directly."""
    warnings.warn(
        f"isaaclab_newton.sensors.{old_name} is deprecated; use isaaclab_newton.sensors.{new_name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class RayCasterCamera(LegacyRayCasterCamera):
    """Deprecated alias for :class:`LegacyRayCasterCamera`."""

    def __init__(self, cfg):
        _warn_legacy_alias("RayCasterCamera", "LegacyRayCasterCamera")
        super().__init__(cfg)


class MultiMeshRayCaster(LegacyMultiMeshRayCaster):
    """Deprecated alias for :class:`LegacyMultiMeshRayCaster`."""

    def __init__(self, cfg):
        _warn_legacy_alias("MultiMeshRayCaster", "LegacyMultiMeshRayCaster")
        super().__init__(cfg)


class MultiMeshRayCasterCamera(LegacyMultiMeshRayCasterCamera):
    """Deprecated alias for :class:`LegacyMultiMeshRayCasterCamera`."""

    def __init__(self, cfg):
        _warn_legacy_alias("MultiMeshRayCasterCamera", "LegacyMultiMeshRayCasterCamera")
        super().__init__(cfg)
