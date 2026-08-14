# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable description of a tiled camera passed to render backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan
    from isaaclab.sensors.camera.camera_cfg import CameraCfg


@dataclass(frozen=True)
class CameraRenderSpec:
    """Stable inputs for :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data`.

    Backends use this instead of holding a reference to the :class:`~isaaclab.sensors.camera.Camera`
    sensor instance, avoiding circular dependencies between sensors and render data.

    Args:
        cfg: Camera configuration (data types, resolution, filters, etc.).
        device: Torch device string (e.g. ``"cuda:0"``) used by GPU annotators and Warp.
        num_instances: Number of tiled camera instances (environments).
        camera_prim_paths: Absolute USD paths for camera prims authored on the source stage. For
            prototype-only stages, this contains only the source or prototype paths.
        view_count: Number of logical camera views in the renderer output. This may exceed
            ``len(camera_prim_paths)`` when a backend expands a source or prototype camera internally.
        camera_path_relative_to_env_0: Legacy camera path relative to env 0. Current renderers resolve
            prototype paths from the clone plan.
    """

    cfg: CameraCfg
    device: str
    num_instances: int
    camera_prim_paths: tuple[str, ...]
    view_count: int
    camera_path_relative_to_env_0: str = ""

    def resolve_camera_prim_paths(self, clone_plan: ClonePlan | None) -> list[str]:
        """Resolve authored prototype cameras to logical camera paths in clone-plan order.

        Args:
            clone_plan: Replication layout, or None when camera paths are already concrete.

        Returns:
            One camera prim path per logical view.

        Raises:
            RuntimeError: If a planned prototype is missing, an environment has no camera source,
                or the resolved count differs from the logical view count.
        """
        from isaaclab import cloner  # noqa: PLC0415

        paths_by_env: dict[int, str] = {}
        plan_env_ids: list[int] | None = None
        if clone_plan is not None and clone_plan.env_ids is not None:
            plan_env_ids = [int(env_id) for env_id in clone_plan.env_ids.detach().cpu().tolist()]
            for source_root, destination, source_path, env_ids in cloner.query.iter_sources(
                clone_plan, self.cfg.prim_path
            ):
                if source_path not in self.camera_prim_paths:
                    raise RuntimeError(f"Camera prototype '{source_path}' is not authored on the source stage.")
                for env_id in env_ids:
                    paths_by_env[env_id] = cloner.path.rebase(source_path, source_root, destination.format(env_id))

        if paths_by_env:
            assert plan_env_ids is not None
            missing = [env_id for env_id in plan_env_ids if env_id not in paths_by_env]
            if missing:
                raise RuntimeError(f"Clone plan has no camera source for environments {missing}.")
            camera_paths = [paths_by_env[env_id] for env_id in plan_env_ids]
        else:
            camera_paths = list(self.camera_prim_paths)
        if len(camera_paths) != self.view_count:
            raise RuntimeError(f"Resolved {len(camera_paths)} camera prim paths, expected {self.view_count}.")
        return camera_paths
