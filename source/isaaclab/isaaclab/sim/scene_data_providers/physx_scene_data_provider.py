# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX scene data provider for Omni/PhysX backend."""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any

from pxr import UsdGeom

logger = logging.getLogger(__name__)

# Path pattern for env prims: /World/envs/env_<id>/...
_ENV_ID_RE = re.compile(r"/World/envs/env_(\d+)")


class PhysxSceneDataProvider:
    """Scene data provider for Omni PhysX backend.

    Supports:
    - body poses via PhysX tensor views, with XformPrimView fallback
    - camera poses & intrinsics
    - USD stage handles
    - Newton model/state handles
    - TODO: mesh data access
    """

    # ---- Environment discovery / metadata -------------------------------------------------

    def _env_id_from_path(self, path: str) -> int | None:
        """Extract env id from path (e.g. /World/envs/env_42/...). Used to map body paths to envs for sync."""
        m = _ENV_ID_RE.search(path)
        return int(m.group(1)) if m else None

    def get_num_envs(self) -> int:
        """Return env count from stage discovery, cached once available."""
        if self._num_envs is not None and self._num_envs > 0:
            return self._num_envs
        discovered_num_envs = self._determine_num_envs_in_scene()
        if discovered_num_envs > 0:
            self._num_envs = discovered_num_envs
            return discovered_num_envs
        return 0

    def _determine_num_envs_in_scene(self) -> int:
        """Infer env count from /World/envs/env_<id> prims."""
        if self._stage is None:
            return 0

        max_env_id = -1
        env_name_re = re.compile(r"^env_(\d+)$")

        envs_root = self._stage.GetPrimAtPath("/World/envs")
        if envs_root.IsValid():
            for child in envs_root.GetChildren():
                match = env_name_re.match(child.GetName())
                if match:
                    max_env_id = max(max_env_id, int(match.group(1)))
        return max_env_id + 1 if max_env_id >= 0 else 0

    def __init__(self, visualizer_cfgs: list[Any] | None, stage, simulation_context) -> None:
        from isaacsim.core.simulation_manager import SimulationManager

        self._simulation_context = simulation_context
        self._stage = stage
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        self._rigid_body_view = None
        self._articulation_view = None
        self._xform_views: dict[str, Any] = {}
        self._xform_view_failures: set[str] = set()
        self._view_body_index_map: dict[str, list[int]] = {}

        # Single source of truth: discovered from stage and cached once available.
        self._num_envs: int | None = None

        viz_types = {getattr(cfg, "visualizer_type", None) for cfg in (visualizer_cfgs or [])}
        self._needs_newton_sync = bool({"newton", "rerun"} & viz_types)

        # Fixed metadata for visualizers. get_metadata() returns this plus num_envs so visualizers
        # can .get("num_envs", 0), .get("physics_backend", ...) etc. without the provider exposing many methods.
        self._metadata = {"physics_backend": "omni"}
        if self._stage is None:
            raise RuntimeError(
                "[PhysxSceneDataProvider] USD stage is None and not available from simulation_context. "
                "Ensure the simulation context has a valid stage when using OV/Newton/Rerun visualizers."
            )
        self._up_axis = UsdGeom.GetStageUpAxis(self._stage)
        self._num_envs_at_last_newton_build: int | None = None  # for _refresh_newton_model_if_needed

        self._device = getattr(self._simulation_context, "device", "cuda:0")
        self._newton_model = None
        self._newton_state = None
        self._filtered_newton_model = None
        self._filtered_newton_state = None
        self._filtered_env_ids_key: tuple[int, ...] | None = None
        self._filtered_body_indices: list[int] = []
        self._rigid_body_paths: list[str] = []
        self._articulation_paths: list[str] = []
        self._set_body_q_kernel = None
        # env_id -> list of body indices (in Newton body_key order)
        self._env_id_to_body_indices: dict[int, list[int]] = {}

        # Initialize Newton pipeline only if needed for visualization
        if self._needs_newton_sync:
            self._build_newton_model_from_usd()
            self._build_env_id_to_body_indices()
            self._setup_rigid_body_view()
            self._setup_articulation_view()

    # ---- Newton model + PhysX view setup --------------------------------------------------

    def _wildcard_env_paths(self, paths: list[str]) -> list[str]:
        """Convert /World/envs/env_0 paths to a wildcard pattern when possible."""
        wildcard_paths = [
            path.replace("/World/envs/env_0", "/World/envs/env_*") for path in paths if "/World/envs/env_0" in path
        ]
        return list(dict.fromkeys(wildcard_paths)) if wildcard_paths else paths

    def _refresh_newton_model_if_needed(self) -> None:
        """Rebuild Newton model/state and PhysX views if discovered env count changes."""
        num_envs = self.get_num_envs()
        if num_envs <= 0:
            return

        needs_rebuild = self._newton_model is None or self._newton_state is None
        needs_rebuild = needs_rebuild or (self._num_envs_at_last_newton_build != num_envs)
        if needs_rebuild:
            self._build_newton_model_from_usd()
            self._build_env_id_to_body_indices()
            self._setup_rigid_body_view()
            self._setup_articulation_view()

    def _build_newton_model_from_usd(self) -> None:
        """Build Newton model from USD and cache body/articulation paths."""
        try:
            from newton import ModelBuilder

            builder = ModelBuilder(up_axis=self._up_axis)
            builder.add_usd(self._stage)
            self._newton_model = builder.finalize(device=self._device)
            self._newton_state = self._newton_model.state()

            # Extract scene structure from Newton model (single source of truth)
            self._rigid_body_paths = list(self._newton_model.body_key)
            self._articulation_paths = list(self._newton_model.articulation_key)

            self._xform_views.clear()
            self._view_body_index_map = {}
            self._env_id_to_body_indices = {}
            self._num_envs_at_last_newton_build = self.get_num_envs()
            # Invalidate any filtered model when full model changes.
            self._filtered_newton_model = None
            self._filtered_newton_state = None
            self._filtered_env_ids_key = None
            self._filtered_body_indices = []
        except ModuleNotFoundError as exc:
            logger.error(
                "[PhysxSceneDataProvider] Newton module not available. "
                "Install the Newton backend to use newton/rerun visualizers."
            )
            logger.debug(f"[PhysxSceneDataProvider] Newton import error: {exc}")
        except Exception as exc:
            logger.error(f"[PhysxSceneDataProvider] Failed to build Newton model from USD: {exc}")
            self._newton_model = None
            self._newton_state = None
            self._rigid_body_paths = []
            self._articulation_paths = []
            self._num_envs_at_last_newton_build = None

    def _build_filtered_newton_model(self, env_ids: list[int]) -> None:
        """Build Newton model/state for a subset of envs."""
        try:
            from newton import ModelBuilder

            builder = ModelBuilder(up_axis=self._up_axis)
            builder.add_usd(self._stage, ignore_paths=[r"/World/envs/.*"])
            for env_id in env_ids:
                builder.add_usd(self._stage, root_path=f"/World/envs/env_{env_id}")
            self._filtered_newton_model = builder.finalize(device=self._device)
            self._filtered_newton_state = self._filtered_newton_model.state()

            full_index_by_path = {path: i for i, path in enumerate(self._rigid_body_paths)}
            filtered_paths = list(self._filtered_newton_model.body_key)
            self._filtered_body_indices = []
            missing = []
            for path in filtered_paths:
                idx = full_index_by_path.get(path)
                if idx is None:
                    missing.append(path)
                else:
                    self._filtered_body_indices.append(idx)
            if missing:
                logger.warning(
                    "[PhysxSceneDataProvider] Filtered model contains %d bodies not in full model.",
                    len(missing),
                )
        except ModuleNotFoundError as exc:
            logger.error(
                "[PhysxSceneDataProvider] Newton module not available. "
                "Install the Newton backend to use newton/rerun visualizers."
            )
            logger.debug(f"[PhysxSceneDataProvider] Newton import error: {exc}")
            self._filtered_newton_model = None
            self._filtered_newton_state = None
            self._filtered_body_indices = []
        except Exception as exc:
            logger.error(f"[PhysxSceneDataProvider] Failed to build filtered Newton model from USD: {exc}")
            self._filtered_newton_model = None
            self._filtered_newton_state = None
            self._filtered_body_indices = []

    def _build_env_id_to_body_indices(self) -> None:
        """Build mapping env_id -> list of body indices from rigid_body_paths."""
        self._env_id_to_body_indices = {}
        for body_idx, path in enumerate(self._rigid_body_paths):
            eid = self._env_id_from_path(path)
            if eid is not None:
                self._env_id_to_body_indices.setdefault(eid, []).append(body_idx)

    def _setup_rigid_body_view(self) -> None:
        """Create PhysX RigidBodyView from Newton's body paths.

        Uses body paths extracted from Newton model to create PhysX tensor API view
        for reading rigid body transforms.
        """
        if self._physics_sim_view is None:
            return
        if not self._rigid_body_paths:
            return
        try:
            paths_to_use = self._wildcard_env_paths(self._rigid_body_paths)
            self._rigid_body_view = self._physics_sim_view.create_rigid_body_view(paths_to_use)
            self._cache_view_index_map(self._rigid_body_view, "rigid_body_view")
        except Exception as exc:
            logger.warning(f"[PhysxSceneDataProvider] Failed to create RigidBodyView: {exc}")
            self._rigid_body_view = None

    def _setup_articulation_view(self) -> None:
        """Create PhysX ArticulationView from Newton's articulation paths."""
        if self._physics_sim_view is None:
            return
        if not self._articulation_paths:
            return
        try:
            paths_to_use = self._wildcard_env_paths(self._articulation_paths)
            exprs = [path.replace(".*", "*") for path in paths_to_use]
            self._articulation_view = self._physics_sim_view.create_articulation_view(
                exprs if len(exprs) > 1 else exprs[0]
            )
            self._cache_view_index_map(self._articulation_view, "articulation_view")
        except Exception as exc:
            logger.warning(f"[PhysxSceneDataProvider] Failed to create ArticulationView: {exc}")
            self._articulation_view = None

    # ---- Pose/velocity read pipeline ------------------------------------------------------

    def _get_view_world_poses(self, view):
        """Read world poses from a PhysX view.

        Returns (positions, orientations) or (None, None). The returned tensors
        are expected to be shaped [..., 3] and [..., 4].
        """
        if view is None:
            return None, None
        try:
            # Canonical API for PhysX tensor views.
            transforms = view.get_transforms()
            if hasattr(transforms, "shape") and transforms.shape[-1] == 7:
                return transforms[..., :3], transforms[..., 3:7]
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("[PhysxSceneDataProvider] get_transforms() unavailable/failed for %s: %s", type(view), exc)
        return None, None

    def _cache_view_index_map(self, view, key: str) -> None:
        """Map PhysX view indices to Newton body_key ordering."""
        prim_paths = getattr(view, "prim_paths", None)
        if not prim_paths or not self._rigid_body_paths:
            return

        # Build map: (env_id, relative_path) -> view_index to align view order.
        view_map: dict[tuple[int | None, str], int] = {}
        for view_idx, path in enumerate(prim_paths):
            env_id, rel = self._split_env_relative_path(path)
            view_map[(env_id, rel)] = view_idx

        # Build reordering: newton_body_index -> view_index so we can scatter
        # PhysX view outputs into Newton body ordering.
        order: list[int | None] = [None] * len(self._rigid_body_paths)
        for body_idx, path in enumerate(self._rigid_body_paths):
            env_id, rel = self._split_env_relative_path(path)
            view_idx = view_map.get((env_id, rel))
            if view_idx is None:
                view_idx = view_map.get((None, rel))  # Try without env_id
            order[body_idx] = view_idx

        if all(idx is not None for idx in order):
            self._view_body_index_map[key] = order  # type: ignore[arg-type]

    def _split_env_relative_path(self, path: str) -> tuple[int | None, str]:
        """Extract (env_id, relative_path) from a prim path."""
        match = re.search(r"/World/envs/env_(\d+)(/.*)", path)
        return (int(match.group(1)), match.group(2)) if match else (None, path)

    def _get_view_velocities(self, view):
        """Read linear/angular velocities from a PhysX view."""
        if view is None:
            return None, None

        try:
            # Canonical API for PhysX tensor views.
            result = view.get_velocities()
            if isinstance(result, tuple) and len(result) == 2:
                return result
            if hasattr(result, "shape") and result.shape[-1] == 6:
                return result[..., :3], result[..., 3:6]
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("[PhysxSceneDataProvider] get_velocities() unavailable/failed for %s: %s", type(view), exc)
        return None, None

    def _apply_view_poses(self, view: Any, view_key: str, positions: Any, orientations: Any, covered: Any) -> int:
        """Fill poses from a PhysX view for bodies not yet covered."""
        import torch

        if view is None:
            return 0

        pos, quat = self._get_view_world_poses(view)
        if pos is None or quat is None:
            return 0

        order = self._view_body_index_map.get(view_key)
        if not order:
            return 0

        pos = pos.to(device=self._device, dtype=torch.float32)
        quat = quat.to(device=self._device, dtype=torch.float32)

        # Scatter view outputs into the canonical Newton body order.
        count = 0
        for newton_idx, view_idx in enumerate(order):
            if view_idx is not None and not covered[newton_idx]:
                positions[newton_idx] = pos[view_idx]
                orientations[newton_idx] = quat[view_idx]
                covered[newton_idx] = True
                count += 1

        return count

    def _apply_xform_poses(self, positions: Any, orientations: Any, covered: Any, xform_mask: Any) -> int:
        """Fill remaining poses using XformPrimView (USD fallback).

        This is slower but more robust when PhysX views don't cover all bodies.
        """
        import torch

        from isaaclab.sim.views import XformPrimView

        uncovered = torch.where(~covered)[0].cpu().tolist()
        if not uncovered:
            return 0

        # Query each uncovered prim path directly from USD.
        count = 0
        for idx in uncovered:
            path = self._rigid_body_paths[idx]
            if path in self._xform_view_failures:
                continue
            try:
                if path not in self._xform_views:
                    self._xform_views[path] = XformPrimView(
                        path, device=self._device, stage=self._stage, validate_xform_ops=False
                    )

                pos, quat = self._xform_views[path].get_world_poses()
                if pos is not None and quat is not None:
                    positions[idx] = pos.to(device=self._device, dtype=torch.float32).squeeze()
                    orientations[idx] = quat.to(device=self._device, dtype=torch.float32).squeeze()
                    covered[idx] = True
                    xform_mask[idx] = True
                    count += 1
            except Exception:
                self._xform_view_failures.add(path)
                continue

        return count

    def _convert_xform_quats(self, orientations: Any, xform_mask: Any) -> Any:
        """Return quaternions in xyzw convention.

        PhysX views, XformPrimView, and resolve_prim_pose() in Isaac Lab all use xyzw.
        Keeping this helper as a no-op preserves a single conversion point if conventions
        ever diverge again.
        """
        return orientations

    def _read_poses_from_best_source(self) -> tuple[Any, Any, str, Any] | None:
        """Merge pose data from articulation, rigid-body, and xform views."""
        if self._newton_state is None or not self._rigid_body_paths:
            return None

        import torch

        num_bodies = len(self._rigid_body_paths)
        if num_bodies != self._newton_state.body_q.shape[0]:
            logger.warning(f"Body count mismatch: body_key={num_bodies}, state={self._newton_state.body_q.shape[0]}")
            return None

        # Allocate outputs in Newton body order.
        positions = torch.zeros((num_bodies, 3), dtype=torch.float32, device=self._device)
        orientations = torch.zeros((num_bodies, 4), dtype=torch.float32, device=self._device)
        covered = torch.zeros(num_bodies, dtype=torch.bool, device=self._device)
        xform_mask = torch.zeros(num_bodies, dtype=torch.bool, device=self._device)

        # Apply sources in preferred order: articulation, rigid bodies, then USD fallback.
        articulation_count = self._apply_view_poses(
            self._articulation_view, "articulation_view", positions, orientations, covered
        )
        rigid_count = self._apply_view_poses(self._rigid_body_view, "rigid_body_view", positions, orientations, covered)
        xform_count = self._apply_xform_poses(positions, orientations, covered, xform_mask)

        if not covered.all():
            logger.warning(f"Failed to read {(~covered).sum().item()}/{num_bodies} body poses")
            return None

        active = sum([articulation_count > 0, rigid_count > 0, xform_count > 0])
        source = (
            "merged"
            if active > 1
            else (
                "articulation_view"
                if articulation_count
                else "rigid_body_view" if rigid_count else "xform_view" if xform_count else "none"
            )
        )

        return positions, orientations, source, xform_mask

    def _get_set_body_q_kernel(self):
        """Get or create the Warp kernel for writing transforms to Newton state."""
        if self._set_body_q_kernel is not None:
            return self._set_body_q_kernel
        try:
            import warp as wp

            @wp.kernel(enable_backward=False)
            def _set_body_q(
                positions: wp.array(dtype=wp.vec3),
                orientations: wp.array(dtype=wp.quatf),
                body_q: wp.array(dtype=wp.transformf),
            ):
                i = wp.tid()
                body_q[i] = wp.transformf(positions[i], orientations[i])

            self._set_body_q_kernel = _set_body_q
            return self._set_body_q_kernel
        except Exception as exc:
            logger.warning(f"[PhysxSceneDataProvider] Warp unavailable for Newton state sync: {exc}")
            return None

    def _get_set_body_q_subset_kernel(self):
        """Kernel that writes only body_q at given indices."""
        kernel = getattr(self, "_set_body_q_subset_kernel", None)
        if kernel is not None:
            return kernel
        try:
            import warp as wp

            @wp.kernel(enable_backward=False)
            def _set_body_q_subset(
                positions: wp.array(dtype=wp.vec3),
                orientations: wp.array(dtype=wp.quatf),
                body_indices: wp.array(dtype=wp.int32),
                body_q: wp.array(dtype=wp.transformf),
            ):
                i = wp.tid()
                bi = body_indices[i]
                body_q[bi] = wp.transformf(positions[i], orientations[i])

            self._set_body_q_subset_kernel = _set_body_q_subset
            return self._set_body_q_subset_kernel
        except Exception as exc:
            logger.debug(f"Warp subset kernel: {exc}")
            return None

    # ---- Newton state sync ----------------------------------------------------------------

    def update(self, env_ids: list[int] | None = None) -> None:
        """Sync PhysX transforms to Newton state for visualization.

        When env_ids is not None, only body indices belonging to those envs are written
        (partial sync). When None, all bodies are synced.
        """
        if not self._needs_newton_sync or self._newton_state is None:
            return

        try:
            import warp as wp

            # Re-check env count in case stage population completed after provider construction.
            self._refresh_newton_model_if_needed()

            result = self._read_poses_from_best_source()
            if result is None:
                return

            positions, orientations, _, xform_mask = result
            orientations_xyzw = self._convert_xform_quats(orientations.reshape(-1, 4), xform_mask)

            positions_wp = wp.from_torch(positions.reshape(-1, 3), dtype=wp.vec3)
            orientations_wp = wp.from_torch(orientations_xyzw, dtype=wp.quatf)

            if env_ids is None or not env_ids or not self._env_id_to_body_indices:
                # Fast path: full state sync in one kernel launch.
                set_body_q = self._get_set_body_q_kernel()
                if set_body_q is None or positions_wp.shape[0] != self._newton_state.body_q.shape[0]:
                    return
                wp.launch(
                    set_body_q,
                    dim=positions_wp.shape[0],
                    inputs=[positions_wp, orientations_wp, self._newton_state.body_q],
                    device=self._device,
                )
            else:
                body_indices = []
                for eid in env_ids:
                    body_indices.extend(self._env_id_to_body_indices.get(eid, []))
                if not body_indices:
                    return
                # Subset path: write only env-selected body indices.
                subset_kernel = self._get_set_body_q_subset_kernel()
                if subset_kernel is None:
                    return
                import torch

                indices_t = torch.tensor(body_indices, dtype=torch.int32, device=self._device)
                pos_subset = positions.reshape(-1, 3)[body_indices]
                ori_subset = orientations_xyzw[body_indices]
                indices_wp = wp.from_torch(indices_t, dtype=wp.int32)
                pos_wp = wp.from_torch(pos_subset.contiguous(), dtype=wp.vec3)
                ori_wp = wp.from_torch(ori_subset.contiguous(), dtype=wp.quatf)
                wp.launch(
                    subset_kernel,
                    dim=len(body_indices),
                    inputs=[pos_wp, ori_wp, indices_wp, self._newton_state.body_q],
                    device=self._device,
                )
        except Exception as exc:
            logger.debug(f"Failed to sync transforms to Newton: {exc}")

    def get_newton_model(self) -> Any | None:
        """Return Newton model when sync is enabled."""
        return self._newton_model if self._needs_newton_sync else None

    def get_newton_model_for_env_ids(self, env_ids: list[int] | None) -> Any | None:
        """Return Newton model for a subset of envs if requested."""
        if not self._needs_newton_sync:
            return None
        if env_ids is None:
            return self._newton_model
        env_ids_key = tuple(sorted(env_ids))
        if self._filtered_newton_model is None or self._filtered_env_ids_key != env_ids_key:
            self._filtered_env_ids_key = env_ids_key
            self._build_filtered_newton_model(list(env_ids_key))
        return self._filtered_newton_model

    def get_newton_state(self, env_ids: list[int] | None = None) -> Any | None:
        """Return Newton state when sync is enabled.

        If env_ids is None, returns the full state. If env_ids is provided, returns a
        state-like object whose body_q contains only the bodies for those envs (same order
        as in the full model, for use with e.g. max_worlds=len(env_ids)).
        """
        if not self._needs_newton_sync or self._newton_state is None:
            return None
        if env_ids is None:
            return self._newton_state
        if not self._env_id_to_body_indices:
            return self._create_empty_subset_state()
        env_ids_key = tuple(sorted(env_ids))
        if self._filtered_newton_model is not None and self._filtered_env_ids_key == env_ids_key:
            if not self._filtered_body_indices:
                return self._create_empty_subset_state()
            try:
                import warp as wp

                body_q_t = wp.to_torch(self._newton_state.body_q)
                subset = body_q_t[self._filtered_body_indices].clone()
                self._filtered_newton_state.body_q = wp.from_torch(subset, dtype=wp.transformf)
                return self._filtered_newton_state
            except Exception:
                return self._newton_state
        body_indices = []
        for eid in env_ids:
            body_indices.extend(self._env_id_to_body_indices.get(eid, []))
        if not body_indices:
            return self._create_empty_subset_state()

        body_q = self._newton_state.body_q
        try:
            import warp as wp

            body_q_t = wp.to_torch(body_q)
            body_q_subset = body_q_t[body_indices].clone()
        except Exception:
            return self._newton_state
        return self._create_subset_state(body_q_subset)

    def _create_empty_subset_state(self):
        """Return a minimal state-like object with empty body_q."""
        if self._newton_state is None:
            return None
        try:
            import warp as wp

            body_q_t = wp.to_torch(self._newton_state.body_q)
            empty = body_q_t[:0].clone()
            return self._create_subset_state(empty)
        except Exception:
            return self._newton_state

    # ---- Newton subset helpers -------------------------------------------------------------

    def _create_subset_state(self, body_q_subset):
        """Return a minimal state-like object for subset rendering."""
        import warp as wp

        if hasattr(body_q_subset, "device") and not isinstance(body_q_subset, wp.array):
            body_q_subset = wp.from_torch(body_q_subset, dtype=wp.transformf)

        class _SubsetState:
            pass

        s = _SubsetState()
        s.body_q = body_q_subset
        return s

    # ---- Public provider API ---------------------------------------------------------------

    def get_usd_stage(self) -> Any:
        """Return the USD stage handle."""
        if self._stage is not None:
            return self._stage
        return getattr(self._simulation_context, "stage", None)

    def get_camera_transforms(self) -> dict[str, Any] | None:
        """Return per-camera, per-env transforms (positions, orientations)."""
        if self._stage is None:
            return None

        import isaaclab.sim as isaaclab_sim

        env_pattern = re.compile(r"(?P<root>/World/envs/env_)(?P<id>\d+)(?P<path>/.*)")
        shared_paths: list[str] = []
        instances: dict[str, list[tuple[int, str]]] = {}
        num_envs = -1

        # Breadth-first walk so we discover camera prims across the full stage.
        stage_prims = deque([self._stage.GetPseudoRoot()])
        while stage_prims:
            prim = stage_prims.popleft()
            prim_path = prim.GetPath().pathString

            world_id = 0
            template_path = prim_path
            if match := env_pattern.match(prim_path):
                # Normalize per-env path to a shared template key (env_%d/...) so
                # visualizers can query one camera path for all env instances.
                world_id = int(match.group("id"))
                template_path = match.group("root") + "%d" + match.group("path")
                if world_id > num_envs:
                    num_envs = world_id

            imageable = UsdGeom.Imageable(prim)
            if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue

            if prim.IsA(UsdGeom.Camera):
                if template_path not in instances:
                    instances[template_path] = []
                instances[template_path].append((world_id, prim_path))
                if template_path not in shared_paths:
                    shared_paths.append(template_path)

            if hasattr(UsdGeom, "TraverseInstanceProxies"):
                child_prims = prim.GetFilteredChildren(UsdGeom.TraverseInstanceProxies())
            else:
                child_prims = prim.GetChildren()
            if child_prims:
                stage_prims.extend(child_prims)

        num_envs += 1
        positions: list[list[list[float] | None]] = []
        orientations: list[list[list[float] | None]] = []

        for template_path in shared_paths:
            per_world_pos: list[list[float] | None] = [None] * num_envs
            per_world_ori: list[list[float] | None] = [None] * num_envs
            for world_id, prim_path in instances.get(template_path, []):
                if world_id < 0 or world_id >= num_envs:
                    continue
                prim = self._stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue
                pos, ori = isaaclab_sim.resolve_prim_pose(prim)
                per_world_pos[world_id] = [float(pos[0]), float(pos[1]), float(pos[2])]
                per_world_ori[world_id] = [float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])]
            positions.append(per_world_pos)
            orientations.append(per_world_ori)

        return {"order": shared_paths, "positions": positions, "orientations": orientations, "num_envs": num_envs}

    def get_metadata(self) -> dict[str, Any]:
        """Return metadata for visualizers (num_envs, physics_backend, etc.)."""
        out = dict(self._metadata)
        out["num_envs"] = self.get_num_envs()
        return out

    def get_transforms(self) -> dict[str, Any] | None:
        """Return merged body transforms from available PhysX views."""
        try:
            result = self._read_poses_from_best_source()
            if result is None:
                return None

            positions, orientations, _, xform_mask = result
            orientations_xyzw = self._convert_xform_quats(orientations, xform_mask)
            return {"positions": positions, "orientations": orientations_xyzw}
        except Exception:
            return None

    def get_velocities(self) -> dict[str, Any] | None:
        """Return linear/angular velocities from available PhysX views."""
        for source, view in (
            ("articulation_view", self._articulation_view),
            ("rigid_body_view", self._rigid_body_view),
        ):
            linear, angular = self._get_view_velocities(view)
            if linear is not None and angular is not None:
                return {"linear": linear, "angular": angular, "source": source}
        return None

    def get_contacts(self) -> dict[str, Any] | None:
        """Contacts not yet supported for PhysX provider."""
        pass
