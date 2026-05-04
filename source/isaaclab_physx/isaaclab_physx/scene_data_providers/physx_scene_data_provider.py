# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX scene data provider for Omni/PhysX backend."""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from typing import Any

import torch
import warp as wp

from pxr import UsdGeom, UsdPhysics

from isaaclab.physics.base_scene_data_provider import BaseSceneDataProvider
from isaaclab.sim.utils.newton_model_utils import replace_newton_shape_colors

logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def _set_body_q_kernel(
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quatf),
    body_q: wp.array(dtype=wp.transformf),
):
    """Write pose arrays into Newton ``body_q`` in one-to-one index order."""
    i = wp.tid()
    body_q[i] = wp.transformf(positions[i], orientations[i])


class PhysxSceneDataProvider(BaseSceneDataProvider):
    """Scene data provider for Omni PhysX backend.

    Supports:
    - body poses via PhysX tensor views, with FrameView fallback
    - camera poses & intrinsics
    - USD stage handles
    - Newton model/state (built locally from the scene's per-group :class:`ClonePlan` map
      when required)
    """

    # ---- Environment discovery / metadata -------------------------------------------------

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

    def __init__(self, stage, simulation_context) -> None:
        """Initialize the PhysX scene data provider.

        Args:
            stage: USD stage handle.
            simulation_context: Active simulation context.
        """
        from isaaclab_physx.physics import PhysxManager as SimulationManager

        self._simulation_context = simulation_context
        self._stage = stage
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        self._rigid_body_view = None
        self._xform_views: dict[str, Any] = {}
        self._xform_view_failures: set[str] = set()
        self._view_body_index_map: dict[str, list[int]] = {}
        self._warned_once: set[str] = set()

        # Single source of truth: discovered from stage and cached once available.
        self._num_envs: int | None = None

        # Determine if newton model sync is required for selected renderers and visualizers
        requirements = self._simulation_context.get_scene_data_requirements()
        self._needs_newton_sync = bool(requirements.requires_newton_model)

        # Fixed metadata for visualizers. get_metadata() returns this plus num_envs so visualizers
        # can .get("num_envs", 0), .get("physics_backend", ...) etc. without the provider exposing many methods.
        self._metadata = {"physics_backend": "omni"}
        if self._stage is None:
            raise RuntimeError(
                "[PhysxSceneDataProvider] USD stage is None and not available from simulation_context. "
                "Ensure the simulation context has a valid stage when using OV/Newton/Rerun/Viser visualizers."
            )
        self._num_envs_at_last_newton_build: int | None = None  # for _refresh_newton_model_if_needed

        self._device = getattr(self._simulation_context, "device", "cuda:0")
        self._newton_model = None
        self._newton_state = None
        self._rigid_body_paths: list[str] = []
        # Paths used to create PhysX views. May include articulation roots for coverage.
        self._rigid_body_view_paths: list[str] = []

        # Reused pose buffers (MR perf): avoid per-call allocations in _read_poses_from_best_source.
        self._pose_buf_num_bodies = 0
        self._positions_buf = None
        self._orientations_buf = None
        self._covered_buf = None
        self._xform_mask_buf = None
        # View index order as device tensors for vectorized scatter in _apply_view_poses.
        self._view_order_tensors: dict[str, Any] = {}
        # Last load outcome (tests / debug): "built" | "missing" | "error".
        self._last_newton_model_build_source: str | None = None
        self._last_newton_model_build_elapsed_ms: float | None = None

        if self._needs_newton_sync:
            self._build_newton_model_from_clone_plans()
            self._setup_rigid_body_view()

    # ---- Newton model + PhysX view setup --------------------------------------------------

    def _wildcard_env_paths(self, paths: list[str]) -> list[str]:
        """Convert /World/envs/env_0 paths to a wildcard pattern when possible."""
        wildcard_paths = [
            path.replace("/World/envs/env_0", "/World/envs/env_*") for path in paths if "/World/envs/env_0" in path
        ]
        return list(dict.fromkeys(wildcard_paths)) if wildcard_paths else paths

    def _refresh_newton_model_if_needed(self) -> None:
        """Reload Newton model/state and PhysX views when the discovered env count changes."""
        num_envs = self.get_num_envs()
        if num_envs <= 0:
            return

        needs_rebuild = self._newton_model is None or self._newton_state is None
        needs_rebuild = needs_rebuild or (self._num_envs_at_last_newton_build != num_envs)
        if needs_rebuild:
            self._build_newton_model_from_clone_plans()
            self._setup_rigid_body_view()

    def _build_newton_model_from_clone_plans(self) -> None:
        """Build Newton model and state from the scene's per-group :class:`ClonePlan` map.

        Reads plans :meth:`InteractiveScene.clone_environments` publishes on
        :class:`SimulationContext`, derives the flat ``(sources, destinations, mask)`` shape
        :func:`isaaclab_newton.cloner.newton_visualizer_prebuild` expects, and caches the
        resulting model/state. Per-prototype source paths recover as
        ``dest_template.format(<first env using this prototype>)``; per-env positions are
        read off ``xformOp:translate`` on the env-level prims derived from the same template.
        Pre-condition violations raise :class:`RuntimeError` (logged as ``"missing"``);
        ``isaaclab_newton`` being absent (optional dep) maps to ``"missing"`` via the
        import's own exception types; unexpected failures fall through to ``"error"``.
        """
        start_t = time.perf_counter()
        source = "missing"
        try:
            plans = self._simulation_context.get_clone_plans()
            if not plans:
                raise RuntimeError("No clone plans on simulation context.")
            from isaaclab_newton.cloner.newton_replicate import newton_visualizer_prebuild

            # Flatten per-group plans into one (sources, destinations, mask) bundle. Source
            # paths recover via ``dest_template.format(<first env using this prototype>)``;
            # all-False rows are dropped (possible when ``num_prototypes > num_envs``).
            plan_list = list(plans.values())
            num_envs = plan_list[0].clone_mask.size(1)
            if any(p.clone_mask.size(1) != num_envs for p in plan_list):
                raise RuntimeError(f"Clone plans disagree on num_envs: {[p.clone_mask.size(1) for p in plan_list]}")
            sources, destinations, mask_rows = [], [], []
            for p in plan_list:
                for i in range(p.clone_mask.size(0)):
                    nz = p.clone_mask[i].nonzero(as_tuple=False)
                    if nz.numel() == 0:
                        continue
                    sources.append(p.dest_template.format(int(nz[0].item())))
                    destinations.append(p.dest_template)
                    mask_rows.append(p.clone_mask[i : i + 1])
            if not sources:
                raise RuntimeError("All clone-plan prototype rows are empty.")
            mask = torch.cat(mask_rows, dim=0)

            # Env-level path template = dest_template up to the first ``{}``. Per-env world
            # positions: xformOp:translate read off each env prim; missing prims fall through.
            env_path_template = plan_list[0].dest_template.split("{}")[0] + "{}"
            positions = torch.zeros((num_envs, 3), dtype=torch.float32, device=self._device)
            for i in range(num_envs):
                prim = self._stage.GetPrimAtPath(env_path_template.format(i))
                if prim.IsValid() and (v := prim.GetAttribute("xformOp:translate").Get()) is not None:
                    positions[i] = torch.tensor([v[0], v[1], v[2]], device=self._device)

            model, state = newton_visualizer_prebuild(
                stage=self._stage,
                sources=sources,
                destinations=destinations,
                env_ids=torch.arange(num_envs, dtype=torch.long, device=mask.device),
                mapping=mask,
                positions=positions,
                device=self._device,
                up_axis=UsdGeom.GetStageUpAxis(self._stage),
            )
            if model is None or state is None:
                raise RuntimeError("newton_visualizer_prebuild returned None.")

            self._newton_model, self._newton_state = model, state
            replace_newton_shape_colors(self._newton_model, self._stage)
            # Newton renamed ``*_key`` → ``*_label`` mid-development; fall back so we work either way.
            # ``dict.fromkeys`` preserves order while deduping — articulation roots can overlap rigid bodies.
            label_or_key = lambda kind: list(getattr(model, f"{kind}_label", None) or getattr(model, f"{kind}_key", []))  # noqa: E731
            self._rigid_body_paths = label_or_key("body")
            self._rigid_body_view_paths = list(dict.fromkeys(self._rigid_body_paths + label_or_key("articulation")))
            # Reset cached views/buffers; rebuilt lazily by ``_setup_rigid_body_view``.
            self._xform_views.clear()
            self._view_order_tensors.clear()
            self._view_body_index_map = {}
            self._pose_buf_num_bodies = 0
            self._positions_buf = self._orientations_buf = self._covered_buf = self._xform_mask_buf = None
            self._num_envs_at_last_newton_build = num_envs
            source = "built"
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning("[PhysxSceneDataProvider] isaaclab_newton not available: %s", exc)
            self._clear_newton_model_state()
        except RuntimeError as exc:
            logger.error("[PhysxSceneDataProvider] %s", exc)
            self._clear_newton_model_state()
        except Exception as exc:
            source = "error"
            logger.error("[PhysxSceneDataProvider] Failed to build Newton model from clone plans: %s", exc)
            self._clear_newton_model_state()
        finally:
            self._last_newton_model_build_elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            self._last_newton_model_build_source = source
            logger.debug(
                "[PhysxSceneDataProvider] Newton model build source=%s elapsed_ms=%.2f",
                source,
                self._last_newton_model_build_elapsed_ms,
            )

    def _clear_newton_model_state(self) -> None:
        """Clear cached Newton model, state, and rigid-body path lists."""
        self._newton_model = None
        self._newton_state = None
        self._rigid_body_paths = []
        self._rigid_body_view_paths = []
        self._num_envs_at_last_newton_build = None

    def _setup_rigid_body_view(self) -> None:
        """Create PhysX RigidBodyView from Newton's body paths.

        Uses body paths extracted from Newton model to create PhysX tensor API view
        for reading rigid body transforms.
        """
        if self._physics_sim_view is None:
            return
        paths = self._rigid_body_view_paths or self._rigid_body_paths
        if not paths:
            return
        # Defensive: only pass true rigid-body prims into PhysX RigidBodyView.
        # Some prebuilt artifacts carry articulation root paths for coverage, but
        # those roots are not guaranteed to be rigid-body prims and can trip native
        # view creation paths on some tasks.
        rigid_paths: list[str] = []
        dropped_non_rigid = 0
        for path in paths:
            prim = self._stage.GetPrimAtPath(path) if self._stage is not None else None
            if prim and prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_paths.append(path)
            else:
                dropped_non_rigid += 1
        if not rigid_paths:
            self._warn_once(
                "rigid-view-no-rigid-paths",
                "[PhysxSceneDataProvider] No rigid-body prim paths available for RigidBodyView creation.",
                level=logging.WARNING,
            )
            return
        try:
            paths_to_use = self._wildcard_env_paths(rigid_paths)
            self._rigid_body_view = self._physics_sim_view.create_rigid_body_view(paths_to_use)
            self._cache_view_index_map(self._rigid_body_view, "rigid_body_view")
        except Exception as exc:
            logger.warning(f"[PhysxSceneDataProvider] Failed to create RigidBodyView: {exc}")
            self._rigid_body_view = None

    # ---- Pose/velocity read pipeline ------------------------------------------------------

    def _warn_once(self, key: str, message: str, *args, level=logging.WARNING) -> None:
        """Log a warning only once for a given key."""
        if key in self._warned_once:
            return
        self._warned_once.add(key)
        logger.log(level, message, *args)

    def _get_view_world_poses(self, view: Any):
        """Read world poses from a PhysX view."""
        if view is None:
            return None, None

        result = view.get_transforms()
        if isinstance(result, tuple) and len(result) == 2:
            return result
        if hasattr(result, "shape"):
            return result[:, :3], result[:, 3:7]

        import warp as wp

        result_t = wp.to_torch(result)
        return result_t[:, :3], result_t[:, 3:7]

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
            # Cache as device tensor for vectorized scatter in _apply_view_poses.
            import torch

            self._view_order_tensors[key] = torch.tensor(order, dtype=torch.long, device=self._device)

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
        import warp as wp

        if view is None:
            return 0

        pos, quat = self._get_view_world_poses(view)
        if pos is None or quat is None:
            return 0

        order = self._view_body_index_map.get(view_key)
        if not order:
            return 0

        # Normalize returned arrays to torch tensors across backends (torch/warp/other).
        if not isinstance(pos, torch.Tensor):
            try:
                pos = wp.to_torch(pos)
            except Exception:
                pos = torch.as_tensor(pos)
        if not isinstance(quat, torch.Tensor):
            try:
                quat = wp.to_torch(quat)
            except Exception:
                quat = torch.as_tensor(quat)

        pos = pos.to(device=self._device, dtype=torch.float32)
        quat = quat.to(device=self._device, dtype=torch.float32)

        # Vectorized scatter when we have a cached order tensor (view fully covers bodies).
        order_t = self._view_order_tensors.get(view_key)
        if order_t is not None:
            uncovered_mask = ~covered
            if uncovered_mask.any():
                newton_indices = uncovered_mask.nonzero(as_tuple=True)[0]
                view_indices = order_t[newton_indices]
                positions[newton_indices] = pos[view_indices]
                orientations[newton_indices] = quat[view_indices]
                covered[newton_indices] = True
                return newton_indices.numel()
            return 0

        # Per-index path when the view does not fully cover bodies or the order cache is missing.
        count = 0
        for newton_idx, view_idx in enumerate(order):
            if view_idx is not None and not covered[newton_idx]:
                positions[newton_idx] = pos[view_idx]
                orientations[newton_idx] = quat[view_idx]
                covered[newton_idx] = True
                count += 1

        return count

    def _apply_xform_poses(self, positions: Any, orientations: Any, covered: Any, xform_mask: Any) -> int:
        """Fill remaining body poses using ``XformPrimView`` for prims not covered by the rigid-body view."""
        import torch

        from isaaclab.sim.views import FrameView

        uncovered = torch.where(~covered)[0].cpu().tolist()
        if not uncovered:
            return 0

        # Query each uncovered prim path directly from USD.
        count = 0
        for idx in uncovered:
            path = self._rigid_body_paths[idx]
            try:
                if path not in self._xform_views:
                    self._xform_views[path] = FrameView(
                        path, device=self._device, stage=self._stage, validate_xform_ops=False
                    )

                pos_w, quat_w = self._xform_views[path].get_world_poses()
                if pos_w is not None and quat_w is not None:
                    positions[idx] = pos_w.torch.to(device=self._device, dtype=torch.float32).squeeze()
                    orientations[idx] = quat_w.torch.to(device=self._device, dtype=torch.float32).squeeze()
                    covered[idx] = True
                    xform_mask[idx] = True
                    count += 1
            except Exception:
                self._xform_view_failures.add(path)
                continue

        if len(self._xform_view_failures) > 0:
            self._warn_once(
                "xform-fallback-failures",
                "[PhysxSceneDataProvider] XformPrimView reads failed for %d body paths.",
                len(self._xform_view_failures),
                level=logging.DEBUG,
            )
        return count

    def _convert_xform_quats(self, orientations: Any, xform_mask: Any) -> Any:
        """Return quaternions in xyzw convention.

        PhysX views, FrameView, and resolve_prim_pose() in Isaac Lab all use xyzw.
        Keeping this helper as a no-op preserves a single conversion point if conventions
        ever diverge again.
        """
        return orientations

    def _read_poses_from_best_source(self) -> tuple[Any, Any, str, Any] | None:
        """Merge pose data from rigid-body and xform views."""
        if self._newton_state is None or not self._rigid_body_paths:
            return None

        import torch

        num_bodies = len(self._rigid_body_paths)
        if num_bodies != self._newton_state.body_q.shape[0]:
            self._warn_once(
                "body-count-mismatch",
                "[PhysxSceneDataProvider] Body count mismatch: body_key=%d, state=%d",
                num_bodies,
                int(self._newton_state.body_q.shape[0]),
            )
            return None

        # Reuse buffers when size unchanged to avoid per-call allocations (MR perf).
        if num_bodies != self._pose_buf_num_bodies or self._positions_buf is None:
            self._pose_buf_num_bodies = num_bodies
            self._positions_buf = torch.zeros((num_bodies, 3), dtype=torch.float32, device=self._device)
            self._orientations_buf = torch.zeros((num_bodies, 4), dtype=torch.float32, device=self._device)
            self._covered_buf = torch.zeros(num_bodies, dtype=torch.bool, device=self._device)
            self._xform_mask_buf = torch.zeros(num_bodies, dtype=torch.bool, device=self._device)
        else:
            self._covered_buf.zero_()
            self._xform_mask_buf.zero_()

        positions = self._positions_buf
        orientations = self._orientations_buf
        covered = self._covered_buf
        xform_mask = self._xform_mask_buf

        rigid_count = self._apply_view_poses(self._rigid_body_view, "rigid_body_view", positions, orientations, covered)
        xform_count = self._apply_xform_poses(positions, orientations, covered, xform_mask)
        if rigid_count == 0:
            self._warn_once(
                "rigid-source-unused",
                (
                    "[PhysxSceneDataProvider] RigidBodyView returned no transforms; "
                    "filled from XformPrimView where needed."
                ),
                level=logging.DEBUG,
            )

        if not covered.all():
            self._warn_once(
                "pose-read-incomplete",
                "[PhysxSceneDataProvider] Failed to read %d/%d body poses.",
                int((~covered).sum().item()),
                num_bodies,
            )
            return None

        active = sum([rigid_count > 0, xform_count > 0])
        source = (
            "merged" if active > 1 else ("rigid_body_view" if rigid_count else "xform_view" if xform_count else "none")
        )
        return positions, orientations, source, xform_mask

    def _get_set_body_q_kernel(self):
        """Return module-level Warp kernel for writing transforms to Newton state."""
        return _set_body_q_kernel

    # ---- Newton state sync ----------------------------------------------------------------

    def update(self) -> None:
        """Sync PhysX transforms into the full Newton state (one kernel launch)."""
        if not self._needs_newton_sync or self._newton_state is None:
            return

        try:
            # Re-check env count in case stage population completed after provider construction.
            self._refresh_newton_model_if_needed()

            result = self._read_poses_from_best_source()
            if result is None:
                return

            positions, orientations, _, xform_mask = result
            orientations_xyzw = self._convert_xform_quats(orientations.reshape(-1, 4), xform_mask)

            positions_wp = wp.from_torch(positions.reshape(-1, 3), dtype=wp.vec3)
            orientations_wp = wp.from_torch(orientations_xyzw, dtype=wp.quatf)

            set_body_q = self._get_set_body_q_kernel()
            if set_body_q is None or positions_wp.shape[0] != self._newton_state.body_q.shape[0]:
                return
            wp.launch(
                set_body_q,
                dim=positions_wp.shape[0],
                inputs=[positions_wp, orientations_wp, self._newton_state.body_q],
                device=self._device,
            )
        except Exception as exc:
            self._warn_once(
                "newton-sync-update-failed",
                "[PhysxSceneDataProvider] Failed to sync transforms to Newton state: %s",
                exc,
            )

    def get_newton_model(self) -> Any | None:
        """Return Newton model when sync is enabled.

        Returns:
            Newton model object, or ``None`` when unavailable.
        """
        return self._newton_model if self._needs_newton_sync else None

    def get_newton_state(self) -> Any | None:
        """Return full Newton state when sync is enabled."""
        if not self._needs_newton_sync or self._newton_state is None:
            return None
        return self._newton_state

    # ---- Public provider API ---------------------------------------------------------------

    def get_usd_stage(self) -> Any:
        """Return USD stage handle.

        Returns:
            USD stage object.
        """
        if self._stage is not None:
            return self._stage
        return getattr(self._simulation_context, "stage", None)

    def get_camera_transforms(self) -> dict[str, Any] | None:
        """Return per-camera, per-environment transforms.

        Returns:
            Dictionary containing camera order, positions, orientations, and environment count,
            or ``None`` when unavailable.
        """
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
        """Return provider metadata for visualizers and renderers.

        Returns:
            Metadata dictionary with backend and environment count.
        """
        out = dict(self._metadata)
        out["num_envs"] = self.get_num_envs()
        return out

    def get_transforms(self) -> dict[str, Any] | None:
        """Return merged body transforms from available PhysX views.

        Returns:
            Dictionary with positions/orientations, or ``None`` when unavailable.
        """
        try:
            result = self._read_poses_from_best_source()
            if result is None:
                return None

            positions, orientations, _, xform_mask = result
            orientations_xyzw = self._convert_xform_quats(orientations, xform_mask)
            return {"positions": positions, "orientations": orientations_xyzw}
        except Exception as exc:
            self._warn_once(
                "get-transforms-failed",
                "[PhysxSceneDataProvider] get_transforms() failed: %s",
                exc,
            )
            return None

    def get_velocities(self) -> dict[str, Any] | None:
        """Return linear/angular velocities from available PhysX views.

        Returns:
            Dictionary with linear/angular velocities, or ``None`` when unavailable.
        """
        for source, view in (("rigid_body_view", self._rigid_body_view),):
            linear, angular = self._get_view_velocities(view)
            if linear is not None and angular is not None:
                return {"linear": linear, "angular": angular, "source": source}
        return None

    def get_contacts(self) -> dict[str, Any] | None:
        """Return contact data for PhysX provider.

        Returns:
            ``None`` because contacts are not currently implemented in this provider.
        """
        return None
