# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OV (Omniverse) scene data provider for Omni PhysX backend."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class OVSceneDataProvider:
    """Scene data provider for Omni PhysX backend."""

    def __init__(self, visualizer_cfgs: list[Any] | None, stage, simulation_context) -> None:
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import UsdGeom

        self._stage = stage
        self._simulation_context = simulation_context
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        self._rigid_body_view = None
        self._articulation_view = None
        self._xform_views: dict[str, Any] = {}
        self._body_key_index_map: dict[str, int] = {}
        self._view_body_index_map: dict[str, list[int]] = {}

        # Determine which visualizers need Newton state sync
        self._has_newton_visualizer = False
        self._has_rerun_visualizer = False
        self._has_ov_visualizer = False
        if visualizer_cfgs:
            for cfg in visualizer_cfgs:
                viz_type = getattr(cfg, "visualizer_type", None)
                if viz_type == "newton":
                    self._has_newton_visualizer = True
                elif viz_type == "rerun":
                    self._has_rerun_visualizer = True
                elif viz_type == "omniverse":
                    self._has_ov_visualizer = True

        # Explicit mode flag for Newton synchronization
        self._needs_newton_sync = self._has_newton_visualizer or self._has_rerun_visualizer

        self._metadata = {
            "physics_backend": "omni",
            "num_envs": self._get_num_envs(),
            "gravity_vector": tuple(self._simulation_context.cfg.gravity),
            "clone_physics_only": False,
        }

        self._device = getattr(self._simulation_context, "device", "cuda:0")
        self._newton_model = None
        self._newton_state = None
        self._rigid_body_paths: list[str] = []
        self._articulation_paths: list[str] = []
        self._set_body_q_kernel = None
        self._up_axis = UsdGeom.GetStageUpAxis(self._stage)

        # Initialize Newton pipeline only if needed for visualization
        if self._needs_newton_sync:
            self._build_newton_model_from_usd()
            self._setup_rigid_body_view()
            self._setup_articulation_view()
        else:
            logger.info("[OVSceneDataProvider] OV visualizer only - skipping Newton model build")

    def _get_num_envs(self) -> int:
        try:
            import carb

            carb_settings_iface = carb.settings.get_settings()
            num_envs = carb_settings_iface.get("/isaaclab/scene/num_envs")
            if num_envs:
                return int(num_envs)
        except Exception:
            return 0
        return 0

    @staticmethod
    def _wildcard_env_paths(paths: list[str]) -> list[str]:
        wildcard_paths = []
        for path in paths:
            if "/World/envs/env_0" in path:
                wildcard_paths.append(path.replace("/World/envs/env_0", "/World/envs/env_*"))
        return list(dict.fromkeys(wildcard_paths)) if wildcard_paths else paths

    def _refresh_newton_model_if_needed(self) -> None:
        num_envs = self._get_num_envs()
        if num_envs <= 0:
            return

        if self._newton_model is None or self._newton_state is None:
            self._build_newton_model_from_usd()
            self._setup_rigid_body_view()
            self._setup_articulation_view()
            return

        if self._metadata.get("num_envs", 0) != num_envs:
            self._build_newton_model_from_usd()
            self._setup_rigid_body_view()
            self._setup_articulation_view()
            return

    def _build_newton_model_from_usd(self) -> None:
        """Build Newton model from USD and extract scene structure."""
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
            self._body_key_index_map = {path: i for i, path in enumerate(self._rigid_body_paths)}
            self._view_body_index_map = {}
        except ModuleNotFoundError as exc:
            logger.error(
                "[SceneDataProvider] Newton module not available. "
                "Install the Newton backend to use newton/rerun visualizers."
            )
            logger.debug(f"[SceneDataProvider] Newton import error: {exc}")
        except Exception as exc:
            logger.error(f"[SceneDataProvider] Failed to build Newton model from USD: {exc}")
            self._newton_model = None
            self._newton_state = None
            self._rigid_body_paths = []
            self._articulation_paths = []

    def _setup_rigid_body_view(self) -> None:
        """Create PhysX RigidBodyView from Newton's body paths.

        Uses body paths extracted from Newton model to create PhysX tensor API view
        for reading rigid body transforms.
        """
        if not self._rigid_body_paths:
            return
        try:
            paths_to_use = self._wildcard_env_paths(self._rigid_body_paths)
            self._rigid_body_view = self._physics_sim_view.create_rigid_body_view(paths_to_use)
            self._cache_view_index_map(self._rigid_body_view, "rigid_body_view")
        except Exception as exc:
            logger.warning(f"[SceneDataProvider] Failed to create RigidBodyView: {exc}")
            self._rigid_body_view = None

    def _setup_articulation_view(self) -> None:
        """Create PhysX ArticulationView from Newton's articulation paths."""
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
            logger.warning(f"[SceneDataProvider] Failed to create ArticulationView: {exc}")
            self._articulation_view = None

    def _get_view_world_poses(self, view):
        """Read world poses from PhysX tensor API view (ArticulationView or RigidBodyView).

        Tries multiple method names for compatibility across PhysX API versions.
        Returns (positions, orientations) tuple or (None, None) if unavailable.
        """
        if view is None:
            return None, None

        method_names = ("get_world_poses", "get_world_transforms", "get_transforms", "get_poses")

        for name in method_names:
            method = getattr(view, name, None)
            if method is None:
                continue
            try:
                result = method()
            except Exception:
                continue

            # Handle tuple return: (positions, orientations)
            if isinstance(result, tuple) and len(result) == 2:
                return result

            # Handle packed array: [..., 7] -> split into pos and quat
            try:
                if hasattr(result, "shape") and result.shape[-1] == 7:
                    positions = result[..., :3]
                    orientations = result[..., 3:7]
                    return positions, orientations
            except Exception:
                continue

        return None, None

    def _cache_view_index_map(self, view, key: str) -> None:
        """Map PhysX view indices to Newton body_key ordering."""
        prim_paths = getattr(view, "prim_paths", None)
        if not prim_paths or not self._rigid_body_paths:
            return

        def split_env(path: str) -> tuple[int | None, str]:
            """Extract environment ID and relative path from prim path."""
            match = re.search(r"/World/envs/env_(\d+)(/.*)", path)
            return (int(match.group(1)), match.group(2)) if match else (None, path)

        # Build map: (env_id, relative_path) -> view_index
        view_map: dict[tuple[int | None, str], int] = {}
        for view_idx, path in enumerate(prim_paths):
            env_id, rel = split_env(path)
            view_map[(env_id, rel)] = view_idx

        # Build reordering: newton_body_index -> view_index
        order: list[int | None] = [None] * len(self._rigid_body_paths)
        for body_idx, path in enumerate(self._rigid_body_paths):
            env_id, rel = split_env(path)
            view_idx = view_map.get((env_id, rel))
            if view_idx is None:
                view_idx = view_map.get((None, rel))  # Try without env_id
            order[body_idx] = view_idx

        if all(idx is not None for idx in order):
            self._view_body_index_map[key] = order  # type: ignore[arg-type]

    def _get_view_velocities(self, view):
        if view is None:
            return None, None

        method = getattr(view, "get_velocities", None)
        if method is not None:
            try:
                result = method()
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                if hasattr(result, "shape") and result.shape[-1] == 6:
                    return result[..., :3], result[..., 3:6]
            except Exception:
                pass

        get_linear = getattr(view, "get_linear_velocities", None)
        get_angular = getattr(view, "get_angular_velocities", None)
        if get_linear is not None and get_angular is not None:
            try:
                return get_linear(), get_angular()
            except Exception:
                return None, None

        return None, None

    def _apply_view_poses(self, view: Any, view_key: str, positions: Any, orientations: Any, covered: Any) -> int:
        """Read poses from a PhysX view and write uncovered bodies to output tensors."""
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

        count = 0
        for newton_idx, view_idx in enumerate(order):
            if view_idx is not None and not covered[newton_idx]:
                positions[newton_idx] = pos[view_idx]
                orientations[newton_idx] = quat[view_idx]
                covered[newton_idx] = True
                count += 1

        return count

    def _apply_xform_poses(self, positions: Any, orientations: Any, covered: Any, xform_mask: Any) -> int:
        """Use XformPrimView fallback for remaining uncovered bodies."""
        import torch

        from isaaclab.sim.views import XformPrimView

        uncovered = torch.where(~covered)[0].cpu().tolist()
        if not uncovered:
            return 0

        count = 0
        for idx in uncovered:
            path = self._rigid_body_paths[idx]
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
                continue

        return count

    def _convert_xform_quats(self, orientations: Any, xform_mask: Any) -> Any:
        """Convert XformPrimView quaternions from wxyz to xyzw where needed."""
        if not xform_mask.any():
            return orientations

        import torch

        from isaaclab.utils.math import convert_quat

        orientations_xyzw = orientations.clone()
        xform_indices = torch.where(xform_mask)[0]
        if len(xform_indices) > 0:
            orientations_xyzw[xform_indices] = convert_quat(orientations[xform_indices], to="xyzw")
        return orientations_xyzw

    def _read_poses_from_best_source(self) -> tuple[Any, Any, str, Any] | None:
        """Merge pose data from ArticulationView, RigidBodyView, and XformPrimView."""
        if self._newton_state is None or not self._rigid_body_paths:
            return None

        import torch

        num_bodies = len(self._rigid_body_paths)
        if num_bodies != self._newton_state.body_q.shape[0]:
            logger.warning(f"Body count mismatch: body_key={num_bodies}, state={self._newton_state.body_q.shape[0]}")
            return None

        positions = torch.zeros((num_bodies, 3), dtype=torch.float32, device=self._device)
        orientations = torch.zeros((num_bodies, 4), dtype=torch.float32, device=self._device)
        covered = torch.zeros(num_bodies, dtype=torch.bool, device=self._device)
        xform_mask = torch.zeros(num_bodies, dtype=torch.bool, device=self._device)

        artic = self._apply_view_poses(self._articulation_view, "articulation_view", positions, orientations, covered)
        rigid = self._apply_view_poses(self._rigid_body_view, "rigid_body_view", positions, orientations, covered)
        xform = self._apply_xform_poses(positions, orientations, covered, xform_mask)

        if not covered.all():
            logger.warning(f"Failed to read {(~covered).sum().item()}/{num_bodies} body poses")
            return None

        active = sum([artic > 0, rigid > 0, xform > 0])
        source = (
            "merged"
            if active > 1
            else ("articulation_view" if artic else "rigid_body_view" if rigid else "xform_view" if xform else "none")
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
            logger.warning(f"[SceneDataProvider] Warp unavailable for Newton state sync: {exc}")
            return None

    def update(self) -> None:
        """Sync PhysX transforms to Newton state for visualization."""
        if not self._needs_newton_sync or self._newton_state is None:
            return

        self._refresh_newton_model_if_needed()

        try:
            import warp as wp

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
            logger.debug(f"Failed to sync transforms to Newton: {exc}")

    def get_newton_model(self) -> Any | None:
        return self._newton_model if self._needs_newton_sync else None

    def get_newton_state(self) -> Any | None:
        return self._newton_state if self._needs_newton_sync else None

    def get_usd_stage(self) -> Any:
        return self._stage

    def get_mesh_data(self) -> dict[str, Any] | None:
        return None

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def get_transforms(self) -> dict[str, Any] | None:
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
        for source, view in (
            ("articulation_view", self._articulation_view),
            ("rigid_body_view", self._rigid_body_view),
        ):
            linear, angular = self._get_view_velocities(view)
            if linear is not None and angular is not None:
                return {"linear": linear, "angular": angular, "source": source}
        return None

    def get_contacts(self) -> dict[str, Any] | None:
        """Contacts not yet supported for OV backend."""
        return None
