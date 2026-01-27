#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OV (Omniverse) scene data provider for Omni PhysX backend."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


class OVSceneDataProvider:
    """Scene data provider for Omni PhysX physics backend.
    
    Native (cheap): USD stage, PhysX transforms/velocities
    Adapted (expensive): Newton Model/State (built from USD, synced each step)
    
    Performance: Only builds Newton data if Newton/Rerun visualizers are active.
    """

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
        self._set_body_q_kernel = None
        self._up_axis = UsdGeom.GetStageUpAxis(self._stage)

        # Only build Newton model if Newton/Rerun visualizers need it
        if self._has_newton_visualizer or self._has_rerun_visualizer:
            self._build_newton_model_from_usd()
            self._setup_rigid_body_view()
            self._setup_articulation_view()
        elif self._has_ov_visualizer:
            logger.info("[OVSceneDataProvider] OV visualizer only - skipping Newton model build")

    def _get_num_envs(self) -> int:
        # TODO(mtrepte): is there a better way to get num_envs?
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
        # TODO(mtrepte): add support for fabric cloning
        try:
            from newton import ModelBuilder
            from isaaclab.sim.utils import find_matching_prim_paths

            num_envs = self._get_num_envs()

            # import ipdb; ipdb.set_trace()
            env_prim_paths = find_matching_prim_paths("/World/envs/env_.*", stage=self._stage)
            print(
                "[SceneDataProvider] Stage env prims before add_usd: "+
                f"num_envs_setting={num_envs}, env_prims={len(env_prim_paths)}"
            )

            builder = ModelBuilder(up_axis=self._up_axis)
            builder.add_usd(self._stage)

            self._newton_model = builder.finalize(device=self._device)
            self._metadata["num_envs"] = num_envs
            self._newton_model.num_envs = self._metadata.get("num_envs", 0)
            self._newton_state = self._newton_model.state()
            self._rigid_body_paths = list(self._newton_model.body_key)
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

    def _setup_rigid_body_view(self) -> None:
        if not self._rigid_body_paths:
            return
        try:
            paths_to_use = self._wildcard_env_paths(list(self._rigid_body_paths))
            self._rigid_body_view = self._physics_sim_view.create_rigid_body_view(paths_to_use)
            self._cache_view_index_map(self._rigid_body_view, "rigid_body_view")
        except Exception as exc:
            logger.warning(f"[SceneDataProvider] Failed to create RigidBodyView: {exc}")
            self._rigid_body_view = None

    def _setup_articulation_view(self) -> None:
        try:
            from pxr import UsdPhysics

            from isaaclab.sim.utils import get_all_matching_child_prims

            root_prims = get_all_matching_child_prims(
                "/World",
                predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
                stage=self._stage,
                traverse_instance_prims=True,
            )
            if not root_prims:
                return

            paths_to_use = self._wildcard_env_paths([prim.GetPath().pathString for prim in root_prims])
            exprs = [path.replace(".*", "*") for path in paths_to_use]
            self._articulation_view = self._physics_sim_view.create_articulation_view(
                exprs if len(exprs) > 1 else exprs[0]
            )
            self._cache_view_index_map(self._articulation_view, "articulation_view")
        except Exception as exc:
            logger.warning(f"[SceneDataProvider] Failed to create ArticulationView: {exc}")
            self._articulation_view = None

    def _get_view_world_poses(self, view):
        # TODO(mtrepte): this can be revisited & simplifiedafter the function naming gets unified
        if view is None:
            return None, None

        method_names = (
            "get_world_poses",
            "get_world_transforms",
            "get_transforms",
            "get_poses",
        )

        for name in method_names:
            method = getattr(view, name, None)
            if method is None:
                continue
            try:
                result = method()
            except Exception:
                continue

            if isinstance(result, tuple) and len(result) == 2:
                return result

            try:
                if hasattr(result, "shape") and result.shape[-1] == 7:
                    positions = result[..., :3]
                    orientations = result[..., 3:7]
                    return positions, orientations
            except Exception:
                continue

        return None, None

    def _cache_view_index_map(self, view, key: str) -> None:
        prim_paths = getattr(view, "prim_paths", None)
        if not prim_paths or not self._rigid_body_paths:
            return

        def split_env(path: str) -> tuple[int | None, str]:
            match = re.search(r"/World/envs/env_(\d+)(/.*)", path)
            return (int(match.group(1)), match.group(2)) if match else (None, path)

        view_map: dict[tuple[int | None, str], int] = {}
        for view_idx, path in enumerate(prim_paths):
            env_id, rel = split_env(path)
            view_map[(env_id, rel)] = view_idx

        order: list[int | None] = [None] * len(self._rigid_body_paths)
        for body_idx, path in enumerate(self._rigid_body_paths):
            env_id, rel = split_env(path)
            view_idx = view_map.get((env_id, rel))
            if view_idx is None:
                view_idx = view_map.get((None, rel))
            order[body_idx] = view_idx

        if all(idx is not None for idx in order):
            self._view_body_index_map[key] = order  # type: ignore[arg-type]

    def _get_view_velocities(self, view):
        if view is None:
            return None, None

        # Preferred: combined velocities
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

        # Fallback: split linear/angular
        get_linear = getattr(view, "get_linear_velocities", None)
        get_angular = getattr(view, "get_angular_velocities", None)
        if get_linear is not None and get_angular is not None:
            try:
                return get_linear(), get_angular()
            except Exception:
                return None, None

        return None, None


    def _get_xform_world_poses(self):
        if not self._rigid_body_paths:
            return None, None
        try:
            import torch

            from isaaclab.sim.views import XformPrimView

            positions = []
            orientations = []
            for path in self._rigid_body_paths:
                view = self._xform_views.get(path)
                if view is None:
                    view = XformPrimView(path, device=self._device, stage=self._stage, validate_xform_ops=False)
                    self._xform_views[path] = view
                pos, quat = view.get_world_poses()
                positions.append(pos)
                orientations.append(quat)
            return (torch.cat(positions, dim=0), torch.cat(orientations, dim=0)) if positions else (None, None)
        except Exception as exc:
            logger.debug(f"[SceneDataProvider] Failed to read XformPrimView poses: {exc}")
            return None, None

    def _get_set_body_q_kernel(self):
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
        """Sync PhysX transforms to Newton state if Newton/Rerun visualizers are active."""
        if not (self._has_newton_visualizer or self._has_rerun_visualizer):
            return  # OV visualizer only - USD auto-synced by omni.physics
        self._refresh_newton_model_if_needed()
        if self._newton_state is None:
            return
        if self._rigid_body_view is None and self._articulation_view is None and not self._rigid_body_paths:
            return

        try:
            import torch
            import warp as wp

            from isaaclab.utils.math import convert_quat

            expected_count = self._newton_state.body_q.shape[0]
            pose_sources = (
                ("articulation_view", lambda: self._get_view_world_poses(self._articulation_view)),
                ("rigid_body_view", lambda: self._get_view_world_poses(self._rigid_body_view)),
                ("xform_view", self._get_xform_world_poses),
            )
            positions = orientations = None
            source_view = "none"
            for name, getter in pose_sources:
                positions, orientations = getter()
                if positions is not None and orientations is not None:
                    if positions.reshape(-1, 3).shape[0] == expected_count:
                        source_view = name
                        break
            if positions is None or orientations is None:
                return
            order = self._view_body_index_map.get(source_view)
            if order:
                positions = positions[order]
                orientations = orientations[order]

            positions = positions.reshape(-1, 3).to(dtype=torch.float32, device=self._device)
            orientations = orientations.reshape(-1, 4).to(dtype=torch.float32, device=self._device)
            # NOTE: PhysX tensor views return quats in xyzw, while XformPrimView returns wxyz.
            # Convert only when needed to avoid scrambling orientations.
            if source_view == "xform_view":
                orientations_xyzw = convert_quat(orientations, to="xyzw")
            else:
                orientations_xyzw = orientations

            positions_wp = wp.from_torch(positions, dtype=wp.vec3)
            orientations_wp = wp.from_torch(orientations_xyzw, dtype=wp.quatf)

            set_body_q = self._get_set_body_q_kernel()
            if set_body_q is None:
                return

            if positions_wp.shape[0] != expected_count:
                logger.debug(
                    "[SceneDataProvider] Body count mismatch for Newton sync "
                    f"(poses={positions_wp.shape[0]}, state={expected_count}, source={source_view})."
                )
                return

            wp.launch(
                set_body_q,
                dim=positions_wp.shape[0],
                inputs=[positions_wp, orientations_wp, self._newton_state.body_q],
                device=self._device,
            )

            # Future extensions:
            # - Populate velocities into self._newton_state.body_qd
            # - Cache mesh/material data for Rerun/renderer integrations
        except Exception as exc:
            logger.debug(f"[SceneDataProvider] Failed to sync Omni transforms to Newton state: {exc}")

    def get_newton_model(self) -> Any | None:
        """ADAPTED: Newton Model built from USD (only if Newton/Rerun visualizers active)."""
        if not (self._has_newton_visualizer or self._has_rerun_visualizer):
            return None
        return self._newton_model

    def get_newton_state(self) -> Any | None:
        """ADAPTED: Newton State synced from PhysX (only if Newton/Rerun visualizers active)."""
        if not (self._has_newton_visualizer or self._has_rerun_visualizer):
            return None
        return self._newton_state

    def get_usd_stage(self) -> Any:
        """NATIVE: USD stage."""
        return self._stage

    def get_mesh_data(self) -> dict[str, Any] | None:
        """NATIVE: Extract mesh data from USD stage (future work)."""
        return None

    def get_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def get_transforms(self) -> dict[str, Any] | None:
        if self._rigid_body_view is None and self._articulation_view is None:
            return None
        try:
            for getter in (
                lambda: self._get_view_world_poses(self._articulation_view),
                lambda: self._get_view_world_poses(self._rigid_body_view),
                self._get_xform_world_poses,
            ):
                positions, orientations = getter()
                if positions is not None and orientations is not None:
                    return {"positions": positions, "orientations": orientations}
            return None
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
