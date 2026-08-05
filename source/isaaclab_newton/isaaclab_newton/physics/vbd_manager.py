# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import warp as wp
from newton import Model
from newton.solvers import SolverVBD
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx

from pxr import UsdGeom

from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors
from isaaclab.sim.utils.stage import get_current_stage

from isaaclab_newton.cloner.newton_clone_utils import (
    _restore_visible_colliders_without_visual_shapes,
    replicate_builder_mapping,
)

from .newton_manager import NewtonManager
from .vbd_manager_cfg import VBDSolverCfg

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext


class NewtonVBDManager(NewtonManager):
    """Newton manager specialization for the VBD solver."""

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize VBD deformable integration when contrib is available."""
        try:
            from isaaclab_contrib.deformable.deformable_object import install_deformable_builder_hooks
        except ModuleNotFoundError as exc:
            if exc.name not in {"isaaclab_contrib", "isaaclab_contrib.deformable"}:
                raise
        else:
            install_deformable_builder_hooks()
        super().initialize(sim_context)

    @classmethod
    def start_simulation(cls) -> None:
        """Start simulation and bind registered deformables to Fabric."""
        if cls._builder is not None:
            cls._builder.color()
        super().start_simulation()
        try:
            from isaaclab_contrib.deformable.deformable_object import setup_registered_deformable_fabric_sync
        except ModuleNotFoundError as exc:
            if exc.name not in {"isaaclab_contrib", "isaaclab_contrib.deformable"}:
                raise
        else:
            setup_registered_deformable_fabric_sync(cls)

    @classmethod
    def instantiate_builder_from_stage(cls):
        """Create a builder while excluding registered deformable meshes from USD import."""
        stage = get_current_stage()
        up_axis = UsdGeom.GetStageUpAxis(stage)

        env_pattern = re.compile(r"^[Ee]nv_(\d+)$")
        world_prim = stage.GetPrimAtPath("/World")
        env_paths: list[tuple[int, str]] = []
        if world_prim and world_prim.IsValid():
            for child in world_prim.GetChildren():
                match = env_pattern.match(child.GetName())
                if match:
                    env_paths.append((int(match.group(1)), child.GetPath().pathString))
        env_paths.sort(key=lambda x: x[0])

        builder = cls.create_builder(up_axis=up_axis)
        schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
        deformable_ignore_paths = [
            path for entry in cls._deformable_registry for path in (entry.sim_mesh_prim_path, entry.vis_mesh_prim_path)
        ]
        hf_ignore_paths = cls._inject_terrain_heightfields(stage, builder)

        if not env_paths:
            ignore_paths = [*hf_ignore_paths, *deformable_ignore_paths]
            import_result = builder.add_usd(stage, ignore_paths=ignore_paths, schema_resolvers=schema_resolvers)
            _restore_visible_colliders_without_visual_shapes(builder, stage, import_result["path_shape_map"])
            replace_newton_builder_shape_colors(builder, stage)
            NewtonManager._world_xforms = [wp.transform()]
            for hook in cls._per_world_builder_hooks:
                hook(builder, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        else:
            ignore_paths = [path for _, path in env_paths] + hf_ignore_paths + deformable_ignore_paths
            import_result = builder.add_usd(stage, ignore_paths=ignore_paths, schema_resolvers=schema_resolvers)
            _restore_visible_colliders_without_visual_shapes(builder, stage, import_result["path_shape_map"])
            replace_newton_builder_shape_colors(builder, stage)

            _, proto_path = env_paths[0]
            source_builders = {proto_path: cls.create_builder(up_axis=up_axis)}
            import_result = source_builders[proto_path].add_usd(
                stage,
                root_path=proto_path,
                ignore_paths=deformable_ignore_paths,
                schema_resolvers=schema_resolvers,
            )
            _restore_visible_colliders_without_visual_shapes(
                source_builders[proto_path], stage, import_result["path_shape_map"]
            )
            replace_newton_builder_shape_colors(source_builders[proto_path], stage)
            cls._cl_protos = source_builders

            global_site_indices, source_site_indices, env_root_sites = cls._cl_inject_sites(builder, source_builders)
            xform_cache = UsdGeom.XformCache()
            poses = []
            for _, env_path in env_paths:
                world_xform = xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(env_path))
                translation = world_xform.ExtractTranslation()
                rotation = world_xform.ExtractRotationQuat()
                imag = rotation.GetImaginary()
                poses.append(
                    (
                        (translation[0], translation[1], translation[2]),
                        (imag[0], imag[1], imag[2], rotation.GetReal()),
                    )
                )

            positions = torch.tensor([pos for pos, _ in poses], dtype=torch.float32)
            quaternions = torch.tensor([quat for _, quat in poses], dtype=torch.float32)
            mapping = torch.ones((1, len(env_paths)), dtype=torch.bool)
            replicate_args = (builder, (proto_path,), mapping, positions, quaternions, source_builders)
            local_site_map, world_xforms = replicate_builder_mapping(
                *replicate_args,
                source_site_indices=source_site_indices,
                env_root_sites=env_root_sites,
                per_world_builder_hooks=cls._per_world_builder_hooks,
            )

            NewtonManager._cl_site_index_map = {label: (idx, None) for label, idx in global_site_indices.items()}
            NewtonManager._cl_site_index_map.update(
                (label, (None, per_world)) for label, per_world in local_site_map.items()
            )
            NewtonManager._world_xforms = world_xforms
            NewtonManager._num_envs = len(env_paths)

        builder.color()
        cls.set_builder(builder)

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: VBDSolverCfg) -> SolverVBD:
        """Construct the configured VBD solver."""
        return SolverVBD(model, **cls._filter_solver_kwargs(SolverVBD, solver_cfg))

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: VBDSolverCfg) -> None:
        """Construct VBD and configure its base-manager state."""
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True
        NewtonManager._supports_rigid_body_force_input = not solver_cfg.integrate_with_external_rigid_solver

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Clear contrib deformable integration when available."""
        try:
            from isaaclab_contrib.deformable.deformable_object import clear_deformable_builder_hooks
        except ModuleNotFoundError as exc:
            if exc.name not in {"isaaclab_contrib", "isaaclab_contrib.deformable"}:
                raise
        else:
            clear_deformable_builder_hooks()

    @classmethod
    def _simulate_physics_only(cls) -> None:
        """Rebuild the VBD particle BVH before stepping physics."""
        if cls._model.particle_count > 0 and hasattr(cls._solver, "rebuild_bvh"):
            cls._solver.rebuild_bvh(cls._state_0)
        super()._simulate_physics_only()
