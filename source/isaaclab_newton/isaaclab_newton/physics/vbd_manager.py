# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VBD Newton manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from newton import Model
from newton.solvers import SolverVBD

from .newton_manager import NewtonManager
from .vbd_manager_cfg import VBDSolverCfg

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim.simulation_context import SimulationContext
    from isaaclab.sim.usd_export import UsdWriter


class NewtonVBDManager(NewtonManager):
    """Newton manager specialization for the VBD solver."""

    @classmethod
    def author_fixed_configuration(cls, writer: UsdWriter, scene: InteractiveScene) -> None:
        """Preserve effective VBD constructor settings alongside deformable schemas."""
        import json

        super().author_fixed_configuration(writer, scene)
        if not writer.include_solver_settings:
            return
        options = cls.export_solver_options(cls._solver, scene.sim.cfg.physics.solver_cfg)
        from newton.usd import PrimType, SchemaResolverNewton

        from isaaclab.assets.physics_properties import UsdAttribute

        target = SchemaResolverNewton.mapping[PrimType.SCENE]["max_solver_iterations"].name
        writer.write_attribute(
            scene.physics_scene_path, UsdAttribute(target, type_name="int"), int(options.pop("iterations"))
        )
        writer.stage.GetRootLayer().customLayerData = {
            **writer.stage.GetRootLayer().customLayerData,
            "isaaclab:newtonDriver": {"solver": "vbd", "options": json.dumps(options)},
        }

    @staticmethod
    def export_solver_options(solver: SolverVBD, cfg: VBDSolverCfg | None = None) -> dict:
        """Read effective VBD settings from the initialized solver.

        Args:
            solver: The initialized native solver.
            cfg: Constructor provenance for settings without native getters.

        Returns:
            Serializable constructor options using resolved native settings.
        """
        import inspect

        aliases = {
            "particle_enable_tile_solve": "use_particle_tile_solve",
            "particle_edge_parallel_epsilon": "_self_contact_edge_edge_parallel_epsilon",
            "rigid_avbd_joint_alpha": "rigid_joint_alpha",
            "rigid_avbd_contact_alpha": "rigid_contact_alpha",
            "rigid_avbd_linear_beta": "rigid_linear_beta",
            "rigid_avbd_angular_beta": "rigid_angular_beta",
            "rigid_contact_k_start": "rigid_contact_k_start_value",
            "rigid_body_contact_buffer_size": "body_body_contact_buffer_pre_alloc",
            "rigid_body_particle_contact_buffer_size": "body_particle_contact_buffer_pre_alloc",
        }
        superseded = {
            "particle_self_contact_radius",
            "particle_collision_detection_interval",
            "rigid_avbd_alpha",
            "rigid_avbd_beta",
            "rigid_contact_stick_motion_eps",
            "rigid_contact_stick_freeze_translation_eps",
            "rigid_contact_stick_freeze_angular_eps",
        }
        options = {}
        for name, parameter in inspect.signature(SolverVBD).parameters.items():
            if name in {"model", "deterministic"} or name in superseded:
                continue
            # Native aliases hold resolved settings; initialization-only capacities use cfg provenance.
            value = getattr(solver, aliases.get(name, name), getattr(cfg, name, parameter.default))
            if name == "rigid_contact_k_start" and value < 0:
                continue  # Negative native sentinel disables a seed when the ramp is inactive.
            if value is None:
                continue
            if name in {"collision_frequency", "collision_frequency_type"}:
                options[name] = {str(int(slot)): int(value) for slot, value in value.items()}
                continue
            if not isinstance(value, (bool, int, float, str)):
                raise NotImplementedError(f"No VBD export representation for {name}: {value!r}.")
            options[name] = value
        return options

    @staticmethod
    def load_exported_solver(model: Model, options: dict) -> SolverVBD:
        """Restore typed collision scheduling before constructing VBD."""
        from newton.solvers import SolverBase

        options = dict(options)
        for name in ("collision_frequency", "collision_frequency_type"):
            if name in options:
                options[name] = {SolverBase.CollisionSlot(int(slot)): value for slot, value in options[name].items()}
        return SolverVBD(model, **options)

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
            cls._builder.color(balance_colors=False)
        super().start_simulation()
        try:
            from isaaclab_contrib.deformable.deformable_object import setup_registered_deformable_fabric_sync
        except ModuleNotFoundError as exc:
            if exc.name not in {"isaaclab_contrib", "isaaclab_contrib.deformable"}:
                raise
        else:
            setup_registered_deformable_fabric_sync(cls)

    @classmethod
    def instantiate_builder_from_stage(cls) -> None:
        """Create and color the VBD builder from the USD stage."""
        super().instantiate_builder_from_stage()
        if cls._builder is None:
            raise RuntimeError("Newton stage import did not create a builder.")
        # Warp's optional balancing pass can cycle indefinitely for valid graph colorings.
        # The initial assignment is sufficient for VBD correctness.
        cls._builder.color(balance_colors=False)

    @classmethod
    def _get_usd_import_ignore_paths(cls) -> list[str]:
        """Return registered deformable mesh paths excluded from USD import."""
        return [
            path for entry in cls._deformable_registry for path in (entry.sim_mesh_prim_path, entry.vis_mesh_prim_path)
        ]

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
