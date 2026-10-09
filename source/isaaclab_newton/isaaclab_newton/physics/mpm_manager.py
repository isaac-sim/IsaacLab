# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implicit MPM Newton manager."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import warp as wp
from newton import (
    BodyFlags,
    Contacts,
    Control,
    GeoType,
    Model,
    ModelBuilder,
    ModelFlags,
    State,
    StateFlags,
)
from newton.solvers import SolverImplicitMPM
from warp.fem import TemporaryStore

from isaaclab.physics import PhysicsManager
from isaaclab.utils.warp import ProxyArray

from .mpm_manager_cfg import MPMSolverCfg
from .newton_manager import NewtonManager

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.sim import SimulationContext


def _canonical_collider_velocity_mode(solver_cfg: MPMSolverCfg) -> str:
    """Resolve deprecated collider-velocity aliases used by older configurations."""
    collider_velocity_mode = solver_cfg.collider_velocity_mode
    deprecated_velocity_modes = {
        "instantaneous": "forward",
        "finite_difference": "backward",
    }
    if replacement := deprecated_velocity_modes.get(collider_velocity_mode):
        warnings.warn(
            f"collider_velocity_mode={collider_velocity_mode!r} is deprecated; use {replacement!r} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        collider_velocity_mode = replacement
    return collider_velocity_mode


def _make_solver_config(solver_cfg: MPMSolverCfg, scene_prim: Usd.Prim | None = None) -> SolverImplicitMPM.Config:
    """Build Newton's implicit MPM config, consuming authored USD when available."""
    collider_velocity_mode = _canonical_collider_velocity_mode(solver_cfg)
    values = {
        "max_iterations": solver_cfg.max_iterations,
        "tolerance": solver_cfg.tolerance,
        "solver": solver_cfg.solver,
        "warmstart_mode": solver_cfg.warmstart_mode,
        "collider_velocity_mode": collider_velocity_mode,
        "voxel_size": solver_cfg.voxel_size,
        "grid_type": solver_cfg.grid_type,
        "grid_padding": solver_cfg.grid_padding,
        "max_active_cell_count": solver_cfg.max_active_cell_count,
        "max_leaf_node_count": solver_cfg.max_leaf_node_count,
        "max_lower_node_count": solver_cfg.max_lower_node_count,
        "max_upper_node_count": solver_cfg.max_upper_node_count,
        "separate_worlds": solver_cfg.separate_worlds,
        "transfer_scheme": solver_cfg.transfer_scheme,
        "integration_scheme": solver_cfg.integration_scheme,
        "critical_fraction": solver_cfg.critical_fraction,
        "air_drag": solver_cfg.air_drag,
        "collider_normal_from_sdf_gradient": solver_cfg.collider_normal_from_sdf_gradient,
        "collider_basis": solver_cfg.collider_basis,
        "strain_basis": solver_cfg.strain_basis,
        "velocity_basis": solver_cfg.velocity_basis,
    }
    if scene_prim is None:
        return SolverImplicitMPM.Config(**values)

    config = SolverImplicitMPM.Config.create_from_usd(scene_prim)
    # Retain the cfg's Python values after validating the authored schema. USD
    # float attributes otherwise quantize grid-sensitive values to float32.
    for name, value in values.items():
        setattr(config, name, value)
    return config


def _author_mpm_scene_config(scene_prim: Usd.Prim, solver_cfg: MPMSolverCfg) -> None:
    """Author schema-representable MPM solver settings on a physics scene."""
    from pxr import Sdf, Vt  # noqa: PLC0415

    # Preserve the raw API metadata when Kit's registry predates the installed codeless schemas.
    if not scene_prim.AddAppliedSchema("NewtonMPMSceneAPI"):
        raise RuntimeError(f"Failed to apply NewtonMPMSceneAPI to '{scene_prim.GetPath()}'.")

    rheology_solvers = (solver_cfg.solver,) if isinstance(solver_cfg.solver, str) else solver_cfg.solver

    attributes = {
        "newton:maxSolverIterations": solver_cfg.max_iterations,
        "newton:mpm:tolerance": solver_cfg.tolerance,
        "newton:mpm:rheologySolvers": Vt.TokenArray(rheology_solvers),
        "newton:mpm:voxelSize": solver_cfg.voxel_size,
        "newton:mpm:gridType": solver_cfg.grid_type,
        "newton:mpm:gridPadding": solver_cfg.grid_padding,
        "newton:mpm:maxActiveCellCount": solver_cfg.max_active_cell_count,
        "newton:mpm:transferScheme": solver_cfg.transfer_scheme,
        "newton:mpm:integrationScheme": solver_cfg.integration_scheme,
        "newton:mpm:criticalFraction": solver_cfg.critical_fraction,
        "newton:mpm:airDrag": solver_cfg.air_drag,
    }
    attributes.update(_basis_attributes("collider", solver_cfg.collider_basis))
    attributes.update(_basis_attributes("strain", solver_cfg.strain_basis))
    attributes.update(_basis_attributes("velocity", solver_cfg.velocity_basis))
    for name, value in attributes.items():
        if name == "newton:mpm:rheologySolvers":
            type_name = Sdf.ValueTypeNames.TokenArray
        elif isinstance(value, bool):
            type_name = Sdf.ValueTypeNames.Bool
        elif isinstance(value, int):
            type_name = Sdf.ValueTypeNames.Int
        elif isinstance(value, str):
            type_name = Sdf.ValueTypeNames.Token
        else:
            type_name = Sdf.ValueTypeNames.Float
        scene_prim.CreateAttribute(name, type_name, custom=False, variability=Sdf.VariabilityUniform).Set(value)


def _basis_attributes(prefix: str, basis: str) -> dict[str, object]:
    """Expand a compact Newton basis name into newton-usd-schemas attributes."""
    if basis.startswith("pic"):
        if prefix == "velocity":
            raise ValueError("The MPM velocity basis cannot use a particle basis.")
        basis_type, order, discontinuous = "particle", 0, False
    else:
        matched = re.fullmatch(r"([PQBS])([0-9]+)(d?)", basis)
        if matched is None:
            raise ValueError(f"Unsupported MPM {prefix} basis {basis!r}.")
        basis_prefix = matched.group(1)
        order = int(matched.group(2))
        discontinuous = order == 0 or bool(matched.group(3))
        if basis_prefix == "P" and order > 0 and not discontinuous:
            raise ValueError(f"Unsupported MPM {prefix} basis {basis!r}: positive-order P bases must be discontinuous.")
        if (prefix == "strain" and basis_prefix in "BS") or (
            prefix == "velocity" and (basis_prefix not in "QB" or not 1 <= order <= 3)
        ):
            return {}
        if basis_prefix in "BS" and not 1 <= order <= 3:
            return {}
        basis_type = {
            "P": "linear",
            "Q": "trilinear",
            "B": "bspline",
            "S": "serendipity",
        }[basis_prefix]

    namespace = f"newton:mpm:{prefix}"
    attributes: dict[str, object] = {
        f"{namespace}BasisType": basis_type,
        f"{namespace}BasisOrder": order,
    }
    if prefix != "velocity":
        attributes[f"{namespace}DiscontinuousBasis"] = discontinuous
    return attributes


class NewtonMPMManager(NewtonManager):
    """:class:`NewtonManager` specialization for Newton's implicit MPM solver.

    MPM advances particle materials in-place and treats rigid geometry as
    colliders, so it does not consume Newton's rigid-body collision pipeline
    and steps with a single :class:`State`.
    """

    _project_outside_colliders: bool = False
    """Whether :meth:`_step_solver` projects particles out of colliders each substep.

    Set from :attr:`MPMSolverCfg.project_outside_colliders` in
    :meth:`_build_solver` and read in :meth:`_step_solver`.
    """
    _implicit_mpm_solver_root: object | None = None
    _implicit_mpm_solver_cache: tuple[SolverImplicitMPM, ...] = ()

    _material_parameter_names = (
        "density",
        "young_modulus",
        "poisson_ratio",
        "viscosity",
        "friction",
        "damping",
        "yield_pressure",
        "tensile_yield_ratio",
        "yield_stress",
        "hardening",
        "dilatancy",
        "hardening_rate",
        "softening_rate",
    )

    @classmethod
    def get_particle_material_parameters(
        cls,
        particle_ids: Sequence[int] | wp.array | slice | None = None,
        parameters: Sequence[str] | None = None,
    ) -> dict[str, ProxyArray]:
        """Read independent material arrays on the simulation device.

        Args:
            particle_ids: Global Newton particle indices, as a sequence, slice, or one-dimensional
                Warp int32/int64 array on the simulation device. None selects all particles.
            parameters: Names from :class:`MPMParticleMaterialCfg`. None selects all material parameters.

        Returns:
            One ProxyArray snapshot per parameter, in the requested particle order. Use ``.warp``
            for Warp operations or ``.torch`` for Torch operations. Density [kg/m^3] is mass divided
            by represented volume, ``8 * radius**3``; zero-mass particles report zero density.

        Raises:
            ValueError: A parameter name or particle selector is invalid.
            IndexError: A particle index is outside the model.
            RuntimeError: No implicit MPM solver is initialized.
        """
        parameter_names = cls._material_parameter_names if parameters is None else parameters
        model, particle_indices = cls._material_model_and_selection(particle_ids, parameter_names)
        invalid_indices = wp.zeros(1, dtype=wp.int32, device=model.device)
        material_parameters = {}
        for parameter_name in parameter_names:
            is_density = parameter_name == "density"
            source = model.particle_mass if is_density else getattr(model.mpm, parameter_name)
            parameter_values = wp.empty(particle_indices.shape[0], dtype=wp.float32, device=model.device)
            wp.launch(
                _read_particle_material,
                dim=particle_indices.shape[0],
                inputs=[source, model.particle_radius, particle_indices, is_density],
                outputs=[parameter_values, invalid_indices],
                device=model.device,
            )
            material_parameters[parameter_name] = ProxyArray(parameter_values)
        if invalid_indices.numpy()[0]:
            raise IndexError("particle_ids contains indices outside the Newton model's particle arrays.")
        return material_parameters

    @classmethod
    def set_particle_material_parameters(
        cls,
        material_parameters: dict[str, float | wp.array],
        particle_ids: Sequence[int] | wp.array | slice | None = None,
    ) -> None:
        """Update material arrays and queue solver synchronization before the next physics step.

        Args:
            material_parameters: Parameter names mapped to scalars or Warp float32 arrays shaped
                ``(num_selected_particles,)`` on the simulation device. Density [kg/m^3] updates
                mass and inverse mass at fixed particle radius; zero-mass particles remain kinematic.
                Damping is a relaxation time [s], as in the cfg. Pass ProxyArray inputs via ``.warp``.
            particle_ids: Global Newton particle indices, as a sequence, slice, or one-dimensional
                Warp int32/int64 array on the simulation device. None selects all particles.

        This updates the shared Newton model. Coupled solvers must support particle-property
        synchronization through ``MODEL_PROPERTIES``. Particle deformation history is preserved.
        All parameters are validated before any model array is written.

        Raises:
            ValueError: A parameter name, array shape, device, dtype, or material value is invalid.
            IndexError: A particle index is outside the model.
            RuntimeError: No implicit MPM solver is initialized.
        """
        if not material_parameters:
            return
        model, particle_indices = cls._material_model_and_selection(particle_ids, material_parameters)
        material_parameters = cls._prepare_material_parameters(model, particle_indices, material_parameters)
        if particle_indices.shape[0] == 0:
            return
        for parameter_name, parameter_values in material_parameters.items():
            is_density = parameter_name == "density"
            destination = model.particle_mass if is_density else getattr(model.mpm, parameter_name)
            wp.launch(
                _write_particle_material,
                dim=particle_indices.shape[0],
                inputs=[parameter_values, particle_indices, model.particle_radius, is_density],
                outputs=[destination, model.particle_inv_mass],
                device=model.device,
            )
        NewtonManager.add_model_change(ModelFlags.MODEL_PROPERTIES)
        if NewtonManager._graph is not None:
            NewtonManager._graph = None
            NewtonManager._graph_capture_pending = True

    @classmethod
    def initialize(cls, sim_context: SimulationContext) -> None:
        """Initialize Newton and author the MPM solver configuration in USD."""
        super().initialize(sim_context)
        scene_prim = sim_context.stage.GetPrimAtPath(sim_context.cfg.physics_prim_path)
        _author_mpm_scene_config(scene_prim, sim_context.cfg.physics.solver_cfg)

    @classmethod
    def _register_builder_attributes(cls, builder: ModelBuilder) -> None:
        """Register the particle custom attributes required by :class:`SolverImplicitMPM`.

        Implicit MPM materials are configured per-particle through Newton
        custom attributes (``mpm:young_modulus``, ``mpm:viscosity``, ...).
        These must be present on the builder *before* particles are added so
        that ``add_particles(custom_attributes=...)`` succeeds and so that
        ``builder.finalize()`` allocates the matching model arrays.

        ``create_builder()`` registers these on each prototype and the shared builder.
        """
        if not builder.has_custom_attribute("mpm:young_modulus"):
            SolverImplicitMPM.register_custom_attributes(builder)

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: ModelBuilder) -> None:
        """Normalize rigid colliders before MPM solver construction.

        Newton's implicit MPM solver treats positive-mass body colliders as
        finite-mass colliders. Isaac Lab kinematic assets can import with a
        computed mass, so clear mass and inertia for kinematic bodies to match
        Newton's direct-builder MPM examples. The solver consumes mesh vertices
        and indices but only accepts the triangle-mesh geometry type, so classify
        convex meshes as meshes without changing their geometry.
        """
        kinematic_flag = int(BodyFlags.KINEMATIC)
        for body_id, flags in enumerate(builder.body_flags):
            if int(flags) & kinematic_flag:
                builder.body_mass[body_id] = 0.0
                builder.body_inv_mass[body_id] = 0.0
                builder.body_inertia[body_id] = wp.mat33()
                builder.body_inv_inertia[body_id] = wp.mat33()
        for shape_id, shape_type in enumerate(builder.shape_type):
            if shape_type == GeoType.CONVEX_MESH:
                builder.shape_type[shape_id] = GeoType.MESH

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: MPMSolverCfg) -> SolverImplicitMPM:
        """Construct the configured implicit MPM solver."""
        scene_prim = None
        sim = PhysicsManager._sim
        if sim is not None:
            scene_prim = sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
            _author_mpm_scene_config(scene_prim, solver_cfg)
        return SolverImplicitMPM(
            model,
            _make_solver_config(solver_cfg, scene_prim),
            temporary_store=TemporaryStore(),
        )

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MPMSolverCfg) -> None:
        """Construct :class:`SolverImplicitMPM` and populate the base-class slots.

        MPM steps in-place on a single :class:`State` and runs collision
        handling internally, so it neither double-buffers state nor drives
        Newton's :class:`CollisionPipeline`.

        Args:
            model: Finalized Newton model the solver should run on.
            solver_cfg: Implicit MPM solver configuration.
        """
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = False
        NewtonManager._supports_rigid_body_force_input = False
        cls._project_outside_colliders = solver_cfg.project_outside_colliders

    @classmethod
    def _supports_cuda_graph_capture(cls) -> bool:
        """Return whether the active MPM grid has capture-stable storage."""
        return cls._solver_supports_cuda_graph_capture(cls._solver)

    @staticmethod
    def _solver_supports_cuda_graph_capture(solver: SolverImplicitMPM) -> bool:
        """Return whether an implicit-MPM solver satisfies Newton's capture contract."""
        if solver.grid_type == "fixed":
            # An unbounded active partition reads its cell count back to the CPU on every step.
            return solver.max_active_cell_count > 0
        if solver.grid_type != "sparse":
            return False

        strain_rebuild_safe = solver.strain_basis.startswith("pic") or solver.strain_basis in (
            "P0",
            "P1d",
            "Q1d",
            "Q1",
        )
        collider_rebuild_safe = solver.collider_basis.startswith("pic") or solver.collider_basis in (
            "Q1",
            "S2",
            "S3",
        )
        return (
            solver.max_active_cell_count > 0
            and solver.grid_padding == 0
            and solver.velocity_basis == "Q1"
            and strain_rebuild_safe
            and collider_rebuild_safe
        )

    @classmethod
    def _check_solver_status(cls) -> None:
        """Raise asynchronous sparse-grid rebuild failures after graph replay."""
        if NewtonManager._graph is None:
            return
        for solver in cls._implicit_mpm_solvers():
            solver.check_sparse_grid_rebuild_status()

    @classmethod
    def _implicit_mpm_solvers(cls) -> tuple[SolverImplicitMPM, ...]:
        """Return direct or coupled implicit-MPM solvers without importing the coupler."""
        root_solver = NewtonManager._solver
        if root_solver is cls._implicit_mpm_solver_root:
            return cls._implicit_mpm_solver_cache
        if isinstance(root_solver, SolverImplicitMPM):
            solvers = (root_solver,)
        elif root_solver is None or not hasattr(root_solver, "entry_names") or not hasattr(root_solver, "solver"):
            solvers = ()
        else:
            solvers = tuple(
                entry_solver
                for name in root_solver.entry_names()
                if isinstance((entry_solver := root_solver.solver(name)), SolverImplicitMPM)
            )
        cls._implicit_mpm_solver_root = root_solver
        cls._implicit_mpm_solver_cache = solvers
        return solvers

    @classmethod
    def _step_solver(
        cls, state_0: State, state_1: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Run one implicit MPM substep, optionally projecting particles out of colliders.

        The implicit solve already resolves colliders at the grid level. When
        :attr:`MPMSolverCfg.project_outside_colliders` is set, the manager also
        runs ``project_outside`` after the step (as in Newton's MPM examples) to
        hard-project particles out of collider interiors. The flag is evaluated
        when the step is first run, so the chosen branch is baked into any
        captured CUDA graph.
        """
        cls._solver.step(state_0, state_1, control, contacts, substep_dt)
        if cls._project_outside_colliders:
            cls._solver.project_outside(state_1, state_1, substep_dt)

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
        """Preserve the existing implicit-MPM behavior for automatic asset resets.

        Shared-grid MPM configurations cannot reset solver history for only a
        subset of worlds. Tasks that use independent MPM worlds and require an
        exact history reset call :meth:`reset_solver_state`
        explicitly after authoring their complete state.

        Args:
            world_mask: Per-world reset mask, intentionally ignored.
        """

    @classmethod
    def reset_solver_state(
        cls,
        state: State | None = None,
        world_mask: wp.array(dtype=wp.bool) | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Reset MPM and coupled-solver history after task state is rewritten.

        When :paramref:`state` is omitted, both distinct manager state buffers
        are reset so a later buffer swap cannot restore stale history. A mask
        follows Newton's canonical ``world_count + 1`` contract, where the last
        entry selects global entities in world -1. A selected single local world
        is promoted to a full reset because a one-world MPM grid has no
        environment offsets.

        Args:
            state: State whose solver-owned history should be reset. If omitted,
                reset both manager states.
            world_mask: Canonical per-world mask, including the final global-world entry.
            flags: State components whose solver-owned history should reset.

        Raises:
            RuntimeError: If the MPM solver or a usable state is not initialized.
            ValueError: If :paramref:`world_mask` does not use Newton's canonical shape.
        """
        solver = NewtonManager._solver
        if solver is None or NewtonManager.backend is None or not cls._implicit_mpm_solvers():
            raise RuntimeError("An implicit MPM solver is not initialized; cannot reset solver state.")
        model = NewtonManager.backend.model

        reset_mask = world_mask
        if world_mask is not None:
            expected_shape = (model.world_count + 1,)
            if world_mask.shape != expected_shape:
                raise ValueError(f"world_mask must have shape {expected_shape}; got {world_mask.shape}.")
            if model.world_count == 1:
                selected = world_mask.numpy()
                if not selected.any():
                    return
                if selected[0] and not selected[-1]:
                    reset_mask = None

        candidates = (state,) if state is not None else (NewtonManager.backend.state_1, NewtonManager.backend.state_0)
        states: list[State] = []
        seen: set[int] = set()
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            states.append(candidate)
        if not states:
            raise RuntimeError("Newton state is not initialized; provide an explicit state to reset.")

        for candidate in states:
            solver.reset(candidate, world_mask=reset_mask, flags=flags)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        """Reset MPM-specific class state on teardown.

        :meth:`_build_solver` sets :attr:`_project_outside_colliders` from the
        active config. Resetting it here keeps a teardown-only :meth:`clear`
        (without a follow-up rebuild) from leaving a stale value on the class,
        mirroring how :meth:`NewtonManager.clear` resets the base-class flags.
        """
        cls._project_outside_colliders = False
        cls._implicit_mpm_solver_root = None
        cls._implicit_mpm_solver_cache = ()

    @classmethod
    def _material_model_and_selection(
        cls, particle_ids: Sequence[int] | wp.array | slice | None, parameter_names: Iterable[str]
    ) -> tuple[Model, wp.array]:
        """Resolve the shared Newton model and particle indices for both material accessors."""
        if NewtonManager.backend is None or not cls._implicit_mpm_solvers():
            raise RuntimeError("An implicit MPM solver must be initialized before accessing particle materials.")
        unknown_parameters = set(parameter_names) - set(cls._material_parameter_names)
        if unknown_parameters:
            raise ValueError(f"Unknown MPM material parameters: {sorted(unknown_parameters)}")
        model = NewtonManager.get_model()
        if particle_ids is None:
            particle_ids = slice(None)
        if isinstance(particle_ids, slice):
            if particle_ids.step is not None and particle_ids.step <= 0:
                raise ValueError("Particle slices must have a positive step.")
            particle_ids = range(model.particle_count)[particle_ids]
        if not isinstance(particle_ids, wp.array):
            particle_ids = wp.array(particle_ids, dtype=wp.int32, device=model.device)
        if (
            particle_ids.ndim != 1
            or particle_ids.dtype not in (wp.int32, wp.int64)
            or particle_ids.device != model.device
        ):
            raise ValueError("particle_ids must be a one-dimensional Warp int32/int64 array on the simulation device.")
        return model, particle_ids

    @classmethod
    def _prepare_material_parameters(
        cls, model: Model, particle_indices: wp.array, material_parameters: dict[str, float | wp.array]
    ) -> dict[str, wp.array]:
        """Normalize inputs and validate the whole batch before writing, with one device readback."""
        prepared_parameters = {}
        invalid_parameters = wp.zeros(len(material_parameters), dtype=wp.int32, device=model.device)
        for parameter_index, (parameter_name, parameter_values) in enumerate(material_parameters.items()):
            if isinstance(parameter_values, (float, int)):
                parameter_values = wp.full(1, float(parameter_values), dtype=wp.float32, device=model.device)
            elif (
                not isinstance(parameter_values, wp.array)
                or parameter_values.dtype != wp.float32
                or parameter_values.device != model.device
                or parameter_values.shape != particle_indices.shape
            ):
                raise ValueError(
                    f"{parameter_name} must be a scalar or Warp float32 array with shape"
                    f" {particle_indices.shape} on {model.device}."
                )
            lower_bound = -1.0 if parameter_name == "poisson_ratio" else 0.0
            upper_bound = {"poisson_ratio": 0.5, "tensile_yield_ratio": 1.0}.get(parameter_name, float("inf"))
            wp.launch(
                _validate_particle_material,
                dim=particle_indices.shape[0],
                inputs=[
                    parameter_values,
                    particle_indices,
                    model.particle_radius,
                    lower_bound,
                    upper_bound,
                    parameter_name in ("density", "young_modulus", "poisson_ratio"),
                    parameter_name == "poisson_ratio",
                    parameter_name == "density",
                    parameter_index,
                ],
                outputs=[invalid_parameters],
                device=model.device,
            )
            prepared_parameters[parameter_name] = parameter_values
        for parameter_name, error in zip(prepared_parameters, invalid_parameters.numpy(), strict=True):
            if error == 2:
                raise IndexError("particle_ids contains indices outside the Newton model's particle arrays.")
            if error:
                raise ValueError(f"Invalid MPM material value for {parameter_name}.")
        return prepared_parameters


@wp.kernel
def _read_particle_material(
    source: wp.array[float],
    particle_radius: wp.array[float],
    particle_indices: wp.array(dtype=Any),
    is_density: bool,
    material_values: wp.array[float],
    invalid_indices: wp.array[int],
):
    """Gather one material parameter in the requested particle order.

    Launch one thread per selected particle. Density is computed from particle mass
    and represented volume, ``8 * radius**3``. The source arrays are not modified.

    Args:
        source: Model-wide parameter values, or particle masses [kg] when reading density.
        particle_radius: Model-wide particle radii [m].
        particle_indices: Global particle indices in the requested output order.
        is_density: Whether to convert particle mass to density [kg/m^3].
        material_values: Output values, one per selected particle.
        invalid_indices: Zero-initialized, single-element status array. Set to 1 if any
            index is out of bounds; the corresponding output value is left unwritten.
    """
    selection_index = wp.tid()
    particle_index = particle_indices[selection_index]
    if particle_index < 0 or particle_index >= source.shape[0]:
        wp.atomic_max(invalid_indices, 0, 1)
        return
    material_value = source[particle_index]
    if is_density:
        radius = particle_radius[particle_index]
        material_value /= 8.0 * radius * radius * radius
    material_values[selection_index] = material_value


@wp.kernel
def _validate_particle_material(
    material_values: wp.array[float],
    particle_indices: wp.array(dtype=Any),
    particle_radius: wp.array[float],
    lower_bound: float,
    upper_bound: float,
    exclude_lower_bound: bool,
    exclude_upper_bound: bool,
    is_density: bool,
    parameter_index: int,
    invalid_parameters: wp.array[int],
):
    """Validate one parameter of a material-update batch without modifying the model.

    Launch one thread per selected particle. Values must be finite and within the
    specified bounds. Density must also produce a finite, positive particle mass.

    Args:
        material_values: One value to broadcast, or one value per selected particle.
        particle_indices: Global particle indices to validate.
        particle_radius: Model-wide particle radii [m], used to validate density-derived mass.
        lower_bound: Minimum allowed value, in the parameter's units.
        upper_bound: Maximum allowed value, in the parameter's units.
        exclude_lower_bound: Whether equality with the lower bound is invalid.
        exclude_upper_bound: Whether equality with the upper bound is invalid.
        is_density: Whether values represent density [kg/m^3].
        parameter_index: Slot for this parameter in the batch status array.
        invalid_parameters: Zero-initialized status array, one element per parameter.
            Records 1 for invalid values or 2 for invalid particle indices, with 2 taking priority.
    """
    selection_index = wp.tid()
    particle_index = particle_indices[selection_index]
    if particle_index < 0 or particle_index >= particle_radius.shape[0]:
        wp.atomic_max(invalid_parameters, parameter_index, 2)
        return
    value_index = wp.where(material_values.shape[0] == 1, 0, selection_index)
    material_value = material_values[value_index]
    valid = wp.isfinite(material_value) and material_value >= lower_bound and material_value <= upper_bound
    if exclude_lower_bound:
        valid = valid and material_value > lower_bound
    if exclude_upper_bound:
        valid = valid and material_value < upper_bound
    if is_density:
        radius = particle_radius[particle_index]
        particle_mass = material_value * (8.0 * radius * radius * radius)
        valid = valid and wp.isfinite(particle_mass) and particle_mass > 0.0
    if not valid:
        wp.atomic_max(invalid_parameters, parameter_index, 1)


@wp.kernel
def _write_particle_material(
    material_values: wp.array[float],
    particle_indices: wp.array(dtype=Any),
    particle_radius: wp.array[float],
    is_density: bool,
    destination: wp.array[float],
    particle_inv_mass: wp.array[float],
):
    """Scatter validated material values into the shared Newton model.

    Launch one thread per selected particle after the entire batch passes validation.
    Density writes update mass and inverse mass while preserving zero-mass particles.

    Args:
        material_values: One value to broadcast, or one value per selected particle.
        particle_indices: Validated global particle indices to update.
        particle_radius: Model-wide particle radii [m].
        is_density: Whether values represent density [kg/m^3] rather than a direct parameter.
        destination: Model-wide parameter array to update, or particle masses [kg] for
            density writes. Existing mass determines whether each particle remains kinematic.
        particle_inv_mass: Model-wide inverse masses [1/kg], updated only for density writes.
            Zero-mass particles retain zero mass and receive zero inverse mass.
    """
    selection_index = wp.tid()
    particle_index = particle_indices[selection_index]
    value_index = wp.where(material_values.shape[0] == 1, 0, selection_index)
    material_value = material_values[value_index]
    if is_density:
        # Zero simulation mass identifies kinematic particles; preserve that classification.
        if destination[particle_index] > 0.0:
            radius = particle_radius[particle_index]
            particle_mass = material_value * (8.0 * radius * radius * radius)
            destination[particle_index] = particle_mass
            particle_inv_mass[particle_index] = 1.0 / particle_mass
        else:
            particle_inv_mass[particle_index] = 0.0
    else:
        destination[particle_index] = material_value
