# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Runtime access to material parameters on the shared Newton particle model."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import cache
from typing import TYPE_CHECKING, Any

import warp as wp
from newton import Model, ModelFlags

from isaaclab.utils.warp import ProxyArray

from .mpm_solver import implicit_mpm_solvers

if TYPE_CHECKING:
    from newton.solvers import SolverImplicitMPM

    from .newton_backend import NewtonBackend

_MATERIAL_PARAMETER_NAMES: tuple[str, ...] = (
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


def get_particle_material_parameters(
    backend: NewtonBackend,
    particle_ids: Sequence[int] | wp.array | slice | None = None,
    parameters: Sequence[str] | None = None,
) -> dict[str, ProxyArray]:
    """Read independent material arrays on the simulation device.

    Args:
        backend: Initialized simulation backend owning the shared model and solver.
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
    parameter_names = _MATERIAL_PARAMETER_NAMES if parameters is None else parameters
    model, particle_indices, start, stride, count = _material_model_and_selection(
        backend, particle_ids, parameter_names
    )
    invalid_indices = wp.zeros(1, dtype=wp.int32, device=model.device)
    material_parameters = {}
    for parameter_name in parameter_names:
        is_density = parameter_name == "density"
        source = model.particle_mass if is_density else getattr(model.mpm, parameter_name)
        parameter_values = wp.empty(count, dtype=wp.float32, device=model.device)
        wp.launch(
            _read_particle_material(is_density, particle_indices.size > 0),
            dim=count,
            inputs=[source, model.particle_radius, particle_indices, start, stride],
            outputs=[parameter_values, invalid_indices],
            device=model.device,
        )
        material_parameters[parameter_name] = ProxyArray(parameter_values)
    if invalid_indices.numpy()[0]:
        raise IndexError("particle_ids contains indices outside the Newton model's particle arrays.")
    return material_parameters


def set_particle_material_parameters(
    backend: NewtonBackend,
    material_parameters: dict[str, float | wp.array],
    particle_ids: Sequence[int] | wp.array | slice | None = None,
) -> None:
    """Update material arrays and queue solver synchronization before the next physics step.

    Args:
        backend: Initialized simulation backend owning the shared model and solver.
        material_parameters: Parameter names mapped to scalars or Warp float32 arrays shaped
            ``(num_selected_particles,)`` on the simulation device. Density [kg/m^3] updates
            mass and inverse mass at fixed particle radius; zero-mass particles remain kinematic.
            Damping is a relaxation time [s], as in the cfg. Pass ProxyArray inputs via ``.warp``.
        particle_ids: Global Newton particle indices, as a sequence, slice, or one-dimensional
            Warp int32/int64 array on the simulation device. None selects all particles.

    This updates the shared Newton model. Coupled solvers must support particle-property
    synchronization through ``MODEL_PROPERTIES``. Particle deformation history is preserved.
    All parameters are validated before any model array is written. Repeated indices are
    rejected for writes; inputs that overlap model destinations are snapshotted first.

    Raises:
        ValueError: A parameter name, array shape, device, dtype, or material value is invalid.
        IndexError: A particle index is outside the model.
        RuntimeError: No implicit MPM solver is initialized.
    """
    if not material_parameters:
        return
    model, particle_indices, start, stride, count = _material_model_and_selection(
        backend, particle_ids, material_parameters
    )
    material_parameters = _prepare_material_parameters(
        model, particle_indices, start, stride, count, material_parameters
    )
    if count == 0:
        return
    for parameter_name, parameter_values in material_parameters.items():
        is_density = parameter_name == "density"
        destination = model.particle_mass if is_density else getattr(model.mpm, parameter_name)
        wp.launch(
            _write_particle_material(is_density, particle_indices.size > 0),
            dim=count,
            inputs=[parameter_values, particle_indices, start, stride, model.particle_radius],
            outputs=[destination, model.particle_inv_mass],
            device=model.device,
        )
    backend.model_changes.add(ModelFlags.MODEL_PROPERTIES)


def _material_model_and_selection(
    backend: NewtonBackend, particle_ids: Sequence[int] | wp.array | slice | None, parameter_names: Iterable[str]
) -> tuple[Model, wp.array, int, int, int]:
    """Resolve the shared Newton model and particle indices for both material accessors."""
    if not implicit_mpm_solvers(backend.solver):
        raise RuntimeError("An implicit MPM solver must be initialized before accessing particle materials.")
    unknown_parameters = set(parameter_names) - set(_MATERIAL_PARAMETER_NAMES)
    if unknown_parameters:
        raise ValueError(f"Unknown MPM material parameters: {sorted(unknown_parameters)}")
    model = backend.model
    if particle_ids is None:
        particle_ids = slice(None)
    if isinstance(particle_ids, slice):
        if particle_ids.step is not None and particle_ids.step <= 0:
            raise ValueError("Particle slices must have a positive step.")
        selection = range(model.particle_count)[particle_ids]
        return (
            model,
            wp.empty(0, dtype=wp.int32, device=model.device),
            selection.start,
            selection.step,
            len(selection),
        )
    if not isinstance(particle_ids, wp.array):
        particle_ids = wp.array(particle_ids, dtype=wp.int32, device=model.device)
    if particle_ids.ndim != 1 or particle_ids.dtype not in (wp.int32, wp.int64) or particle_ids.device != model.device:
        raise ValueError("particle_ids must be a one-dimensional Warp int32/int64 array on the simulation device.")
    return model, particle_ids, 0, 1, particle_ids.size


def _prepare_material_parameters(
    model: Model,
    particle_indices: wp.array,
    start: int,
    stride: int,
    count: int,
    material_parameters: dict[str, float | wp.array],
) -> dict[str, wp.array]:
    """Normalize inputs and validate the whole batch before writing, with one device readback."""
    prepared_parameters = {}
    invalid_parameters = wp.zeros(len(material_parameters) + 1, dtype=wp.int32, device=model.device)
    if particle_indices.size:
        wp.launch(
            _validate_unique_particle_indices,
            dim=count,
            inputs=[particle_indices, model.particle_count],
            outputs=[wp.zeros(model.particle_count, dtype=wp.int32, device=model.device), invalid_parameters],
            device=model.device,
        )
    destinations = [model.particle_mass, model.particle_inv_mass] + [
        getattr(model.mpm, name) for name in material_parameters if name != "density"
    ]
    for parameter_index, (parameter_name, parameter_values) in enumerate(material_parameters.items()):
        if isinstance(parameter_values, (float, int)):
            parameter_values = wp.full(1, float(parameter_values), dtype=wp.float32, device=model.device)
        elif (
            not isinstance(parameter_values, wp.array)
            or parameter_values.dtype != wp.float32
            or parameter_values.device != model.device
            or parameter_values.shape != (count,)
        ):
            raise ValueError(
                f"{parameter_name} must be a scalar or Warp float32 array with shape {(count,)} on {model.device}."
            )
        if parameter_values.size and any(
            parameter_values.ptr < dest.ptr + dest.size * 4
            and dest.ptr < parameter_values.ptr + (parameter_values.size - 1) * parameter_values.strides[0] + 4
            for dest in destinations
        ):
            parameter_values = wp.clone(parameter_values)
        lower_bound = -1.0 if parameter_name == "poisson_ratio" else 0.0
        upper_bound = {"poisson_ratio": 0.5, "tensile_yield_ratio": 1.0}.get(parameter_name, float("inf"))
        wp.launch(
            _validate_particle_material(
                parameter_name == "density",
                particle_indices.size > 0,
                parameter_name in ("young_modulus", "poisson_ratio"),
                parameter_name == "poisson_ratio",
            ),
            dim=count,
            inputs=[
                parameter_values,
                particle_indices,
                start,
                stride,
                model.particle_radius,
                model.particle_mass,
                lower_bound,
                upper_bound,
                parameter_index,
            ],
            outputs=[invalid_parameters],
            device=model.device,
        )
        prepared_parameters[parameter_name] = parameter_values
    errors = invalid_parameters.numpy()
    if errors[-1] == 2:
        raise IndexError("particle_ids contains indices outside the Newton model's particle arrays.")
    if errors[-1]:
        raise ValueError("particle_ids must not contain repeated indices for material updates.")
    for parameter_name, error in zip(prepared_parameters, errors[:-1], strict=True):
        if error == 2:
            raise IndexError("particle_ids contains indices outside the Newton model's particle arrays.")
        if error:
            raise ValueError(f"Invalid MPM material value for {parameter_name}.")
    return prepared_parameters


@cache
def _read_particle_material(is_density: bool, indexed: bool) -> wp.Kernel:
    """Compile the material operation for fixed selection and parameter modes."""

    @wp.kernel
    def kernel(
        source: wp.array[float],
        particle_radius: wp.array[float],
        particle_indices: wp.array(dtype=Any),
        start: int,
        stride: int,
        material_values: wp.array[float],
        invalid_indices: wp.array[int],
    ):
        """Gather snapshots, converting mass to density for the density specialization."""
        selection_index = wp.tid()
        particle_index = wp.int64(start) + wp.int64(stride) * wp.int64(selection_index)
        if wp.static(indexed):
            particle_index = wp.int64(particle_indices[selection_index])
        if particle_index < 0 or particle_index >= source.shape[0]:
            wp.atomic_max(invalid_indices, 0, 1)
            return
        material_value = source[particle_index]
        if wp.static(is_density):
            radius = particle_radius[particle_index]
            material_value /= 8.0 * radius * radius * radius
        material_values[selection_index] = material_value

    return kernel


@cache
def _validate_particle_material(
    is_density: bool, indexed: bool, exclude_lower_bound: bool, exclude_upper_bound: bool
) -> wp.Kernel:
    """Compile the material operation for fixed selection and parameter modes."""

    @wp.kernel
    def kernel(
        material_values: wp.array[float],
        particle_indices: wp.array(dtype=Any),
        start: int,
        stride: int,
        particle_radius: wp.array[float],
        existing_mass: wp.array[float],
        lower_bound: float,
        upper_bound: float,
        parameter_index: int,
        invalid_parameters: wp.array[int],
    ):
        """Validate a whole material batch before any model writes."""
        selection_index = wp.tid()
        particle_index = wp.int64(start) + wp.int64(stride) * wp.int64(selection_index)
        if wp.static(indexed):
            particle_index = wp.int64(particle_indices[selection_index])
        if particle_index < 0 or particle_index >= particle_radius.shape[0]:
            wp.atomic_max(invalid_parameters, parameter_index, 2)
            return
        value_index = wp.where(material_values.shape[0] == 1, 0, selection_index)
        material_value = material_values[value_index]
        valid = wp.isfinite(material_value) and material_value >= lower_bound and material_value <= upper_bound
        if wp.static(exclude_lower_bound):
            valid = valid and material_value > lower_bound
        if wp.static(exclude_upper_bound):
            valid = valid and material_value < upper_bound
        if wp.static(is_density):
            radius = particle_radius[particle_index]
            particle_mass = material_value * (8.0 * radius * radius * radius)
            if existing_mass[particle_index] > 0.0:
                valid = valid and wp.isfinite(particle_mass) and particle_mass > 0.0
        if not valid:
            wp.atomic_max(invalid_parameters, parameter_index, 1)

    return kernel


@cache
def _write_particle_material(is_density: bool, indexed: bool) -> wp.Kernel:
    """Compile the material operation for fixed selection and parameter modes."""

    @wp.kernel
    def kernel(
        material_values: wp.array[float],
        particle_indices: wp.array(dtype=Any),
        start: int,
        stride: int,
        particle_radius: wp.array[float],
        destination: wp.array[float],
        particle_inv_mass: wp.array[float],
    ):
        """Write selected values, preserving kinematic mass in the density specialization."""
        selection_index = wp.tid()
        particle_index = wp.int64(start) + wp.int64(stride) * wp.int64(selection_index)
        if wp.static(indexed):
            particle_index = wp.int64(particle_indices[selection_index])
        value_index = wp.where(material_values.shape[0] == 1, 0, selection_index)
        material_value = material_values[value_index]
        if wp.static(is_density):
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

    return kernel


@wp.kernel
def _validate_unique_particle_indices(
    indices: wp.array(dtype=Any), particle_count: int, seen: wp.array[int], errors: wp.array[int]
):
    """Reject invalid and repeated scatter destinations before material writes."""
    index = indices[wp.tid()]
    if index < 0 or index >= particle_count:
        wp.atomic_max(errors, errors.shape[0] - 1, 2)
    elif wp.atomic_add(seen, index, 1) > 0:
        wp.atomic_max(errors, errors.shape[0] - 1, 1)


def mpm_material_graph_signature(solvers: tuple[SolverImplicitMPM, ...]) -> tuple:
    """Track MPM material buffers and host-side branches recorded by CUDA capture."""
    signatures = []
    for solver in solvers:
        model = solver._mpm_model
        arrays = (
            model.particle_radius,
            model.particle_volume,
            model.particle_density,
            model.particle_flags,
            model.material_particle_flags,
            model.material_particle_volume,
            *(getattr(model.material_parameters, name) for name in _MATERIAL_PARAMETER_NAMES if name != "density"),
        )
        signatures.append(
            (
                tuple((array.ptr, array.shape) for array in arrays),
                model.has_compliant_particles,
                model.has_hardening,
                model.has_viscosity,
                model.has_dilatancy,
            )
        )
    return tuple(signatures)
