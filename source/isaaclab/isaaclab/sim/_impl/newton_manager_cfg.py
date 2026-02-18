# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from .solvers_cfg import MJWarpSolverCfg, NewtonSolverCfg


@configclass
class HydroelasticCfg:
    """Configuration for hydroelastic contact handling.

    Hydroelastic contacts use SDF overlap between shape pairs to compute distributed
    contact surfaces with area-weighted forces via marching cubes. This requires SDF
    to be enabled on the target shapes (via ``sdf_max_resolution`` or
    ``sdf_target_voxel_size`` on :class:`NewtonCfg`).

    Both shapes in a colliding pair must have the hydroelastic flag for hydroelastic
    contacts to be generated. Shapes that only have SDF (but not hydroelastic) will
    fall back to standard point contacts.

    Note:
        Hydroelastic contacts require the unified collision pipeline
        (``CollisionPipelineUnified``), which is used when ``use_mujoco_contacts=False``
        on :class:`~isaaclab.sim._impl.solvers_cfg.MJWarpSolverCfg`, or when using
        a non-MuJoCo solver (XPBD, Featherstone).
    """

    k_hydro: float = 1e10
    """Hydroelastic stiffness coefficient applied to shapes.

    Controls the compliance of the hydroelastic contact surface. Higher values produce
    stiffer contacts. Default matches Newton's ``ShapeConfig.k_hydro``.
    """

    shape_patterns: list[str] | None = None
    """Regex patterns to select which shapes get hydroelastic contacts.

    If None, all shapes that have SDF enabled will also get hydroelastic contacts.
    If provided, only shapes whose key (USD prim path) matches at least one pattern
    will have the ``HYDROELASTIC`` flag set.

    Example: ``[".*Gear.*", ".*gear.*"]``
    """

    reduce_contacts: bool = True
    """Whether to reduce contacts to a smaller representative set per shape pair."""

    output_contact_surface: bool = False
    """Whether to output hydroelastic contact surface vertices for visualization."""

    sticky_contacts: float = 0.0
    """Stickiness factor for temporal contact persistence.

    A small positive value (e.g. ``1e-6``) can prevent jittering contacts.
    """

    normal_matching: bool = True
    """Whether to adjust reduced contact normals so their net force direction matches
    the unreduced reference. Only active when ``reduce_contacts`` is True."""

    moment_matching: bool = False
    """Whether to adjust reduced contact friction coefficients so their net maximum
    moment matches the unreduced reference. Only active when ``reduce_contacts`` is True."""

    margin_contact_area: float = 1e-2
    """Contact area used for non-penetrating contacts at the margin."""

    betas: tuple[float, float] = (10.0, -0.5)
    """Penetration beta values for contact reduction heuristics."""

    buffer_mult_broad: int = 1
    """Multiplier for the preallocated broadphase buffer. Increase if a broadphase
    overflow warning is issued."""

    buffer_mult_iso: int = 1
    """Multiplier for preallocated iso-surface extraction buffers. Increase if an
    iso buffer overflow warning is issued."""

    buffer_mult_contact: int = 1
    """Multiplier for the preallocated face contact buffer. Increase if a face
    contact overflow warning is issued."""

    grid_size: int = 256 * 8 * 128
    """Grid size for contact handling. Can be tuned for performance."""


@configclass
class NewtonCfg:
    """Configuration for Newton-related parameters.

    These parameters are used to configure the Newton physics simulation.
    """

    num_substeps: int = 1
    """Number of substeps to use for the solver."""

    debug_mode: bool = False
    """Whether to enable debug mode for the solver."""

    use_cuda_graph: bool = True
    """Whether to use CUDA graphing when simulating.

    If set to False, the simulation performance will be severely degraded.
    """

    solver_cfg: NewtonSolverCfg = MJWarpSolverCfg()

    # SDF collision settings (applied to mesh shapes after USD import)
    sdf_max_resolution: int | None = None
    """Maximum dimension for sparse SDF grid (must be divisible by 8).

    If set, mesh collision shapes loaded from USD will have SDF-based collision enabled.
    Requires a CUDA-capable GPU. Set to None (default) to disable SDF generation.
    Typical values: 128, 256, 512.
    """

    sdf_narrow_band_range: tuple[float, float] = (-0.1, 0.1)
    """The narrow band distance range (inner, outer) for SDF computation.

    Only used when sdf_max_resolution or sdf_target_voxel_size is set.
    """

    sdf_target_voxel_size: float | None = None
    """Target voxel size for sparse SDF grid.

    If provided, enables SDF generation and takes precedence over sdf_max_resolution.
    Requires a CUDA-capable GPU. Set to None (default) to use sdf_max_resolution instead.
    """

    sdf_contact_margin: float | None = None
    """Contact margin for SDF shapes. If None, uses the builder's default."""

    sdf_shape_patterns: list[str] | None = None
    """List of regex patterns to match shape keys (USD prim paths) for SDF.

    If None, SDF is applied to all mesh shapes. If provided, only matching shapes get SDF.
    Example: ``[".*Gear.*", ".*gear.*"]``
    """

    hydroelastic_cfg: HydroelasticCfg | None = None
    """Hydroelastic contact configuration.

    If None (default), hydroelastic contacts are disabled and standard point contacts
    are used. When set, shapes matching the SDF patterns (or the hydroelastic-specific
    ``shape_patterns``) will have the ``HYDROELASTIC`` flag enabled and use distributed
    surface contacts computed via SDF overlap.

    Requires SDF to be enabled (``sdf_max_resolution`` or ``sdf_target_voxel_size``).
    """
