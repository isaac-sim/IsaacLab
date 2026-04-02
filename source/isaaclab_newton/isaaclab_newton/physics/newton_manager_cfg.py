# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton physics manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .newton_manager import NewtonManager


@configclass
class HydroelasticCfg:
    """Configuration for hydroelastic contact handling.

    Hydroelastic contacts use SDF overlap between shape pairs to compute distributed
    contact surfaces with area-weighted forces via marching cubes. This requires SDF
    to be enabled on the parent :class:`SDFCfg`.

    Both shapes in a colliding pair must have the hydroelastic flag for hydroelastic
    contacts to be generated. Shapes that only have SDF (but not hydroelastic) will
    fall back to standard point contacts.

    .. note::
        Hydroelastic contacts require the unified collision pipeline, which is used
        when ``use_mujoco_contacts=False`` on :class:`MJWarpSolverCfg`, or when using
        a non-MuJoCo solver (XPBD, Featherstone).
    """

    k_hydro: float = 1e10
    """Hydroelastic stiffness coefficient [Pa] applied to shapes.

    Controls the compliance of the hydroelastic contact surface. Higher values produce
    stiffer contacts.
    """

    shape_patterns: list[str] | None = None
    """Regex patterns to select which shapes get hydroelastic contacts.

    If None, all shapes that have SDF enabled will also get hydroelastic contacts.
    If provided, only shapes whose label (USD prim path) matches at least one pattern
    will have the ``HYDROELASTIC`` flag set.

    Example: ``[".*Gear.*", ".*gear.*"]``
    """

    reduce_contacts: bool = True
    """Whether to reduce contacts to a smaller representative set per shape pair."""

    output_contact_surface: bool = False
    """Whether to output hydroelastic contact surface vertices for visualization."""

    normal_matching: bool = True
    """Whether to adjust reduced contact normals so their net force direction matches
    the unreduced reference. Only active when :attr:`reduce_contacts` is True."""

    moment_matching: bool = False
    """Whether to adjust reduced contact friction coefficients so their net maximum
    moment matches the unreduced reference. Only active when :attr:`reduce_contacts`
    is True."""

    margin_contact_area: float = 1e-2
    """Contact area [m^2] used for non-penetrating contacts at the margin."""

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
class SDFCfg:
    """Configuration for SDF (Signed Distance Field) collision on Newton meshes.

    When provided as ``sdf_cfg`` on :class:`NewtonCfg`, mesh collision shapes loaded
    from USD will have SDF-based collision enabled via Newton's ``mesh.build_sdf()``
    API at simulation start. At least one of ``max_resolution`` or ``target_voxel_size``
    must be set.

    Regex patterns in ``shape_patterns`` and ``pattern_resolutions`` allow selective
    SDF application and per-shape resolution tuning.
    """

    max_resolution: int | None = None
    """Maximum dimension [voxels] for sparse SDF grid (must be divisible by 8).

    If set, mesh collision shapes loaded from USD will have SDF-based collision enabled.
    Typical values: 128, 256, 512.
    """

    narrow_band_range: tuple[float, float] = (-0.1, 0.1)
    """Narrow band distance range (inner, outer) [m] for SDF computation."""

    target_voxel_size: float | None = None
    """Target voxel size [m] for sparse SDF grid.

    If provided, takes precedence over :attr:`max_resolution`.
    """

    margin: float | None = None
    """Collision margin [m] for SDF shapes. If None, uses the builder's default."""

    body_patterns: list[str] | None = None
    """List of regex patterns to match body labels (USD prim paths) for SDF.

    If None, no body-level filtering is applied. If provided, bodies whose label
    matches at least one pattern will have SDF applied to all their mesh shapes.
    Example: ``[".*elbow.*", ".*wrist.*"]``
    """

    shape_patterns: list[str] | None = None
    """List of regex patterns to match shape labels (USD prim paths) for SDF.

    If None, no shape-level filtering is applied. If provided, only shapes whose
    label matches at least one pattern get SDF.
    Example: ``[".*Gear.*", ".*gear.*"]``

    .. note::
        At least one of :attr:`body_patterns` or :attr:`shape_patterns` must be set
        for SDF to be applied. If both are None, no shapes will receive SDF.
    """

    use_visual_meshes: bool = False
    """Whether to create collision shapes from visual meshes for bodies that have
    no collision geometry. When False (default), only existing collision meshes
    are patched with SDF. When True, bodies that match the configured patterns
    but lack collision shapes will get a new collision shape created from their
    first visual mesh.
    """

    pattern_resolutions: dict[str, int] | None = None
    """Per-pattern SDF resolution overrides.

    Maps regex pattern to max_resolution for matching shapes. Shapes not matching any
    pattern here use the global ``max_resolution``. First matching pattern wins.
    Example: ``{".*elbow.*": 128, ".*power_supply.*": 512}``
    """

    hydroelastic_cfg: HydroelasticCfg | None = None
    """Hydroelastic contact configuration.

    If None (default), hydroelastic contacts are disabled and standard point contacts
    are used. When set, shapes matching the SDF patterns (or the hydroelastic-specific
    ``shape_patterns``) will have the ``HYDROELASTIC`` flag enabled and use distributed
    surface contacts computed via SDF overlap.
    """


@configclass
class NewtonCollisionPipelineCfg:
    """Configuration for Newton's collision pipeline.

    Full-featured collision pipeline with GJK/MPR narrow phase and pluggable broad phase.
    When set on :attr:`NewtonCfg.collision_cfg`, Newton's collision pipeline replaces the
    solver's native collision handling.

    For more details, see the `Newton Collisions Guide`_.

    .. _Newton Collisions Guide: https://newton-physics.github.io/newton/latest/concepts/collisions.html
    """

    broad_phase: str = "explicit"
    """Broad phase algorithm for collision detection.

    Options: ``"explicit"`` (precomputed pairs), ``"nxn"`` (all-pairs), ``"sap"`` (sweep-and-prune).
    """

    reduce_contacts: bool = True
    """Whether to reduce contacts for mesh-mesh collisions."""

    rigid_contact_max: int | None = None
    """Maximum number of rigid contacts to allocate. If None, auto-estimated from model."""

    max_triangle_pairs: int = 1_000_000
    """Maximum triangle pairs allocated by narrow phase for mesh/heightfield collisions."""

    soft_contact_max: int | None = None
    """Maximum number of soft contacts. If None, computed as ``shape_count * particle_count``."""

    soft_contact_margin: float = 0.01
    """Margin [m] for soft contact generation."""

    requires_grad: bool | None = None
    """Whether to enable gradient computation. If None, uses ``model.requires_grad``."""


@configclass
class NewtonSolverCfg:
    """Configuration for Newton solver-related parameters.

    These parameters are used to configure the Newton solver. For more information, see the `Newton documentation`_.

    .. _Newton documentation: https://newton.readthedocs.io/en/latest/
    """

    solver_type: str = "None"
    """Solver type.

    Used to select the right solver class.
    """


@configclass
class MJWarpSolverCfg(NewtonSolverCfg):
    """Configuration for MuJoCo Warp solver-related parameters.

    These parameters are used to configure the MuJoCo Warp solver. For more information, see the
    `MuJoCo Warp documentation`_.

    .. _MuJoCo Warp documentation: https://github.com/google-deepmind/mujoco_warp
    """

    solver_type: str = "mujoco_warp"
    """Solver type. Can be "mujoco_warp"."""

    njmax: int = 300
    """Number of constraints per environment (world)."""

    nconmax: int | None = None
    """Number of contact points per environment (world)."""

    iterations: int = 100
    """Number of solver iterations."""

    ls_iterations: int = 50
    """Number of line search iterations for the solver."""

    solver: str = "newton"
    """Solver type. Can be "cg" or "newton", or their corresponding MuJoCo integer constants."""

    integrator: str = "euler"
    """Integrator type. Can be "euler", "rk4", or "implicitfast", or their corresponding MuJoCo integer constants."""

    use_mujoco_cpu: bool = False
    """Whether to use the pure MuJoCo backend instead of `mujoco_warp`."""

    disable_contacts: bool = False
    """Whether to disable contact computation in MuJoCo."""

    default_actuator_gear: float | None = None
    """Default gear ratio for all actuators."""

    actuator_gears: dict[str, float] | None = None
    """Dictionary mapping joint names to specific gear ratios, overriding the `default_actuator_gear`."""

    update_data_interval: int = 1
    """Frequency (in simulation steps) at which to update the MuJoCo Data object from the Newton state.

    If 0, Data is never updated after initialization.
    """

    save_to_mjcf: str | None = None
    """Optional path to save the generated MJCF model file.

    If None, the MJCF model is not saved.
    """

    impratio: float = 1.0
    """Frictional-to-normal constraint impedance ratio."""

    cone: str = "pyramidal"
    """The type of contact friction cone. Can be "pyramidal" or "elliptic"."""

    ccd_iterations: int = 35
    """Maximum iterations for convex collision detection (GJK/EPA).

    Increase this if you see warnings about ``opt.ccd_iterations`` needing to be increased,
    which typically occurs with complex collision geometries (e.g. multi-finger hands).
    """

    ls_parallel: bool = False
    """Whether to use parallel line search."""

    use_mujoco_contacts: bool = True
    """Whether to use MuJoCo's contact solver."""


@configclass
class XPBDSolverCfg(NewtonSolverCfg):
    """An implicit integrator using eXtended Position-Based Dynamics (XPBD) for rigid and soft body simulation.

    References:
        - Miles Macklin, Matthias Müller, and Nuttapong Chentanez. 2016. XPBD: position-based simulation of compliant
          constrained dynamics. In Proceedings of the 9th International Conference on Motion in Games (MIG '16).
          Association for Computing Machinery, New York, NY, USA, 49-54. https://doi.org/10.1145/2994258.2994272
        - Matthias Müller, Miles Macklin, Nuttapong Chentanez, Stefan Jeschke, and Tae-Yong Kim. 2020. Detailed rigid
          body simulation with extended position based dynamics. In Proceedings of the ACM SIGGRAPH/Eurographics
          Symposium on Computer Animation (SCA '20). Eurographics Association, Goslar, DEU,
          Article 10, 1-12. https://doi.org/10.1111/cgf.14105

    """

    solver_type: str = "xpbd"
    """Solver type. Can be "xpbd"."""

    iterations: int = 2
    """Number of solver iterations."""

    soft_body_relaxation: float = 0.9
    """Relaxation parameter for soft body simulation."""

    soft_contact_relaxation: float = 0.9
    """Relaxation parameter for soft contact simulation."""

    joint_linear_relaxation: float = 0.7
    """Relaxation parameter for joint linear simulation."""

    joint_angular_relaxation: float = 0.4
    """Relaxation parameter for joint angular simulation."""

    joint_linear_compliance: float = 0.0
    """Compliance parameter for joint linear simulation."""

    joint_angular_compliance: float = 0.0
    """Compliance parameter for joint angular simulation."""

    rigid_contact_relaxation: float = 0.8
    """Relaxation parameter for rigid contact simulation."""

    rigid_contact_con_weighting: bool = True
    """Whether to use contact constraint weighting for rigid contact simulation."""

    angular_damping: float = 0.0
    """Angular damping parameter for rigid contact simulation."""

    enable_restitution: bool = False
    """Whether to enable restitution for rigid contact simulation."""


@configclass
class FeatherstoneSolverCfg(NewtonSolverCfg):
    """A semi-implicit integrator using symplectic Euler.

    It operates on reduced (also called generalized) coordinates to simulate articulated rigid body dynamics
    based on Featherstone's composite rigid body algorithm (CRBA).

    See: Featherstone, Roy. Rigid Body Dynamics Algorithms. Springer US, 2014.

    Semi-implicit time integration is a variational integrator that
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method
    """

    solver_type: str = "featherstone"
    """Solver type. Can be "featherstone"."""

    angular_damping: float = 0.05
    """Angular damping parameter for rigid contact simulation."""

    update_mass_matrix_interval: int = 1
    """Frequency (in simulation steps) at which to update the mass matrix."""

    friction_smoothing: float = 1.0
    """Friction smoothing parameter."""

    use_tile_gemm: bool = False
    """Whether to use tile-based GEMM for the mass matrix."""

    fuse_cholesky: bool = True
    """Whether to fuse the Cholesky decomposition."""


@configclass
class NewtonCfg(PhysicsCfg):
    """Configuration for Newton physics manager.

    This configuration includes Newton-specific simulation settings and solver configuration.
    """

    class_type: type[NewtonManager] | str = "{DIR}.newton_manager:NewtonManager"
    """The class type of the NewtonManager."""

    num_substeps: int = 1
    """Number of substeps to use for the solver."""

    debug_mode: bool = False
    """Whether to enable debug mode for the solver."""

    use_cuda_graph: bool = True
    """Whether to use CUDA graphing when simulating.

    If set to False, the simulation performance will be severely degraded.
    """

    solver_cfg: NewtonSolverCfg = MJWarpSolverCfg()
    """Solver configuration. Default is MJWarpSolverCfg()."""

    collision_cfg: NewtonCollisionPipelineCfg | None = None
    """Newton collision pipeline configuration.

    If None (default), the solver's native collision is used (e.g., MuJoCo's internal
    contact solver). If set, Newton's :class:`CollisionPipeline` is used with the
    specified parameters.
    """

    sdf_cfg: SDFCfg | None = None
    """SDF collision configuration. If None (default), SDF is disabled."""
