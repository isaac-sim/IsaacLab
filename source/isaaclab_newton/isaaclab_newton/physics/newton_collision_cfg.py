# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton collision pipeline."""

from __future__ import annotations

from typing import Any, Literal

from isaaclab.utils.configclass import configclass


@configclass
class HydroelasticSDFCfg:
    """Configuration for SDF-based hydroelastic collision handling.

    Hydroelastic contacts generate distributed contact areas instead of point contacts,
    providing more realistic force distribution for manipulation and compliant surfaces.

    For more details, see the `Newton hydroelastic contacts guide`_.

    .. _Newton hydroelastic contacts guide: https://newton-physics.github.io/newton/latest/concepts/collisions.html#hydroelastic-contacts
    """

    reduce_contacts: bool = True
    """Whether to reduce contacts to a smaller representative set per shape pair.

    When False, all generated contacts are passed through without reduction.

    Defaults to ``True`` (same as Newton's default).
    """

    buffer_fraction: float = 1.0
    """Fraction of worst-case hydroelastic buffer allocations. Range: (0, 1].

    Lower values reduce memory usage but may cause overflows in dense scenes.
    Overflows are bounds-safe and emit warnings; increase this value when warnings appear.

    Defaults to ``1.0`` (same as Newton's default).
    """

    normal_matching: bool = True
    """Whether to rotate reduced contact normals to align with aggregate force direction.

    Only active when ``reduce_contacts`` is True.

    Defaults to ``True`` (same as Newton's default).
    """

    anchor_contact: bool = False
    """Whether to add an anchor contact at the center of pressure for each normal bin.

    The anchor contact helps preserve moment balance. Only active when ``reduce_contacts`` is True.

    Defaults to ``False`` (same as Newton's default).
    """

    moment_matching: bool = False
    """Redistribute reduced contact forces to preserve moment balance per normal bin.

    PhysX patch-friction analog; only active when ``reduce_contacts`` is True.
    Defaults to ``False`` (Newton default).
    """

    margin_contact_area: float = 0.01
    """Contact area [m^2] used for non-penetrating contacts at the margin.

    Defaults to ``0.01`` (same as Newton's default).
    """

    output_contact_surface: bool = False
    """Whether to output hydroelastic contact surface vertices for visualization.

    Defaults to ``False`` (same as Newton's default).
    """

    buffer_mult_broad: int = 1
    """Multiplier for preallocated broadphase buffer.

    Defaults to ``1`` (same as Newton's default).
    """

    buffer_mult_iso: int = 1
    """Multiplier for iso-surface extraction buffers.

    Defaults to ``1`` (same as Newton's default).
    """

    buffer_mult_contact: int = 1
    """Multiplier for face contact buffer.

    Defaults to ``1`` (same as Newton's default).
    """

    grid_size: int = 262144
    """Grid size for hydroelastic contact handling (256 * 8 * 128).

    Defaults to ``262144`` (same as Newton's default).
    """


@configclass
class NewtonCollisionPipelineCfg:
    """Configuration for Newton collision pipeline.

    Full-featured collision pipeline with GJK/MPR narrow phase and pluggable broad phase.
    When this config is set on :attr:`NewtonCfg.collision_cfg`:

    - **MJWarpSolverCfg**: Newton's collision pipeline replaces MuJoCo's internal contact solver.
    - **Other solvers** (XPBD, Featherstone, etc.): Configures the collision pipeline parameters
      (these solvers always use Newton's collision pipeline).

    Key features:

    - GJK/MPR algorithms for convex-convex collision detection
    - Multiple broad phase options: NXN (all-pairs), SAP (sweep-and-prune), EXPLICIT (precomputed pairs)
    - Mesh-mesh collision via SDF with contact reduction
    - Optional hydroelastic contact model for compliant surfaces

    For more details, see the `Newton collision pipeline guide`_ and `CollisionPipeline API`_.

    .. _Newton collision pipeline guide: https://newton-physics.github.io/newton/latest/concepts/collisions.html
    .. _CollisionPipeline API: https://newton-physics.github.io/newton/api/_generated/newton.CollisionPipeline.html
    """

    broad_phase: Literal["explicit", "nxn", "sap"] = "explicit"
    """Broad phase algorithm for collision detection.

    Options:

    - ``"explicit"``: Use precomputed shape pairs from ``model.shape_contact_pairs``.
    - ``"nxn"``: All-pairs brute force. Simple but O(n^2) complexity.
    - ``"sap"``: Sweep-and-prune. Good for scenes with many dynamic objects.

    Defaults to ``"explicit"`` (same as Newton's default when ``broad_phase=None``).
    """

    reduce_contacts: bool = True
    """Whether to reduce contacts for mesh-mesh collisions.

    When True, uses shared memory contact reduction to select representative contacts.
    Improves performance and stability for meshes with many vertices.

    Defaults to ``True`` (same as Newton's default).
    """

    rigid_contact_max: int | None = None
    """Maximum number of rigid contacts to allocate.

    Resolution order:

    1. If provided, use this value.
    2. Else if ``model.rigid_contact_max > 0``, use the model value.
    3. Else estimate automatically from model shape and pair metadata.

    Defaults to ``None`` (auto-estimate, same as Newton's default).
    """

    max_triangle_pairs: int = 1_000_000
    """Maximum number of triangle pairs allocated by narrow phase for mesh and heightfield collisions.

    Increase this when scenes with large/complex meshes or heightfields report
    triangle-pair overflow warnings.

    Defaults to ``1_000_000`` (same as Newton's default).
    """

    soft_contact_max: int | None = None
    """Maximum number of soft contacts to allocate.

    If None, computed as ``shape_count * particle_count``.

    Defaults to ``None`` (auto-compute, same as Newton's default).
    """

    soft_contact_margin: float = 0.01
    """Margin [m] for soft contact generation.

    Defaults to ``0.01`` (same as Newton's default).
    """

    requires_grad: bool | None = None
    """Whether to enable gradient computation for collision.

    If ``None``, uses ``model.requires_grad``.

    Defaults to ``None`` (same as Newton's default).
    """

    sdf_hydroelastic_config: HydroelasticSDFCfg | None = None
    """Configuration for SDF-based hydroelastic collision handling.

    If ``None``, hydroelastic contacts are disabled.
    If set, enables hydroelastic contacts with the specified parameters.

    Defaults to ``None`` (hydroelastic disabled, same as Newton's default).
    """

    def to_pipeline_args(self) -> dict[str, Any]:
        """Build keyword arguments for :class:`newton.CollisionPipeline`.

        Converts this configuration into the dict expected by
        ``CollisionPipeline.__init__``, handling nested config conversion
        (e.g. :class:`HydroelasticSDFCfg` → ``HydroelasticSDF.Config``).

        Returns:
            Keyword arguments suitable for ``CollisionPipeline(model, **args)``.
        """
        from newton.geometry import HydroelasticSDF

        cfg_dict = self.to_dict()
        hydro_cfg = cfg_dict.pop("sdf_hydroelastic_config", None)
        if hydro_cfg is not None:
            cfg_dict["sdf_hydroelastic_config"] = HydroelasticSDF.Config(**hydro_cfg)
        return cfg_dict


@configclass
class SDFCfg:
    """Configuration for SDF mesh collision shapes.

    Specifies how SDF (Signed Distance Field) voxel grids are built and assigned
    to bodies or shapes in a Newton model.  Bodies and shapes are selected by
    regex patterns; the SDF resolution can be set globally or overridden
    per-pattern.

    Optional hydroelastic stiffness can be assigned to matched SDF shapes.
    Pipeline-level hydroelastic parameters (contact reduction, buffer sizes,
    etc.) are configured separately via
    :attr:`NewtonCollisionPipelineCfg.sdf_hydroelastic_config`.

    Note:
        At least one of :attr:`body_patterns` or :attr:`shape_patterns` must be
        set.  At least one of :attr:`max_resolution` or
        :attr:`target_voxel_size` must be set.
    """

    max_resolution: int | None = None
    """Maximum voxel dimension for the SDF grid.

    Must be divisible by 8. Typical values: 128, 256, 512. When both this
    and :attr:`target_voxel_size` are set, both are forwarded to Newton's
    ``mesh.build_sdf()``; Newton uses :attr:`target_voxel_size` to derive
    the actual resolution.

    Defaults to ``None``.
    """

    target_voxel_size: float | None = None
    """Target voxel size [m] for the SDF grid.

    When both this and :attr:`max_resolution` are set, both are forwarded to
    Newton and this value is used to derive the actual resolution.

    Defaults to ``None``.
    """

    narrow_band_range: tuple[float, float] = (-0.1, 0.1)
    """Narrow band distance range (inner, outer) [m].

    Defines the signed-distance extent stored in the SDF voxel grid.
    Negative values are inside the mesh, positive values outside.

    Defaults to ``(-0.1, 0.1)``.
    """

    margin: float | None = None
    """Collision margin [m] for SDF shapes.

    When ``None``, the Newton builder default is used.

    Defaults to ``None``.
    """

    body_patterns: list[str] | None = None
    """Regex patterns for body labels.

    Matched bodies receive SDF collision shapes on all their mesh geometries.
    At least one of :attr:`body_patterns` or :attr:`shape_patterns` must be set.

    Defaults to ``None``.
    """

    shape_patterns: list[str] | None = None
    """Regex patterns for shape labels.

    Matched shapes receive SDF collision geometry directly.
    At least one of :attr:`body_patterns` or :attr:`shape_patterns` must be set.

    Defaults to ``None``.
    """

    pattern_resolutions: dict[str, int] | None = None
    """Per-pattern SDF resolution overrides.

    Maps a regex string to a ``max_resolution`` value.  Patterns are evaluated
    in insertion order; the first match wins.  Unmatched bodies or shapes fall
    back to the global :attr:`max_resolution` or :attr:`target_voxel_size`.

    Defaults to ``None``.
    """

    use_visual_meshes: bool = False
    """Whether to create collision shapes from visual meshes.

    When ``True``, matched bodies that lack explicit collision geometry have SDF
    collision shapes built from their visual meshes instead.

    Defaults to ``False``.
    """

    k_hydro: float | None = None
    """Hydroelastic stiffness [Pa] assigned to matched SDF shapes.

    When ``None``, no ``HYDROELASTIC`` flag is set and hydroelastic contacts are
    disabled for these shapes.  When set, matched shapes receive the flag with
    this stiffness value.

    Defaults to ``None``.
    """

    hydroelastic_shape_patterns: list[str] | None = None
    """Regex patterns restricting which SDF shapes receive hydroelastic stiffness.

    Only relevant when :attr:`k_hydro` is set.  When ``None``, all SDF shapes
    matched by :attr:`body_patterns` or :attr:`shape_patterns` get the
    hydroelastic flag.  When set, only shapes whose labels match at least one
    pattern here receive it.

    Defaults to ``None``.
    """
