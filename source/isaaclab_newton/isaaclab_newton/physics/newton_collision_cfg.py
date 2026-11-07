# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton collision pipeline."""

from __future__ import annotations

import math
from typing import Any, Literal

from isaaclab.utils import configclass, to_dict


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

    margin_contact_area: float = 0.01
    """Contact area [m^2] used for non-penetrating contacts at the margin.

    Defaults to ``0.01`` (same as Newton's default).
    """

    output_contact_surface: bool = False
    """Whether to output hydroelastic contact surface vertices for visualization.

    Defaults to ``False`` (same as Newton's default).
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

    @configclass
    class SDFAllShapesCfg:
        """Strict SDF provisioning for every finite shape collider in the assembled model.

        The backend applies this policy after USD-authored mesh approximations and
        scene replication have completed. It preserves boxes because Newton can
        create their texture SDFs and collision edges directly, tessellates other
        analytic colliders, keeps the resulting collision geometry, and requests
        a texture SDF for every finite colliding shape before model finalization.
        Infinite planes remain analytic half-spaces; unsupported finite geometry
        raises instead of silently retaining a non-SDF collision path.
        """

        sdf_max_resolution: int | None = 64
        """Maximum SDF grid dimension [dimensionless].

        Must be positive, less than ``65536``, and divisible by 8. Ignored when
        :attr:`sdf_target_voxel_size` is set.
        """

        sdf_target_voxel_size: float | None = None
        """Target SDF voxel size [m].

        When set, this takes precedence over :attr:`sdf_max_resolution`.
        """

        sdf_narrow_band_inner: float = -0.1
        """Inner narrow-band distance for SDF generation [m]. Must be negative."""

        sdf_narrow_band_outer: float = 0.1
        """Outer narrow-band distance for SDF generation [m]. Must be positive."""

        sdf_texture_format: Literal["uint8", "uint16", "float32"] = "uint16"
        """Subgrid texture storage format for generated SDFs."""

        sdf_padding: float | None = None
        """SDF AABB padding [m]. When ``None``, each shape's contact gap is used."""

        primitive_mesh_segments: int = 32
        """Circumferential tessellation segments for curved analytic colliders."""

        plane_thickness: float = 0.1
        """Full thickness [m] of the downward slab replacing each finite collision plane."""

        def __post_init__(self) -> None:
            if self.sdf_max_resolution is None and self.sdf_target_voxel_size is None:
                raise ValueError("Set sdf_max_resolution or sdf_target_voxel_size for strict SDF provisioning.")
            if self.sdf_max_resolution is not None:
                if self.sdf_max_resolution <= 0 or self.sdf_max_resolution >= (1 << 16):
                    raise ValueError(
                        f"sdf_max_resolution must be positive and less than {1 << 16} (got {self.sdf_max_resolution})."
                    )
                if self.sdf_max_resolution % 8 != 0:
                    raise ValueError(f"sdf_max_resolution must be divisible by 8 (got {self.sdf_max_resolution}).")
            if self.sdf_target_voxel_size is not None and (
                not math.isfinite(self.sdf_target_voxel_size) or self.sdf_target_voxel_size <= 0.0
            ):
                raise ValueError(
                    f"sdf_target_voxel_size must be finite and positive (got {self.sdf_target_voxel_size})."
                )
            if not (
                math.isfinite(self.sdf_narrow_band_inner)
                and math.isfinite(self.sdf_narrow_band_outer)
                and self.sdf_narrow_band_inner < 0.0 < self.sdf_narrow_band_outer
            ):
                raise ValueError(
                    "sdf_narrow_band_inner and sdf_narrow_band_outer must be finite and satisfy inner < 0 < outer."
                )
            if self.sdf_padding is not None and (not math.isfinite(self.sdf_padding) or self.sdf_padding < 0.0):
                raise ValueError(f"sdf_padding must be finite and nonnegative (got {self.sdf_padding}).")
            if self.primitive_mesh_segments < 8:
                raise ValueError(f"primitive_mesh_segments must be at least 8 (got {self.primitive_mesh_segments}).")
            if not math.isfinite(self.plane_thickness) or self.plane_thickness <= 0.0:
                raise ValueError(f"plane_thickness must be finite and positive (got {self.plane_thickness}).")

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

    max_triangle_pairs: int | None = None
    """Maximum number of triangle pairs allocated by narrow phase for mesh and heightfield collisions.

    Also sizes the global contact-reduction buffer and hashtable, which all worlds share.
    If ``None``, Newton estimates it from the model, so it grows with the number of worlds.
    Set it explicitly to choose the memory budget, or when overflow warnings appear.

    Defaults to ``None`` (auto-estimate, same as Newton's default).
    """

    speculative_contact_gap_max: float | None = None
    """Maximum velocity-derived rigid-contact detection gap [m].

    ``None`` disables speculative contacts, matching Newton's default.
    """

    sdf_all_shapes: SDFAllShapesCfg | None = None
    """Optional strict SDF provisioning for every finite colliding shape."""

    soft_contact_max: int | None = None
    """Maximum number of soft contacts to allocate.

    If None, computed as ``shape_count * particle_count``.

    Defaults to ``None`` (auto-compute, same as Newton's default).
    """

    soft_contact_margin: float = 0.01
    """Margin [m] for soft contact generation.

    Defaults to ``0.01`` (same as Newton's default).
    """

    enable_rigid_soft_full_surface_contact: bool = False
    """Whether to generate soft contacts against full-surface-capable rigid colliders.

    When ``True``, Newton adds edge and triangle-interior soft contacts (in addition to the
    per-vertex particle contacts) so rigid features that pass between soft vertices are caught.
    Analytic shapes (boxes, capsules, spheres) are full-surface-capable without an SDF; any
    participating mesh/convex collider must carry a volume SDF.

    Defaults to ``False`` (same as Newton's default).
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

        cfg_dict = to_dict(self)
        cfg_dict.pop("sdf_all_shapes", None)
        hydro_cfg = cfg_dict.pop("sdf_hydroelastic_config", None)
        if hydro_cfg is not None:
            cfg_dict["sdf_hydroelastic_config"] = HydroelasticSDF.Config(**hydro_cfg)
        return cfg_dict
