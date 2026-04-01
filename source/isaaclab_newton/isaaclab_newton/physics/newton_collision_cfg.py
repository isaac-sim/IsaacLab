# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton collision pipeline."""

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass


@configclass
class HydroelasticSDFCfg:
    """Configuration for SDF-based hydroelastic collision handling.

    Hydroelastic contacts generate distributed contact areas instead of point contacts,
    providing more realistic force distribution for manipulation and compliant surfaces.

    For more details, see the `Newton Collisions Guide`_.

    .. _Newton Collisions Guide: https://newton-physics.github.io/newton/latest/concepts/collisions.html#hydroelastic-contacts
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

    For more details, see the `Newton Collisions Guide`_ and `CollisionPipeline API`_.

    .. _Newton Collisions Guide: https://newton-physics.github.io/newton/latest/concepts/collisions.html
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
