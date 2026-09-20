# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Newton VBD solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class VBDSolverCfg(NewtonSolverCfg):
    """Configuration for the Vertex Block Descent solver."""

    class_type: type[NewtonManager] | str = "{DIR}.vbd_manager:NewtonVBDManager"
    """Manager class for the VBD solver."""

    iterations: int = 10
    """Number of VBD iterations per substep."""

    integrate_with_external_rigid_solver: bool = False
    """Whether an external solver integrates rigid bodies."""

    particle_enable_self_contact: bool = False
    """Whether to enable particle self-contact."""

    particle_self_contact_radius: float = 0.005
    """Particle radius used for self-contact detection [m]."""

    particle_self_contact_margin: float = 0.005
    """Self-contact detection margin [m]."""

    particle_collision_detection_interval: int = -1
    """How often particle self-contact detection is applied.

    ``< 0``: once before initialization. ``0``: once before and once after
    initialization. ``k >= 1``: before every ``k`` VBD iterations.
    """

    particle_vertex_contact_buffer_size: int = 32
    """Preallocation size for each vertex contact buffer."""

    particle_edge_contact_buffer_size: int = 64
    """Preallocation size for each edge contact buffer."""

    particle_topological_contact_filter_threshold: int = 2
    """Topological distance below which self-contacts are discarded."""

    particle_rest_shape_contact_exclusion_radius: float = 0.0
    """Rest-shape separation threshold for filtering contacts [m]."""

    rigid_compliant_alm: bool | None = None
    """Whether to use compliant ALM for rigid contacts, joints, drives, and limits.

    ``None`` preserves Newton's default, which selects the legacy path in Newton 1.6
    and emits a deprecation warning when VBD integrates rigid bodies. Set ``True`` to
    adopt compliant ALM with finite authored stiffness, or ``False`` to retain the legacy
    path explicitly during migration. Material parameters may need retuning.
    """

    rigid_avbd_alpha: float | None = None
    """Shared C0 stabilization strength for rigid joints and body-body contacts.

    Values must be in ``[0, 1]``. ``None`` preserves Newton's mode defaults: ``0.95``
    for the legacy path and ``0.0`` for compliant ALM. Newton's joint-specific and
    contact-specific alpha overrides take precedence when supplied by a config subclass.
    """

    rigid_contact_hard: bool = True
    """Whether to use hard body-body contacts on the legacy VBD path.

    ``False`` selects legacy penalty-only contacts. This setting does not select the
    contact mode when :attr:`rigid_compliant_alm` is ``True``.

    .. deprecated:: Newton 1.6
        Set :attr:`rigid_compliant_alm` to ``True`` and author finite contact stiffness.
    """

    rigid_contact_k_start: float = 1.0e2
    """Initial stiffness seed for rigid-body contacts [N/m]."""

    rigid_body_contact_buffer_size: int = 64
    """Per-body capacity of the body-body contact list.

    Increase this value when Newton reports a per-body body-body contact buffer overflow.
    Only used when :attr:`integrate_with_external_rigid_solver` is ``False``.
    """

    rigid_body_particle_contact_buffer_size: int = 256
    """Per-body capacity of the particle, edge, and face soft-contact list.

    Increase this value when Newton reports a per-body particle contact buffer overflow.
    Only used when :attr:`integrate_with_external_rigid_solver` is ``False``.
    """
