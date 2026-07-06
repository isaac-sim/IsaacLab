# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for Newton coupled solvers.

The configurations describe named solver entries and the interfaces that
couple them. A manager turns each entry's ownership selectors into a Newton
``ModelView`` and dispatches to the selected experimental coupled solver.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, field
from typing import TYPE_CHECKING, Literal

from isaaclab_newton.physics import NewtonSolverCfg

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager
    from newton import CollisionPipeline, ModelView


@configclass
class CoupledSolverEntryCfg:
    """Configuration for one named sub-solver and its model ownership.

    Bodies are selected by scene entity or full Newton body-label regex.
    Joints and shapes attached to selected bodies are included by default;
    additional shapes can be selected directly by their full labels.
    """

    name: str = MISSING
    """Unique name used by coupling mappings to reference this entry."""

    solver_cfg: NewtonSolverCfg = MISSING
    """Configuration used to construct this entry's Newton solver."""

    bodies: list[SceneEntityCfg | str] = field(default_factory=list)
    """Bodies owned by this entry.

    A :class:`~isaaclab.managers.SceneEntityCfg` selects bodies below the
    corresponding scene asset and may narrow them with ``body_names``. A
    string is treated as a regex matched against full Newton body labels.
    """

    particles: list[int] = field(default_factory=list)
    """Parent-model particle indices owned by this entry."""

    all_particles: bool = False
    """Whether this entry owns every particle in the parent model."""

    include_child_joints: bool = True
    """Whether joints whose child body is selected are owned by this entry."""

    include_body_shapes: bool = True
    """Whether shapes attached to selected bodies are owned by this entry."""

    include_static_shapes: bool = False
    """Whether this entry owns all shapes whose body index is ``-1``."""

    shape_label_patterns: list[str] = field(default_factory=list)
    """Regexes matched against full Newton shape labels for additional ownership."""


@configclass
class CoupledProxyCfg:
    """Configuration for one directed virtual-proxy mapping."""

    source: str = MISSING
    """Name of the entry that owns the source bodies."""

    destination: str = MISSING
    """Name of the entry that receives the proxy bodies."""

    bodies: list[SceneEntityCfg | str] = field(default_factory=list)
    """Source bodies exposed as proxies in the destination entry.

    Selectors use the same scene-entity and full-label-regex semantics as
    :attr:`CoupledSolverEntryCfg.bodies`.
    """

    particles: list[int] = field(default_factory=list)
    """Source particle indices exposed as proxies in the destination entry."""

    mode: Literal["lagged", "staggered"] = "lagged"
    """Proxy transfer mode passed to Newton's coupled-proxy solver."""

    mass_scale: float = 1.0
    """Scale applied to proxy effective mass and inertia in the destination view."""

    collide_interval: int | None = None
    """Collision refresh interval when :attr:`collision_pipeline_factory` is set.

    ``None`` refreshes contacts on every proxy pass. Explicit values must be
    positive integers.
    """

    collision_pipeline_factory: Callable[[ModelView], CollisionPipeline | None] | None = None
    """Optional factory for the proxy destination's collision pipeline."""

    shape_material_ke: float | None = None
    """Optional contact-stiffness override for shapes on selected proxy bodies [N/m]."""

    shape_material_kd: float | None = None
    """Optional contact-damping override for shapes on selected proxy bodies [N*s/m]."""

    shape_material_mu: float | None = None
    """Optional friction override for shapes on selected proxy bodies."""

    shape_margin: float | None = None
    """Optional contact-margin override for shapes on selected proxy bodies [m]."""

    shape_gap: float | None = None
    """Optional contact-gap override for shapes on selected proxy bodies [m]."""


@configclass
class CoupledSolverCfg(NewtonSolverCfg):
    """Base configuration for a Newton experimental coupled solver.

    Every body and particle in the parent model must have one unambiguous
    owner among :attr:`entries`. Joint and shape ownership is also unique, but
    may intentionally omit constraints or collision geometry from all views.
    Use a concrete subclass to configure the coupling interfaces.
    """

    class_type: type[NewtonManager] | str = "{DIR}.coupled_manager:NewtonCoupledSolverManager"
    """Manager class for the coupled solver."""

    entries: list[CoupledSolverEntryCfg] = field(default_factory=list)
    """Ordered named sub-solver entries and their ownership selectors."""


@configclass
class CoupledProxySolverCfg(CoupledSolverCfg):
    """Configuration for Newton's lagged-impulse virtual-proxy coupling."""

    proxies: list[CoupledProxyCfg] = field(default_factory=list)
    """Directed proxy mappings between named solver entries."""

    iterations: int = 1
    """Number of proxy relaxation passes per coupled step."""


@configclass
class CoupledAdmmSolverCfg(CoupledSolverCfg):
    """Configuration for Newton's linearized ADMM coupling."""

    contact_pairs: list[tuple[str, str]] | None = None
    """Symmetric contact interfaces between named solver entries.

    ``None`` asks Newton to detect every distinct entry pair automatically.
    An empty list disables ADMM contact coupling.
    """

    iterations: int = 5
    """Number of ADMM dual iterations per coupled step."""

    rho: float = 1.0
    """ADMM penalty parameter [dimensionless]."""

    gamma: float = 0.0
    """Proximal mass scaling parameter [dimensionless]."""

    baumgarte: float = 0.0
    """Position-error correction fraction [dimensionless]."""

    joint_stiffness: float = 1.0e4
    """Translational cross-solver joint stiffness [N/m]."""

    joint_damping: float = 0.0
    """Translational cross-solver joint damping [N*s/m]."""

    joint_angular_stiffness: float = 1.0e4
    """Angular cross-solver joint stiffness [N*m/rad]."""

    joint_angular_damping: float = 0.0
    """Angular cross-solver joint damping [N*m*s/rad]."""

    joint_proximal_bodies: bool = True
    """Whether cross-solver joint neighbors remain visible as inertial proxies."""

    joint_proximal_destination_entries: list[str] | None = None
    """Optional entries that receive cross-solver joint proximal bodies."""

    joint_proximal_mass_scale: float = 1.0
    """Mass scale applied to cross-solver joint proximal bodies."""

    rigid_contact_matching: Literal["disabled", "latest", "sticky"] = "disabled"
    """Frame-to-frame matching mode for collision-detected rigid contacts."""

    contact_matching_pos_threshold: float | None = None
    """Maximum midpoint distance for matching rigid contacts [m]."""

    contact_matching_normal_dot_threshold: float | None = None
    """Minimum normal dot product for matching rigid contacts."""

    contact_matching_force_scale: float = 0.9
    """Scale applied to the previous ADMM dual when a rigid contact matches."""
