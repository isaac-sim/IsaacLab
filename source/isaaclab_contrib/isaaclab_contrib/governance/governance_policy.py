# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Governance policy definitions for safety-constrained environments."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class GovernancePolicy:
    """Declarative safety policy for governing agent actions.

    Defines the constraints that must be satisfied before any action is
    applied to the environment. Actions that violate these constraints
    are either clamped (modified to satisfy bounds) or denied (replaced
    with a safe default).

    The policy follows a *permission-before-power* principle: every action
    must pass governance checks before it reaches the physics simulation.

    Attributes:
        action_bounds: Per-dimension (min, max) for action values. Actions
            outside these bounds are clamped. If ``None``, no bound check
            is performed.
        max_rate_of_change: Maximum allowed change in action value between
            consecutive steps (per dimension). Prevents sudden jumps that
            could damage hardware during sim-to-real transfer.
            If ``None``, no rate limit is applied.
        spatial_bounds: Axis-aligned bounding box for agent positions.
            Keys are coordinate names (e.g., ``"x"``, ``"y"``, ``"z"``),
            values are ``(min, max)`` tuples. If an agent's projected
            position would leave this box, the action is denied.
            If ``None``, no spatial check is performed.
        geofence_polygons: List of geofence polygons. Each polygon is a
            list of ``(x, y)`` vertices defining an authorized operating
            region. If provided, agents must remain within at least one
            polygon. Uses ray-casting point-in-polygon test.
        min_separation: Minimum allowed distance between any two agents
            (meters). If a proposed action would bring agents closer than
            this threshold, the offending action is clamped.
            If ``None``, no separation check is performed.
        fail_closed: If ``True`` (default), any governance check failure
            results in a safe default action (zeros). If ``False``,
            violations are logged but actions pass through unclamped.
        require_external_verification: If ``True``, actions are sent to
            an external governance gateway for cryptographic verification
            before execution. Default ``False`` (local-only checks).
        gateway_url: URL of the external governance gateway. Only used
            when ``require_external_verification`` is ``True``.
        metadata: Arbitrary metadata attached to every provenance record.
    """

    action_bounds: tuple[float, float] | None = None
    max_rate_of_change: float | None = None
    spatial_bounds: dict[str, tuple[float, float]] | None = None
    geofence_polygons: list[list[tuple[float, float]]] | None = None
    min_separation: float | None = None
    fail_closed: bool = True
    require_external_verification: bool = False
    gateway_url: str | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate policy configuration."""
        if self.require_external_verification and not self.gateway_url:
            raise ValueError(
                "gateway_url must be set when require_external_verification is True"
            )
        if self.action_bounds is not None:
            lo, hi = self.action_bounds
            if lo >= hi:
                raise ValueError(
                    f"action_bounds lower ({lo}) must be strictly less than upper ({hi})"
                )
        if self.max_rate_of_change is not None and self.max_rate_of_change <= 0:
            raise ValueError("max_rate_of_change must be positive")
        if self.min_separation is not None and self.min_separation <= 0:
            raise ValueError("min_separation must be positive")
