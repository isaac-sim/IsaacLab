# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task handling presets, separate from the original asset's crush-demo metadata."""

import math


def workcell_grid(background: str, spacing: float) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
    """Return grid origin [m] and node counts, leaving room for discarded tissue.

    Args:
        background: Workcell background (``studio`` or ``ebc``).
        spacing: MPM grid-node spacing [m], not particle sampling distance.
    """
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Grid spacing must be finite and positive")
    origin = (-0.12, -0.12, -0.016 if background == "ebc" else -0.008)
    resolution = (121, 191 if background == "ebc" else 121, 121)
    if background == "ebc":
        # Extend toward the reject dish without moving existing grid nodes or
        # coarsening contact. Only active nodes are stepped by the solver.
        padding = math.ceil(0.04 / spacing)
        origin = (origin[0] - padding * spacing, origin[1] - padding * spacing, origin[2])
        resolution = (resolution[0] + padding, resolution[1] + padding, resolution[2])
    return origin, resolution


def simulation_parameters(profile: dict, physics_profile: str, mpm_hz: int | None = None) -> dict:
    """Return effective tissue parameters and a CFL-safe integration rate [Hz]."""
    if physics_profile not in ("handling", "legacy"):
        raise ValueError(f"Unknown berry physics profile: {physics_profile}")
    source = profile["simulation"]
    params = {key: value for key, value in source.items() if key not in ("binding", "grid", "hz")}
    berry = profile["berry"]
    params.update(h=source["grid"], bruise_stress=8500, bruise_rate=8)
    if berry == "raspberry":
        params["bulk_yield"] = 0.8
    frequency = 5040 if berry == "raspberry" else math.ceil(source["hz"] / 120) * 120
    if physics_profile == "handling":
        # Empirical handling priors, not a force/displacement fit to a single video.
        # Lower nu raises shear stiffness without an excessive bulk-wave timestep cost.
        params.update(
            young={"raspberry": 12000.0, "blackberry": 13500.0, "blueberry": 15000.0, "strawberry": 18000.0}[berry],
            poisson=0.4,
            bulk_yield=0.8 if berry == "raspberry" else 0.35,
            interface_yield=0.35,
            softening=0.25,
        )
        if source.get("tear_end", 0) > 0:
            params.update(tear_onset=2.5, tear_end=5.0)
        nu = params["poisson"]
        wave_speed = math.sqrt(params["young"] * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * params["density"]))
        frequency = max(frequency, math.ceil(wave_speed / (params["h"] * 0.44 * 120)) * 120)
    params["hz"] = frequency if mpm_hz is None else mpm_hz
    if params["hz"] <= 0 or params["hz"] % 120:
        raise ValueError("MPM frequency must be a positive multiple of 120 Hz")
    return params
