# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mass-preserving tissue resolution presets; Gaussian appearance is untouched."""

import numpy as np


def physics_resolution(proxy: dict, parameters: dict, preset: str) -> tuple[dict, dict, dict]:
    """Select a tissue quadrature and preserve its total volume [m³] and mass [kg].

    The half preset retains alternating sites of a regular cubic lattice, not
    alternating array entries. Contact spacing represents the new volume per
    particle. Grid spacing and integration rate remain unchanged.
    """
    if preset not in ("full", "half"):
        raise ValueError(f"Unknown physics resolution: {preset}")
    count = len(proxy["xyz"])
    spacing = float(proxy["spacing"])
    volume = float(parameters.get("particle_volume", spacing**3))
    reduced, params = proxy, dict(parameters)
    if preset == "half":
        xyz = np.asarray(proxy["xyz"])
        lattice = (xyz - xyz.min(0)) / spacing
        cells = np.rint(lattice).astype(np.int64)
        if not np.allclose(lattice, cells, atol=1e-3, rtol=0):
            raise ValueError(
                "Half resolution requires regular-lattice tissue; use --asset_version v2 or full resolution"
            )
        if len(np.unique(proxy["regions"])) != 1:
            raise ValueError("Half resolution currently requires a single tissue material region")
        keep = cells.sum(1) % 2 == 0
        if keep.sum() < 32 or keep.sum() == count:
            raise ValueError("Not enough spatially distributed particles for half-resolution Gaussian binding")
        volume *= count / int(keep.sum())
        # Keep the asset and its arrays intact; Gaussian bindings are rebuilt
        # against these positions by BerryGaussianStream at viewer creation.
        reduced = {
            "xyz": xyz[keep].copy(),
            "regions": np.asarray(proxy["regions"])[keep].copy(),
            "interface": np.asarray(proxy["interface"])[keep].copy(),
            "spacing": np.float32(np.cbrt(volume)),
            "particle_volume": np.float64(volume),
        }
        params["particle_volume"] = volume
    report = {
        "preset": preset,
        "source_particles": count,
        "particles": len(reduced["xyz"]),
        "particle_volume_m3": volume,
        "contact_spacing_m": float(reduced["spacing"]),
        "mass_kg": volume * len(reduced["xyz"]) * params["density"],
    }
    return reduced, params, report
