# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Berry tissue stiffness and the particle material that spawns it, shared by the tissue solvers."""

from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg

# Empirical handling priors [Pa], not a force/displacement fit to measured fruit.
YOUNG_MODULUS = {"raspberry": 12000.0, "blackberry": 13500.0, "blueberry": 15000.0, "strawberry": 18000.0}
POISSON_RATIO = 0.4


def particle_material(profile: dict) -> MPMParticleMaterialCfg:
    """Return the particle material of a berry from its asset profile: density, stiffness and friction.

    Each tissue solver sets its own yield limits on top (``set_tissue``).

    Args:
        profile: The berry asset's ``/Berry/TaskData/Profile`` metadata.
    """
    source = profile["simulation"]
    return MPMParticleMaterialCfg(
        density=float(source["density"]),
        young_modulus=YOUNG_MODULUS[profile["berry"]],
        poisson_ratio=POISSON_RATIO,
        friction=float(source["friction"]),
    )
