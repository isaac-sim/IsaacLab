# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities to detect physics backend from config without importing Kit or backends."""


def physics_backend_requires_kit(env_cfg) -> bool:
    """Return True if the env's physics backend requires Isaac Sim Kit (e.g. PhysX), False otherwise.

    Used to decide whether to launch AppLauncher before creating the environment.
    Newton backend does not require Kit; PhysX and default (None) do.

    This only inspects the config and type name of ``env_cfg.sim.physics``, so it is safe to call
    before any Kit/Omniverse imports.

    Args:
        env_cfg: Environment config with a ``sim`` attribute that has a ``physics`` attribute
            (e.g. ManagerBasedRLEnvCfg, DirectRLEnvCfg).

    Returns:
        True if Kit must be launched (PhysX or physics is None); False if Newton (no Kit).
    """
    sim = getattr(env_cfg, "sim", None)
    if sim is None:
        return True
    physics = getattr(sim, "physics", None)
    if physics is None:
        return True  # SimulationContext defaults to PhysxCfg when None
    name = type(physics).__name__
    # Newton backend does not require Isaac Sim Kit
    if name == "NewtonCfg":
        return False
    return True
