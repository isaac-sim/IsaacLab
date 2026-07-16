# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity locomotion experimental task registrations (manager-based).

The per-robot ``*-Warp-v0`` variants reuse the stable flat cfgs and only
disable the randomization events that have no warp twins yet (see
:func:`disable_unsupported_randomization_events`). Once those twins exist,
the variants can be dropped in favor of ``--frontend warp`` on the stable
task ids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_experimental.envs.frontend import register_mdp_route

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg

# Warp twins for the stable velocity MDP terms live in this package's ``mdp``.
register_mdp_route("isaaclab_tasks.core.velocity", f"{__name__}.mdp")


def disable_unsupported_randomization_events(cfg: ManagerBasedRLEnvCfg) -> None:
    """Disable stable randomization events that have no warp twins yet.

    The warp event manager invokes term functions with a Warp env mask, and no
    warp twins exist yet for the rigid-body material/mass randomization events.
    A stable term on a warp manager is a hard error at adaptation time, so the
    warp task variants disable these terms until twins are available.
    """
    for name in ("physics_material", "add_base_mass"):
        if getattr(cfg.events, name, None) is not None:
            setattr(cfg.events, name, None)
