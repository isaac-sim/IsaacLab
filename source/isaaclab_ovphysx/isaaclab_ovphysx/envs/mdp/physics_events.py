# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX backend implementations for MDP event terms.

OVPhysX shares the PhysX tensor APIs, so every term except material randomization reuses the
PhysX implementation (re-exported below). Material randomization has a dedicated no-op because
the OVPhysX material binding is not yet available.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from isaaclab_physx.envs.mdp.physics_events import (
    RandomizeActuatorGains,
    RandomizeFixedTendonParameters,
    RandomizeJointParameters,
    RandomizePhysicsSceneGravity,
    RandomizeRigidBodyColliderOffsets,
    RandomizeRigidBodyCom,
    RandomizeRigidBodyScale,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

logger = logging.getLogger(__name__)

__all__ = [
    "RandomizeActuatorGains",
    "RandomizeFixedTendonParameters",
    "RandomizeJointParameters",
    "RandomizePhysicsSceneGravity",
    "RandomizeRigidBodyColliderOffsets",
    "RandomizeRigidBodyCom",
    "RandomizeRigidBodyMaterial",
    "RandomizeRigidBodyScale",
]


class RandomizeRigidBodyMaterial:
    """OVPhysX no-op implementation for material randomization.

    OVPhysX has no material-randomization tensor binding yet (wheel-side gap): the
    ``RIGID_BODY_MATERIAL`` binding is missing and randomization would require per-body view
    creation that OVPhysX does not yet expose. This implementation warns once and does nothing.
    """

    def __init__(
        self, cfg: EventTermCfg, env: ManagerBasedEnv, asset: RigidObject | Articulation, asset_cfg: SceneEntityCfg
    ):
        logger.warning(
            "randomize_rigid_body_material is a no-op on the OVPhysX backend "
            "(wheel-side gap — see docs/superpowers/specs/2026-04-27-ovphysx-contact-api-gaps.md)."
        )

    def __call__(self, *args, **kwargs):
        pass
