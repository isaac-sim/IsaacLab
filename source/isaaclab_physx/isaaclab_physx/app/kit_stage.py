# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit USD-context ownership of the simulation stage."""

from __future__ import annotations

from dataclasses import field

import omni.usd
from pxr import Usd, UsdUtils

from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass


class KitStageBackend:
    """Attach the simulation stage to Kit's USD context, where Kit extensions discover it.

    Kit extensions such as PhysX views, articulations, and the viewport read the stage of the
    USD context rather than Isaac Lab's current stage.
    """

    def __init__(self, cfg: KitStageBackendCfg):
        context = omni.usd.get_context()
        if context.get_stage() is not cfg.stage:
            context.attach_stage_with_callback(UsdUtils.StageCache.Get().GetId(cfg.stage).ToLongInt())

    def close(self) -> None:
        """Close the stage of Kit's USD context.

        The simulation closes backends before it clears the stage cache; clearing the cache first
        makes Kit fail with "Removal of UsdStage from cache failed" and can hang teardown.
        """
        omni.usd.get_context().close_stage()


@configclass
class KitStageBackendCfg(BackendCfg):
    """Kit USD-context attachment; the stage is borrowed from the active simulation."""

    class_type: type = KitStageBackend
    stage: Usd.Stage = field(kw_only=True, metadata={"copy": False})
