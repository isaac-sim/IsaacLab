# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit USD-context ownership of the simulation stage."""

from __future__ import annotations

import contextlib
from dataclasses import field

import omni.kit.app
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


def show_stage_in_viewport(usd_path: str) -> None:
    """Open a USD file in the Kit viewport and keep the app running until the user closes it.

    Opens the stage through the Kit USD context so it appears in the viewport (or the
    livestream client), then spins the Kit update loop until the window is closed or the
    loop is interrupted. Must only be called inside a running Kit process with a GUI.

    Args:
        usd_path: Path of the USD file to display.

    Raises:
        RuntimeError: If the stage cannot be opened.
    """
    # A failed open leaves the previously loaded stage in the viewport, which would look like a
    # successful preview of the wrong asset, so surface the failure instead of blocking on it.
    result = omni.usd.get_context().open_stage(usd_path)
    opened = result[0] if isinstance(result, tuple) else result
    if opened is False:
        raise RuntimeError(f"Failed to open the USD stage in the Kit viewport: {usd_path}")

    app = omni.kit.app.get_app_interface()
    with contextlib.suppress(KeyboardInterrupt):
        while app.is_running():
            app.update()
