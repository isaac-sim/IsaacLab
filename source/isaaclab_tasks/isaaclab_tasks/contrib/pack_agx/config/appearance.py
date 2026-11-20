# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD composition for the two calibrated packing props."""

from pathlib import Path

from pxr import Usd

from isaaclab.sim import UsdFileCfg, clone
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd_file


@clone
def spawn_pack_prop(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Compose the packing scene's appearance over the prop from the configured asset server."""
    prim = spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    appearance = Path(__file__).with_name("pack_appearance.usda")
    prim.GetReferences().AddReference(
        str(appearance), f"/{Path(cfg.usd_path).stem}", position=Usd.ListPositionFrontOfPrependList
    )
    return prim
