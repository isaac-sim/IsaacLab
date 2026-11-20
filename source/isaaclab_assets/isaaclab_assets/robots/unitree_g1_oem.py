# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Revision-pinned Unitree G1 USD packages retrieved through SimReady."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import replace

if TYPE_CHECKING:
    from isaaclab.sim import UsdFileCfg

G1_OEM_BARE_URI = (
    "hf://unitreerobotics/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0@323e350252b9c3aee9c40acbfdad84f6ce46a5ac"
)
G1_OEM_FLOATING_DEX3_URI = (
    "hf://unitreerobotics/unitree_sim_isaaclab_usds/assets/robots/g1-29dof_wholebody_dex3"
    "@27b13d380531c0c6ea6ab109511ee4872594e271"
)


def retrieve_g1_oem_usd(uri: str) -> str:
    """Return the primary USD of a pinned Unitree package from SimReady Retrieve's cache."""
    try:
        result = subprocess.run(["simready-retrieve", "fetch", uri], check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "The G1 OEM preset requires the 'simready-retrieve' CLI on PATH. "
            "See https://nvidia.github.io/simready-foundation/latest/guides/libraries.html for installation."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"SimReady Retrieve could not fetch {uri}: {error.stderr.strip()}") from error

    path = Path(result.stdout.strip())
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError(f"SimReady Retrieve did not return a local USD for {uri}: {result.stdout.strip()!r}")
    return str(path)


def spawn_g1_oem_usd(prim_path: str, cfg: UsdFileCfg, translation=None, orientation=None, **kwargs):
    """Use the standard USD spawner after materializing the OEM package."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    local_cfg = replace(cfg, usd_path=retrieve_g1_oem_usd(cfg.usd_path))
    return spawn_from_usd(prim_path, local_cfg, translation, orientation, **kwargs)
