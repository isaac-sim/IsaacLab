# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rizon4s arm and Sharpa right hand used by the manipulation demos.

The hosted robot and its fourteen texture maps are the Fabrics-Sim bundle at
revision d0dbd1ddaefc4996db546949a7dfb37e39afcbeb.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, read_file

if TYPE_CHECKING:
    from isaaclab.actuators import ActuatorBaseCfg


_ROBOT_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Environments/RizonSharpaRJ45/Robots/Rizon4sSharpa"
_ROBOT_FILENAME = "rizon4s_sharpa_no_spheres_generated.usd"
ROBOT_USD_PATH = f"{_ROBOT_DIR}/{_ROBOT_FILENAME}"
ROBOT_BUNDLE_SHA256 = "ae5d22792b44fb6d29a7691d4276bc061a5529132f01e7a0eb5795a482595d63"
_ASSET_ROOT_ENV = "ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT"
# Order is part of the published bundle digest.
_BUNDLE_FILES = (
    _ROBOT_FILENAME,
    "textures/t_Rizon4s_EmissiveMask.1001.png",
    "textures/t_Rizon4s_EmissiveMask.1002.png",
    *(f"textures/t_Rizon4s_{channel}.{tile}.png" for channel in ("alb", "nor", "orm") for tile in (1001, 1002, 1003)),
    *(f"textures/t_Sharpa_{channel}.png" for channel in ("alb", "nor", "orm")),
)

PALM_BODY_NAME = "r_palm_ctrl"
"""Palm frame with X toward the knuckles and Z out of the palm."""

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
HAND_JOINT_NAMES = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)


def arm_actuators() -> dict[str, ActuatorBaseCfg]:
    """Use the Rizon4s arm drives with friction and armature from the Sharpa robot USD."""
    from isaaclab_assets.robots.flexiv import FLEXIV_RIZON4S_CFG

    return {
        name: actuator.replace(friction=None, armature=None) for name, actuator in FLEXIV_RIZON4S_CFG.actuators.items()
    }


@lru_cache(maxsize=4)
def _verified_usd(directory: Path) -> Path:
    manifest = hashlib.sha256()
    for name in _BUNDLE_FILES:
        with (directory / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest.update(f"{digest}  {name}\n".encode())
    if manifest.hexdigest() != ROBOT_BUNDLE_SHA256:
        raise ValueError(f"Rizon4s/Sharpa USD or textures in {directory} differ from the pinned asset bundle.")
    return directory / _ROBOT_FILENAME


def resolve_robot_usd() -> str:
    """Return the hosted USD or verify the bundle specified by ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT."""
    if configured := os.environ.get(_ASSET_ROOT_ENV):
        root = Path(configured).expanduser().resolve()
        return str(_verified_usd(root.parent if root.suffix.lower() in {".usd", ".usda", ".usdc"} else root))
    return ROBOT_USD_PATH


def local_robot_usd() -> Path:
    """Return the verified local bundle for tools that edit or compose its USD layers."""
    if os.environ.get(_ASSET_ROOT_ENV):
        return Path(resolve_robot_usd())
    root = Path.home() / ".cache/isaaclab/fabrics-sim/rizon4s-sharpa/sha256" / ROBOT_BUNDLE_SHA256
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{root}.lock"):
        missing = [name for name in _BUNDLE_FILES if not (root / name).is_file()]
        if missing:
            with tempfile.TemporaryDirectory(dir=root.parent) as temporary:
                staging = Path(temporary)
                for name in _BUNDLE_FILES:
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if name in missing:
                        target.write_bytes(read_file(f"{_ROBOT_DIR}/{name}").getbuffer())
                    else:
                        shutil.copyfile(root / name, target)
                _verified_usd(staging)
                if root.exists():
                    for name in missing:
                        (root / name).parent.mkdir(parents=True, exist_ok=True)
                        (staging / name).replace(root / name)
                else:
                    staging.rename(root)
        return _verified_usd(root)
