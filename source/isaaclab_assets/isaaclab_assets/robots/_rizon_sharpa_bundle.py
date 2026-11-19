# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the pinned Fabrics-Sim bundle for the RJ45 and teapot demos."""

import hashlib
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from filelock import FileLock

from isaaclab.utils.assets import read_file

from .rizon_sharpa import ROBOT_USD_PATH

_ROBOT_DIR, _ROBOT_FILENAME = ROBOT_USD_PATH.rsplit("/", 1)
ROBOT_BUNDLE_SHA256 = "ae5d22792b44fb6d29a7691d4276bc061a5529132f01e7a0eb5795a482595d63"
ROBOT_ASSET_ROOT_ENV = "ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT"
# Order is part of the published bundle digest.
_BUNDLE_FILES = (
    _ROBOT_FILENAME,
    "textures/t_Rizon4s_EmissiveMask.1001.png",
    "textures/t_Rizon4s_EmissiveMask.1002.png",
    *(f"textures/t_Rizon4s_{channel}.{tile}.png" for channel in ("alb", "nor", "orm") for tile in (1001, 1002, 1003)),
    *(f"textures/t_Sharpa_{channel}.png" for channel in ("alb", "nor", "orm")),
)


def resolve_robot_usd() -> str:
    """Return the hosted USD or verify the bundle specified by ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT."""
    if configured := os.environ.get(ROBOT_ASSET_ROOT_ENV):
        root = Path(configured).expanduser()
        directory = root.parent if root.suffix.lower() in {".usd", ".usda", ".usdc"} else root
        return str(_verified_usd(directory.resolve()))
    return ROBOT_USD_PATH


def local_robot_usd() -> Path:
    """Return the verified local bundle for tools that edit or compose its USD layers."""
    if os.environ.get(ROBOT_ASSET_ROOT_ENV):
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


@lru_cache(maxsize=4)
def _verified_usd(directory: Path) -> Path:
    manifest = hashlib.sha256()
    for name in _BUNDLE_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing Rizon4s Sharpa asset: {path}. Set {ROBOT_ASSET_ROOT_ENV} to the bundle directory."
            )
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest.update(f"{digest}  {name}\n".encode())
    digest = manifest.hexdigest()
    if digest != ROBOT_BUNDLE_SHA256:
        raise RuntimeError(f"Rizon4s Sharpa bundle SHA-256 mismatch: expected {ROBOT_BUNDLE_SHA256}, got {digest}.")
    return directory / _ROBOT_FILENAME
