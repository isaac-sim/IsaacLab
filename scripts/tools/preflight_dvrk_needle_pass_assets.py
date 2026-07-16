# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the remote assets pinned by the dVRK needle-pass task."""

from pathlib import Path
from runpy import run_path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
needle_assets = run_path(REPOSITORY_ROOT / "source/isaaclab_tasks/isaaclab_tasks/contrib/needle_pass/assets.py")
dvrk_asset = run_path(REPOSITORY_ROOT / "source/isaaclab_assets/isaaclab_assets/robots/dvrk_asset.py")

NEEDLE_ASSET = needle_assets["NEEDLE_ASSET"]
SUTURE_PAD_ASSET = needle_assets["SUTURE_PAD_ASSET"]
verify_remote_asset_sha256 = needle_assets["verify_remote_asset_sha256"]
DVRK_PSM_USD_PATH = dvrk_asset["DVRK_PSM_USD_PATH"]
DVRK_PSM_USD_SHA256 = dvrk_asset["DVRK_PSM_USD_SHA256"]


def main() -> None:
    """Download and hash each task asset as an explicit online preflight."""

    assets = (
        ("needle", NEEDLE_ASSET.url, NEEDLE_ASSET.sha256),
        ("suture pad", SUTURE_PAD_ASSET.url, SUTURE_PAD_ASSET.sha256),
        ("dVRK PSM", DVRK_PSM_USD_PATH, DVRK_PSM_USD_SHA256),
    )
    for name, url, sha256 in assets:
        verify_remote_asset_sha256(url, sha256)
        print(f"Verified {name}: {sha256}")


if __name__ == "__main__":
    main()
