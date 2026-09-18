# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset roots shared by the RLinf VLA post-training tasks.

Assemble-trocar and the two H2 + Sharpa tasks draw their scene USDs from one Hugging Face dataset
repository laid out like the Isaac Healthcare bundle, so overlapping props are fetched and cached
once. The robot USDs are declared next to the articulation instead, in ``isaaclab_assets.robots``.

Point ``ISAACLAB_RLINF_DEMO_ASSET_ROOT`` at a mirror keeping the same subtrees -- a
``snapshot_download`` of the repository, for instance -- to serve them from disk. The layout matches
the Isaac Healthcare bundle so the tree can be published to the Isaac asset server as-is; this is the
only constant to update once it is.
"""

from __future__ import annotations

import os

RLINF_DEMO_ASSET_ROOT: str = os.environ.get(
    "ISAACLAB_RLINF_DEMO_ASSET_ROOT",
    "https://huggingface.co/datasets/LiFanxing/IsaacLabRLinfDemo/resolve/main/assets",
)
"""Root of the asset bundle shared by the RLinf demo tasks."""

# Props powered by Lightwheel (https://lightwheel.ai/). These assets are licensed under the Creative
# Commons Attribution-NonCommercial 4.0 International License, whose terms are collected in
# ``docs/licenses/assets/lightwheel-license.txt``. Per-asset terms are served next to each USD, as
# ``LICENSE.txt`` in place of the file's base name.
PROP_ASSET_ROOT: str = f"{RLINF_DEMO_ASSET_ROOT}/Props/Lightwheel"
"""Table, object and background USDs."""

NUREC_ASSET_ROOT: str = f"{RLINF_DEMO_ASSET_ROOT}/Props/NuRec"
"""NuRec background captures."""
