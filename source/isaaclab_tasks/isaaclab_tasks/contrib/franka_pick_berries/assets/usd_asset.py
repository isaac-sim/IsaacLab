# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read Gaussian appearance and MPM tissue directly from one USD/USDZ asset."""

from pathlib import Path

import numpy as np

from pxr import Usd

from isaaclab.utils.assets import retrieve_file_path

from .usd_metadata import read_array, read_data


def load_berry(path: str | Path) -> tuple[Usd.Stage, dict, dict, dict]:
    """Load a berry's appearance, tissue positions [m], and simulation metadata.

    ``path`` may be a Nucleus URL or a local path; it is resolved and locally cached
    via :func:`isaaclab.utils.assets.retrieve_file_path`.
    """
    stage = Usd.Stage.Open(retrieve_file_path(str(path)))
    if not stage or stage.GetDefaultPrim().GetAttribute("berry:schemaVersion").Get() != 2:
        raise ValueError(f"Expected a schema-v2 self-contained berry USD asset: {path}")
    gaussians = stage.GetPrimAtPath("/Berry/Gaussians")
    asset = {}
    for key, name in {
        "xyz": "positions",
        "scales": "scales",
        "rotations": "orientations",
        "alpha": "opacities",
        "sh": "radiance:sphericalHarmonicsCoefficients",
    }.items():
        asset[key] = np.asarray(gaussians.GetAttribute(name).Get(), dtype=np.float32).copy()
    asset["sh"] = asset["sh"].reshape(-1, 16, 3)
    for key in gaussians.GetAttribute("berry:arrayKeys").Get():
        asset[key] = read_array(gaussians, "berry:arrays:" + key)
    tissue = stage.GetPrimAtPath("/Berry/Tissue")
    proxy = {key: read_array(tissue, "berry:arrays:" + key) for key in tissue.GetAttribute("berry:arrayKeys").Get()}
    proxy["xyz"] = np.asarray(tissue.GetAttribute("points").Get(), dtype=np.float32).copy()
    return stage, asset, proxy, read_data(stage, "/Berry/TaskData/Profile")
