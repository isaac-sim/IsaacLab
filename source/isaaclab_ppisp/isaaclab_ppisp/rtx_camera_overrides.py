# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated RTX camera exposure overrides; the RTX renderers now apply them themselves."""

from __future__ import annotations

import warnings
from typing import Any


def apply_rtx_exposure_overrides(stage: Any, prim_paths: list[str]) -> None:
    """Pin RTX-side exposure to neutral and apply the RTX exposure API schemas on each camera prim.

    .. deprecated:: 3.0
        The Isaac RTX and OVRTX renderers apply these overrides themselves to cameras with PPISP or an
        ``rgb_radiance`` output. Remove the call. This function will be removed in a future release.

    Args:
        stage: USD stage containing the camera prims.
        prim_paths: Resolved camera prim paths.
    """
    warnings.warn(
        "isaaclab_ppisp.apply_rtx_exposure_overrides is deprecated: the Isaac RTX and OVRTX renderers apply it "
        "themselves to cameras with PPISP or an rgb_radiance output. Remove the call.",
        DeprecationWarning,
        stacklevel=2,
    )
    from isaaclab.renderers.rtx_camera_overrides import _apply_rtx_exposure_overrides

    _apply_rtx_exposure_overrides(stage, prim_paths)
