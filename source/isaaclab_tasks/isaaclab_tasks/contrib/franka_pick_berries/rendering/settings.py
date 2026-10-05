# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in sampling/AA overrides; leave Gaussian, color and lighting settings alone."""

from importlib.metadata import version

from packaging.version import Version

from pxr import Sdf, Usd


def require_live_gaussian_renderer():
    """Reject the public renderer known to ghost live deformations in this task."""
    renderer_version = Version(version("ovrtx"))
    # This published MR build sorts before 0.6.0 under PEP 440.
    pinned_development_build = Version("0.6.0.dev382408+mr50035.92a010ff")
    if (
        renderer_version != pinned_development_build and not Version("0.6.0") <= renderer_version < Version("0.7.0")
    ) or Version(version("ovstage")) < Version("0.3.0"):
        raise RuntimeError(
            "Berry rendering requires the validated OVRTX 0.6 / OVStage 0.3 runtime. "
            "OVRTX 0.5 can read back updated arrays while rendering a ghosted field. "
            "Run bash source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/setup/setup.sh "
            "--renderer-internal "
            "(NVIDIA network access required); see the adjacent README.md. "
            "Physics-only --no_render does not require these wheels."
        )


def apply_sampling_settings(product: Usd.Prim, rtpt_spp: int | None = None) -> dict:
    """Author RTPT samples on one render product; antialiasing keeps the renderer's default.

    ``None`` preserves the inherited setting. RTPT uses ``rtpt:spp``, not the
    separate offline ``pt:samplesPerPixel`` attribute.
    """
    if rtpt_spp is not None and rtpt_spp < 1:
        raise ValueError("RTPT samples must be positive")
    authored = {}
    if rtpt_spp is not None:
        authored["omni:rtx:rtpt:spp"] = rtpt_spp
        product.CreateAttribute("omni:rtx:rtpt:spp", Sdf.ValueTypeNames.Int).Set(rtpt_spp)
    return authored
