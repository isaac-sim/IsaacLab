# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-render PPISP (Physically Plausible Image Signal Processing) for IsaacLab.

Provides the renderer-backend-agnostic ISP pipeline that converts the renderer's
HDR scene-linear AOV to LDR RGBA at the end of a render tick. Renderer backends
(:class:`~isaaclab_physx.renderers.IsaacRtxRenderer`,
:class:`~isaaclab_ov.renderers.OVRTXRenderer`,
:class:`~isaaclab_newton.renderers.NewtonWarpRenderer`) compose
:class:`PpispPipeline` internally when the camera's
:attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg` is set.
"""

import os

import toml

from isaaclab.utils.module import lazy_export

ISAACLAB_PPISP_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_toml_path = os.path.join(_pkg_dir, "config", "extension.toml")
if not os.path.isfile(_toml_path):
    _toml_path = os.path.join(ISAACLAB_PPISP_EXT_DIR, "config", "extension.toml")

ISAACLAB_PPISP_METADATA = toml.load(_toml_path)
"""Extension metadata dictionary parsed from the extension.toml file."""

__version__ = ISAACLAB_PPISP_METADATA["package"]["version"]

lazy_export()
