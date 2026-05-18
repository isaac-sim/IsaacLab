# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for the cable / 1D-rod asset (Newton backend).

Mirrors the structure of :mod:`isaaclab_contrib.deformable`: the asset class
and its cfg are re-exported at package level, while the replicate-hook
plumbing (registry entry and per-world builder hooks) stays accessible through
:mod:`isaaclab_contrib.cable.cable_object` for callers that wire it from
solver managers.
"""

from isaaclab.utils.module import lazy_export

from .attachment_cfg import CableAttachmentCfg

__all__ = ["CableAttachmentCfg"]

lazy_export()
