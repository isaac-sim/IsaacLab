# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spacemouse device for SE(2) and SE(3) control."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .se2_spacemouse import Se2SpaceMouse
    from .se2_spacemouse_cfg import Se2SpaceMouseCfg
    from .se3_spacemouse import Se3SpaceMouse
    from .se3_spacemouse_cfg import Se3SpaceMouseCfg

from isaaclab.utils.module import lazy_export

lazy_export(
    ("se2_spacemouse", "Se2SpaceMouse"),
    ("se2_spacemouse_cfg", "Se2SpaceMouseCfg"),
    ("se3_spacemouse", "Se3SpaceMouse"),
    ("se3_spacemouse_cfg", "Se3SpaceMouseCfg"),
)
