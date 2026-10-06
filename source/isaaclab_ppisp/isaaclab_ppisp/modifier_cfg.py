# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the PPISP observation modifier."""

from typing import Literal

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg

from .cfg import PpispCfg, PpispDiscoveryMode
from .modifier import PpispModifier


@configclass
class PpispModifierCfg(ModifierCfg):
    """Configure PPISP in :attr:`ObservationTermCfg.modifiers`."""

    func: type[PpispModifier] = PpispModifier
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera")
    isp_cfg: PpispCfg | PpispDiscoveryMode | None = PpispDiscoveryMode.AUTO_CAMERA
    output: Literal["rgb", "rgba"] = "rgb"
    input_source: Literal["camera", "previous"] = "camera"
    normalize: bool = False
    permute: bool = False
