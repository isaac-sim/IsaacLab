# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from isaaclab.utils import configclass
from isaaclab.utils.tensor_processors import DictTensorProcessorCfg

from .cfg import PpispCfg


@configclass
class PpispProcessorCfg(DictTensorProcessorCfg):
    """Process one scene-linear HDR tensor into RGB or RGBA.

    ``inputs`` and ``outputs`` each contain one actual dictionary key. Supply concrete PPISP
    settings in ``params={"cfg": ppisp_cfg}``. Add ``"channels": 4`` for RGBA. The source camera
    needs explicit neutral exposure; this processor never changes renderer settings.
    """

    class_type = "{DIR}.processor:PpispProcessor"
    inputs: list[str] = ["rgb_hdr"]
    outputs: list[str] = ["rgb"]
    params: dict[str, Any] = {"cfg": PpispCfg()}
