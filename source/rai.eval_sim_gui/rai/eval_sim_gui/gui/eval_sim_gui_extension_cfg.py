# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass
from rai.eval_sim.eval_sim import EvalSimCfg


@configclass
class WindowCfg:
    title: str = "Isaac Lab Evaluation Simulator (EvalSim)"
    visible: bool = True
    width: int = 350
    height: int = 450


@configclass
class EvalSimGUIExtensionCfg:
    # EvalSim settings. By default, these are loaded from the user's config file.
    eval_sim: EvalSimCfg = EvalSimCfg.from_yaml()

    # window settings
    window: WindowCfg = WindowCfg()
