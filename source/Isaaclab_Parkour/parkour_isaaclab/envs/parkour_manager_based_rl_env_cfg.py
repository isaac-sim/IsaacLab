# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from parkour_isaaclab.envs.parkour_ui import ParkourManagerBasedRLEnvWindow

@configclass
class ParkourManagerBasedRLEnvCfg(ManagerBasedRLEnvCfg):
    ui_window_class_type: type | None = ParkourManagerBasedRLEnvWindow
    parkours: object = MISSING