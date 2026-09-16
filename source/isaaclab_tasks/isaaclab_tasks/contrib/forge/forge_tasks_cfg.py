# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from isaaclab.utils import config_field

from isaaclab_tasks.contrib.factory.factory_tasks_cfg import FactoryTask, GearMesh, NutThread, PegInsert


@dataclass
class ForgeTask(FactoryTask):
    action_penalty_ee_scale: float = config_field(0.0)
    action_penalty_asset_scale: float = config_field(0.001)
    action_grad_penalty_scale: float = config_field(0.1)
    contact_penalty_scale: float = config_field(0.05)
    delay_until_ratio: float = config_field(0.25)
    contact_penalty_threshold_range: Any = config_field([5.0, 10.0])


@dataclass
class ForgePegInsert(PegInsert, ForgeTask):
    contact_penalty_scale: float = config_field(0.2)


@dataclass
class ForgeGearMesh(GearMesh, ForgeTask):
    contact_penalty_scale: float = config_field(0.05)


@dataclass
class ForgeNutThread(NutThread, ForgeTask):
    contact_penalty_scale: float = config_field(0.05)
