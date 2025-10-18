
from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.utils import configclass
# from isaaclab.utils.modifiers import ModifierCfg
# from isaaclab.utils.noise import NoiseCfg

# from isaaclab.managers.scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from .parkour_manager import ParkourTerm

@configclass
class ParkourTermCfg:

    class_type: type[ParkourTerm] = MISSING

    debug_vis:bool = False 
