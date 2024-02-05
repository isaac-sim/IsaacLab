from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.orbit.utils.math as math_utils
import omni.isaac.orbit.utils.string as string_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as obs

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


"""
Root penalties.
"""


def upright_posture_bonus(
    env: RLTaskEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()