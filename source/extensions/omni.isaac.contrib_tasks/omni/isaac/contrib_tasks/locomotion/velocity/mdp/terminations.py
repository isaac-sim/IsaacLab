"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.orbit.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv

"""
MDP terminations.
"""



"""
Root terminations.
"""


def base_height_rel(
        env: RLTaskEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the base height relative to the contact surface is below a certain threshold."""

    raise NotImplementedError("This function is not implemented yet.") 