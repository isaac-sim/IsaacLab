# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms
from .observations import rel_nut_bolt_bottom_distance, rel_nut_bolt_tip_distance
from .rewards import l2_norm

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def nut_fully_screwed(
    env: ManagerBasedRLEnv,
    threshold: float = 1e-3,
) -> torch.Tensor:
    """_summary_

    Args:
        env (ManagerBasedRLEnv): _description_
        threshold (float, optional): _description_. Defaults to 1e-3.

    Returns:
        torch.Tensor: _description_
    """
    
    diff = rel_nut_bolt_bottom_distance(env)
    dis = l2_norm(diff)
    return dis < threshold

def nut_successfully_threaded(
    env: ManagerBasedRLEnv,
    threshold: float = 1e-3,
) -> torch.Tensor:
    """_summary_

    Args:
        env (ManagerBasedRLEnv): _description_
        threshold (float, optional): _description_. Defaults to 1e-3.

    Returns:
        torch.Tensor: _description_
    """
    
    diff = rel_nut_bolt_tip_distance(env)
    dis = l2_norm(diff)
    return dis < threshold
