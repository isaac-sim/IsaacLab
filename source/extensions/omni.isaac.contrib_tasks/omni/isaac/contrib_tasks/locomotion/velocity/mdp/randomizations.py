"""Common functions that can be used to enable certain randomizations.

Randomization includes anythin related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`omni.isaac.orbit.managers.RandomizationTermCfg` object to enable
the randomization introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv, RLTaskEnv