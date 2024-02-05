"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.orbit.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import Camera, ContactSensor, RayCaster

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv, RLTaskEnv

"""
Root state.
"""


def base_up_proj(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg) -> torch.Tensor:
    """Projection of the base up vector onto the world up vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute base up vector
    base_up_vec = math_utils.quat_rotate(asset.data.root_quat_w, -asset.GRAVITY_VEC_W)

    return base_up_vec[:, 2].unsqueeze(-1)


"""
Sensors
"""


def contact_states(env: BaseEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg) -> torch.Tensor:
    """Contact states of the asset, where 1 indicates contact and 0 indicates no contact."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    contact_states = torch.where(
        net_contact_forces > 0, torch.ones_like(net_contact_forces), torch.zeros_like(net_contact_forces)
    )
    return contact_states