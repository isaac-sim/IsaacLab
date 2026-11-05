# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""An ordinary Cartpole RL task with optional keyboard control during playback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
from isaaclab_teleop import CaptureTeleopSourceCfg, TeleopActionCfg, se3_keyboard_teleop_cfg

from isaaclab.utils import configclass

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv

KEYBOARD_STEP = 0.4


def keyboard_to_cart_action(actions: torch.Tensor, env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Map the keyboard's x signal to a dimensionless Cartpole effort action."""
    return (actions[..., :1] / KEYBOARD_STEP).clamp(-1.0, 1.0)


@configclass
class CartpoleInteractiveEnvCfg(CartpoleEnvCfg):
    """Keep the Cartpole task and add an optional input source for playback."""

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.actions.joint_effort, TeleopActionCfg):
            return
        # Wrap after parent initialization, which may tune the original controller.
        self.actions.joint_effort = TeleopActionCfg(
            controller=self.actions.joint_effort,
            source=CaptureTeleopSourceCfg(
                device=se3_keyboard_teleop_cfg(pos_sensitivity=KEYBOARD_STEP, gripper_term=False),
            ),
            env_ids=[0],
            action_adapter=keyboard_to_cart_action,
        )


# Retain the original environment implementation and all registered RL agent configurations.
cartpole_spec = gym.spec("Isaac-Cartpole")
gym.register(
    id="Isaac-Cartpole-Interactive",
    entry_point=cartpole_spec.entry_point,
    disable_env_checker=True,
    kwargs={
        **cartpole_spec.kwargs,
        "env_cfg_entry_point": f"{__name__}:CartpoleInteractiveEnvCfg",
    },
)
