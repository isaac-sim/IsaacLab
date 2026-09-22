# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks Replicator texture randomization on the cart-pole scene in ``prestartup`` and ``reset`` modes."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import math

import pytest
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.test.integration_scene_cfgs import CartpoleTestSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

pytestmark = pytest.mark.integration

_WOOD_TEXTURES = [
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/{name}_BaseColor.png"
    for name in (
        "Bamboo_Planks/Bamboo_Planks",
        "Cherry/Cherry",
        "Oak/Oak",
        "Timber/Timber",
        "Timber_Cladding/Timber_Cladding",
        "Walnut_Planks/Walnut_Planks",
    )
]


def _texture_randomizer(body_name: str, mode: str) -> EventTerm:
    return EventTerm(
        func=mdp.randomize_visual_texture_material,
        mode=mode,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[body_name]),
            "texture_paths": _WOOD_TEXTURES,
            "event_name": f"{body_name}_texture_randomizer",
            "texture_rotation": (math.pi / 2, math.pi / 2),
        },
    )


def _reset_joint(
    joint_name: str, position_range: tuple[float, float], velocity_range: tuple[float, float]
) -> EventTerm:
    return EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[joint_name]),
            "position_range": position_range,
            "velocity_range": velocity_range,
        },
    )


@configclass
class ActionsCfg:
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # the prestartup mode runs on the main thread, which the Kit-side texture writes require at startup
    cart_texture_randomizer = _texture_randomizer("cart", mode="prestartup")
    pole_texture_randomizer = _texture_randomizer("pole", mode="reset")
    reset_cart_position = _reset_joint("slider_to_cart", (-1.0, 1.0), (-0.1, 0.1))
    reset_pole_position = _reset_joint(
        "cart_to_pole", (-0.125 * math.pi, 0.125 * math.pi), (-0.01 * math.pi, 0.01 * math.pi)
    )


@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    # USD-level randomization requires individually parsed assets
    scene = CartpoleTestSceneCfg(num_envs=4, env_spacing=2.5, replicate_physics=False)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_texture_randomization(device):
    """The environment builds, resets, and steps with texture randomization in both event modes."""
    sim_utils.create_new_stage()
    env_cfg = CartpoleEnvCfg()
    env_cfg.sim.device = device
    env = ManagerBasedEnv(cfg=env_cfg)
    try:
        with torch.inference_mode():
            for count in range(20):
                if count % 10 == 0:
                    env.reset()
                env.step(torch.randn_like(env.action_manager.action))
    finally:
        env.close()
        sim_utils.close_stage()


def test_texture_randomization_rejects_replicate_physics():
    sim_utils.create_new_stage()
    cfg_failure = CartpoleEnvCfg()
    cfg_failure.scene.replicate_physics = True
    try:
        with pytest.raises(RuntimeError):
            ManagerBasedEnv(cfg_failure)
    finally:
        sim_utils.close_stage()
