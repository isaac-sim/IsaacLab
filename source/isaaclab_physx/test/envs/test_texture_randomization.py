# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replicator texture and color events on a cartpole scene."""

from isaaclab.test.utils import DeviceScope, launch_test_simulation, test_devices

launch_test_simulation(physics="isaacsim_physx", enable_cameras=True)

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


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=5.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Author the cart texture before Kit starts simulation.
    cart_texture_randomizer = EventTerm(
        func=mdp.randomize_visual_texture_material,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["cart"]),
            "texture_paths": [
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
            ],
            "event_name": "cart_texture_randomizer",
            "texture_rotation": (math.pi / 2, math.pi / 2),
        },
    )

    pole_texture_randomizer = EventTerm(
        func=mdp.randomize_visual_texture_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "texture_paths": [
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
            ],
            "event_name": "pole_texture_randomizer",
            "texture_rotation": (math.pi / 2, math.pi / 2),
        },
    )

    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


@configclass
class CartpoleEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    scene = CartpoleTestSceneCfg(env_spacing=2.5)

    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        self.decimation = 4
        self.sim.dt = 0.005


# Texture authoring through Replicator is device independent, so one device covers it.
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_texture_randomization(device):
    """Test texture randomization for cartpole environment."""
    sim_utils.create_new_stage()

    try:
        env_cfg = CartpoleEnvCfg()
        env_cfg.scene.num_envs = 2
        env_cfg.scene.replicate_physics = False
        env_cfg.sim.device = device

        env = ManagerBasedEnv(cfg=env_cfg)

        try:
            env.reset()
            env.step(torch.randn_like(env.action_manager.action))
            for term_name in ("cart_texture_randomizer", "pole_texture_randomizer"):
                term_cfg = env.event_manager.get_term_cfg(term_name)
                texture_paths = set(term_cfg.params["texture_paths"])
                applied = [
                    material.GetChild("Shader").GetAttribute("inputs:diffuse_texture").Get()
                    for material in term_cfg.func.material_prims
                ]
                assert len(applied) == env.num_envs
                assert all(texture is not None and texture.path in texture_paths for texture in applied), applied

            color_params = {
                "event_name": "cart_color_randomizer",
                "asset_cfg": SceneEntityCfg("robot", body_names=["cart"]),
                "colors": {"r": (0.25, 0.25), "g": (0.5, 0.5), "b": (0.75, 0.75)},
            }
            color_term = mdp.randomize_visual_color(
                EventTerm(func=mdp.randomize_visual_color, mode="reset", params=color_params), env
            )
            color_term(env, None, **color_params)
            assert color_term.material_prims
            for material in color_term.material_prims:
                color = material.GetChild("Shader").GetAttribute("inputs:diffuse_color_constant").Get()
                assert tuple(color) == pytest.approx((0.25, 0.5, 0.75))
            env.step(torch.zeros_like(env.action_manager.action))
        finally:
            env.close()
    finally:
        sim_utils.close_stage()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_texture_randomization_failure_replicate_physics(device):
    """Test texture randomization failure when replicate physics is set to True."""
    sim_utils.create_new_stage()

    try:
        cfg_failure = CartpoleEnvCfg()
        cfg_failure.scene.num_envs = 2
        cfg_failure.scene.replicate_physics = True
        cfg_failure.sim.device = device

        with pytest.raises(RuntimeError, match="Scene replication is enabled"):
            env = ManagerBasedEnv(cfg_failure)
            env.close()
    finally:
        sim_utils.close_stage()
