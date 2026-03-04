# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


@configclass
class PhysicsCfg(PresetCfg):
    default = PhysxCfg()
    physx = PhysxCfg()
    newton = NewtonCfg()


@configclass
class MultiBackendRendererCfg(PresetCfg):
    default: IsaacRtxRendererCfg = IsaacRtxRendererCfg()
    newton_renderer: NewtonWarpRendererCfg = NewtonWarpRendererCfg()
    isaac_sim_rtx: IsaacRtxRendererCfg = default


@configclass
class MultiDataTypeCartpoleTiledCameraCfg(PresetCfg):
    @configclass
    class CartpoleTiledCameraCfg(TiledCameraCfg):
        prim_path: str = "/World/envs/env_.*/Camera"
        offset: TiledCameraCfg.OffsetCfg = TiledCameraCfg.OffsetCfg(
            pos=(-5.0, 0.0, 2.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"
        )
        data_types: list[str] = []
        spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        )
        width: int = 100
        height: int = 100
        renderer_cfg: MultiBackendRendererCfg = MultiBackendRendererCfg()

    default = CartpoleTiledCameraCfg(data_types=["rgb"])
    depth = CartpoleTiledCameraCfg(data_types=["depth"])
    albedo = CartpoleTiledCameraCfg(data_types=["albedo"])
    simple_shading_constant_diffuse = CartpoleTiledCameraCfg(data_types=["simple_shading_constant_diffuse"])
    simple_shading_diffuse_mdl = CartpoleTiledCameraCfg(data_types=["simple_shading_diffuse_mdl"])
    simple_shading_full_mdl = CartpoleTiledCameraCfg(data_types=["simple_shading_full_mdl"])
    rgb = default


@configclass
class CartpoleCameraPresetsEnvCfg(PresetCfg):
    @configclass
    class BaseCartpoleCameraEnvCfg(DirectRLEnvCfg):
        # env
        decimation = 2
        episode_length_s = 5.0
        action_scale = 100.0  # [N]

        # simulation
        sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=PhysicsCfg())

        # robot
        robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        cart_dof_name = "slider_to_cart"
        pole_dof_name = "cart_to_pole"

        # camera
        tiled_camera: MultiDataTypeCartpoleTiledCameraCfg = MultiDataTypeCartpoleTiledCameraCfg()
        write_image_to_file = False

        # spaces
        action_space = 1
        state_space = 0
        observation_space = [100, 100, 3]

        # change viewer settings
        viewer = ViewerCfg(eye=(20.0, 20.0, 20.0))

        # scene
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=20.0, replicate_physics=True)

        # reset
        max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
        initial_pole_angle_range = [-0.125, 0.125]  # the range in which the pole angle is sampled from on reset [rad]

        # reward scales
        rew_scale_alive = 1.0
        rew_scale_terminated = -2.0
        rew_scale_pole_pos = -1.0
        rew_scale_cart_vel = -0.01
        rew_scale_pole_vel = -0.005

    default = BaseCartpoleCameraEnvCfg()
    depth = BaseCartpoleCameraEnvCfg(observation_space=[100, 100, 1])
    albedo = BaseCartpoleCameraEnvCfg(observation_space=[100, 100, 3])
    simple_shading_constant_diffuse = BaseCartpoleCameraEnvCfg(observation_space=[100, 100, 3])
    simple_shading_diffuse_mdl = BaseCartpoleCameraEnvCfg(observation_space=[100, 100, 3])
    simple_shading_full_mdl = BaseCartpoleCameraEnvCfg(observation_space=[100, 100, 3])
