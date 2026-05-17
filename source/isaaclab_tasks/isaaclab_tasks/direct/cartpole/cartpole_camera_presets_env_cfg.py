# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_newton.physics import NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


@configclass
class PhysicsCfg(PresetCfg):
    physx = PhysxCfg()
    newton_mjwarp = NewtonCfg()
    ovphysx = OvPhysxCfg()
    default = physx


@configclass
class CartpoleCameraDataTypesCfg(PresetCfg):
    """Camera data-type selector for the cartpole tiled camera.

    Every variant of the cartpole camera task uses the same camera pose,
    intrinsics, and resolution; only the ``data_types`` list changes. Keeping
    just that list in a small PresetCfg lets the parent
    :class:`CartpoleTiledCameraCfg` share a single instance of everything else.
    """

    rgb = ["rgb"]
    albedo = ["albedo"]
    semantic_segmentation = ["semantic_segmentation"]
    simple_shading_constant_diffuse = ["simple_shading_constant_diffuse"]
    simple_shading_diffuse_mdl = ["simple_shading_diffuse_mdl"]
    simple_shading_full_mdl = ["simple_shading_full_mdl"]
    depth = ["depth"]
    default = rgb


@configclass
class CartpoleTiledCameraCfg(CameraCfg):
    """Single instance of the cartpole tiled camera with selectable data types."""

    prim_path: str = "/World/envs/env_.*/Camera"
    offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(
        pos=(-5.0, 0.0, 2.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"
    )
    data_types: CartpoleCameraDataTypesCfg = CartpoleCameraDataTypesCfg()
    spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    )
    width: int = 100
    height: int = 100
    renderer_cfg: MultiBackendRendererCfg = MultiBackendRendererCfg()


@configclass
class CartpoleObservationSpaceCfg(PresetCfg):
    """Per-data-type policy observation shape selector.

    ``observation_space`` is the only env-level field that varies per data-type
    variant; keeping it in its own nested PresetCfg lets the env cfg share
    every other field (sim, robot, scene, rewards, ...) as a single instance
    instead of duplicating the full env cfg per variant.
    """

    rgb = [100, 100, 3]
    albedo = rgb
    simple_shading_constant_diffuse = rgb
    simple_shading_diffuse_mdl = rgb
    simple_shading_full_mdl = rgb
    depth = [100, 100, 1]
    semantic_segmentation = [100, 100, 4]
    default = rgb


@configclass
class CartpoleCameraPresetsEnvCfg(DirectRLEnvCfg):
    """Cartpole camera env cfg with selectable data-type variant.

    Shared fields (sim, robot, scene, rewards, ...) are declared once. The
    fields that vary per ``presets=<name>`` -- the policy observation shape
    and the camera data types -- are nested
    :class:`~isaaclab_tasks.utils.PresetCfg` selectors that the framework
    resolver pins at ``gym.make`` time.
    """

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

    # camera (data_types selected via ``presets=<name>``)
    tiled_camera: CartpoleTiledCameraCfg = CartpoleTiledCameraCfg()
    write_image_to_file = False

    frame_stack: int = -1
    """Number of frames to stack along the channel dim.

    ``-1`` (default) auto-resolves to ``2`` for the Newton + Warp combo and ``1`` otherwise.
    Set to ``1`` to force single-frame; set to ``N > 1`` to force an explicit stack size.
    """

    # spaces (observation shape varies per variant via the nested PresetCfg)
    action_space = 1
    state_space = 0
    observation_space: CartpoleObservationSpaceCfg = CartpoleObservationSpaceCfg()

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
