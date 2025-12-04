import math
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils import configclass
from robot.a10_cfg import A10_CFG
from isaaclab.sensors import CameraCfg


@configclass
class A10SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot = A10_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # 注意：下面的 prim_path 需要与你的 URDF 中相机挂载的 link 对齐。
    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/head_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.4, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.3), rot=(0.98481, 0.17365, 0.0, 0.0), convention="opengl"),
    )

    right_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Arm2_ee/right_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.14756, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, -1.0, 0.0, 0.0), convention="opengl"),
    )

    left_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Arm1_ee/left_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.4, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, -1.0, 0.0, 0.0), convention="opengl"),
    )

