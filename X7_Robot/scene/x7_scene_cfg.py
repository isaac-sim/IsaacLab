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
from robot.x7_duo_cfg import X7_DUO_CFG
from isaaclab.sensors import CameraCfg


@configclass
class X7SceneCfg(InteractiveSceneCfg):
    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # 机器人
    robot = X7_DUO_CFG.replace( # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    #灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # sensors
    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/head_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.4, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.4, 0.1, 0.0), rot=(0.67409, 0.21355, 0.67409, -0.213555), convention="opengl"),
    )

    left_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Link7/left_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.14756, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.3), rot=(0.0, -0.0, -1.0, -0.0), convention="opengl"),
    )

    right_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Link14/right_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.4, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.3), rot=(0.0, -0.0, -1.0, -0.0), convention="opengl"),
    )
    
