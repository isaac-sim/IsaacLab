# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.reach_cube import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.reach_cube.reach_env_cfg_rgb_rsl_rl import ReachEnvCfg

##
# Pre-defined configs
##
# from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG  # isort: skip

# sensors
import numpy as np
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from omni.isaac.lab.sensors import CameraCfg, TiledCameraCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.math import quat_from_euler_xyz

"""
cameras: Dict[str, CameraCfg] = {
    "front_camera": CameraCfg(pos= [1.2, 0, 1.5], quat= quat),
    "wrist_camera": CameraCfg(pos=[0.04339012, -0.03256147, 0.04373], quat=[0.0, 1.0, 0.0, 0.0], attached_prim="Robot/franka/virtual_eef_link")
}
"""

def rads2degrees(rads):
    return rads * 180.0 / np.pi

def degrees2rads(degrees):
    return degrees * np.pi / 180.0

def top_down_in_world():
    """Slightly rotated top-down position camera"""
    coord_sys = "world"
    position = [1, 0, 1]
    roll, pitch, yaw = torch.tensor([-180]), torch.tensor([90]), torch.tensor([0])
    rot_quat = quat_from_euler_xyz(roll=roll, pitch=pitch, yaw=yaw).numpy() + 0.0
    rot_quat = np.around(rot_quat, decimals=4).flatten()
    rot_quat = (rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])

    return position, rot_quat, coord_sys

def sideview_in_world_ccw():
    """Counterclockwise sideview position camera relative to being behind the robot"""
    coord_sys = "world"
    position = (0.5, -2.3, 0.5) #-1.3
    pitch, yaw, roll = \
        torch.tensor([degrees2rads(0)]), \
        torch.tensor([degrees2rads(90)]), \
        torch.tensor([degrees2rads(0)])
    rot_quat = quat_from_euler_xyz(pitch=pitch, yaw=yaw, roll=roll).numpy() + 0.0
    rot_quat = np.around(rot_quat, decimals=4).flatten()
    rot_quat = (rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])

    return position, rot_quat, coord_sys

def diagonal_view_in_world_ccw():
    coord_sys = "world"
    position = (1.5, -1.5, 0.3)
    pitch, yaw, roll = \
        torch.tensor([degrees2rads(0)]), \
        torch.tensor([degrees2rads(130)]), \
        torch.tensor([degrees2rads(0)])
    rot_quat = quat_from_euler_xyz(pitch=pitch, yaw=yaw, roll=roll).numpy() + 0.0
    rot_quat = np.around(rot_quat, decimals=4).flatten()
    rot_quat = (rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])

    return position, rot_quat, coord_sys


def diagonal_looking_down_view_in_world_ccw():
    coord_sys = "world"
    position = (1.6, -1.3, 0.9)
    pitch, yaw, roll = \
        torch.tensor([degrees2rads(21)]), \
        torch.tensor([degrees2rads(130)]), \
        torch.tensor([degrees2rads(0)])
    rot_quat = quat_from_euler_xyz(pitch=pitch, yaw=yaw, roll=roll).numpy() + 0.0
    rot_quat = np.around(rot_quat, decimals=4).flatten()
    rot_quat = (rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])

    return position, rot_quat, coord_sys

def compute_camera_intrinsics_matrix(image_width, image_heigth, focal_length, horizontal_aperture, device):
    """
        Due to limitations of Omniverse camera, we need to assume that the camera is a spherical lens, 
        i.e. has square pixels, and the optical center is centered at the camera eye.
    """
    horizontal_fov = np.degrees(2 * np.arctan2(horizontal_aperture, (2.0 * focal_length)))
    vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
    horizontal_fov = degrees2rads(horizontal_fov)

    f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
    f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)

    K = torch.tensor([
        [f_x, 0.0, image_width / 2.0], 
        [0.0, f_y, image_heigth / 2.0], 
        [0.0, 0.0, 1.0]], 
    device=device, dtype=torch.float)
    return K


@configclass
class FrankaCubeReachEnvCfg_rsl_rl(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # sensors
        # NOTE: {ENV_REGEX_NS} = /World/envs/env_.*/
        print(f"Creating camera sensor (RGB)")
        
        #position, rotation, coord_frame = top_down_in_world()
        #position, rotation, coord_frame = sideview_in_world_ccw()
        #position, rotation, coord_frame = diagonal_view_in_world_ccw()
        position, rotation, coord_frame = diagonal_looking_down_view_in_world_ccw()

        RESOLUTION = (480, 640) #(480, 640)
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.1,
            height=RESOLUTION[0],
            width=RESOLUTION[1],
            data_types=["rgb", "distance_to_image_plane"],         # rgb-d
            #data_types=["rgb", "semantic_segmentation"],          # rgb-seg
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, 
                focus_distance=400.0, 
                horizontal_aperture=20.955, 
                clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=position, rot=rotation, convention=coord_frame),
            #colorize_semantic_segmentation=False, # True: uint8 (4 channels, RGBA), False: uint32 (1 channel)
            #colorize_instance_id_segmentation=False,
        )

        # self.scene.K = compute_camera_intrinsics_matrix(
        #     image_heigth=RESOLUTION[0], 
        #     image_width=RESOLUTION[1], 
        #     focal_length=self.scene.camera.spawn.focal_length,
        #     horizontal_aperture=self.scene.camera.spawn.horizontal_aperture,
        #     device=DEVICE)
        # print(f"Camera intrinsics (K): {self.scene.K}", end="\n")

        # Listens to the required transforms
        # marker_cfg = FRAME_MARKER_CFG.copy()
        # marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        # marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            #visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaCubeReachEnvCfg_rsl_rl_PLAY(FrankaCubeReachEnvCfg_rsl_rl):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False