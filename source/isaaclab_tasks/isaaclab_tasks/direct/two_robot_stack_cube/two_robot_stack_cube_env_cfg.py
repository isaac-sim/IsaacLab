# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.markers.config import CUBOID_MARKER_CFG, FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, FrameTransformerCfg, OffsetCfg
from isaaclab.sim import PhysxCfg, PinholeCameraCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class TwoRobotStackCubeCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 4  # 4 / (decimation * dt) = 200 steps
    # - spaces definition
    action_space = 18
    observation_space = 40
    state_space = 0

    obs_mode: str = "state"
    robot_controller: str = "joint_space"

    GOAL_RADIUS = 0.06
    DEX_CUBE_SIZE = 0.06
    DEX_CUBE_SCALE = 0.8

    # simulation
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 100,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=2.5,
        replicate_physics=True,
    )

    robot_left_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot_left")
    robot_left_cfg.spawn.activate_contact_sensors = True
    robot_left_cfg.init_state.pos = (0, 0.75, 0)
    robot_left_cfg.init_state.rot = tuple(
        quat_from_euler_xyz(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(-torch.pi / 2.0),
        ).tolist()
    )  # identity quaternion
    robot_right_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot_right")
    robot_right_cfg.spawn.activate_contact_sensors = True
    robot_right_cfg.init_state.pos = (0, -0.75, 0)
    robot_right_cfg.init_state.rot = tuple(
        quat_from_euler_xyz(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(torch.pi / 2.0),
        ).tolist()
    )

    sensors = (
        CameraCfg(
            prim_path="/World/envs/env_.*/Camera",  # Fixed camera
            # prim_path="/World/envs/env_.*/Robot/panda_hand/Camera",  # attach to the wrist link
            update_period=0.0,
            height=128,
            width=128,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=1.0,
                horizontal_aperture=36.0,  # gives ~90° HFOV with f=24
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.6, 0.0, 0.5),  # offset from center view
                rot=(
                    tuple(
                        quat_from_euler_xyz(
                            torch.tensor(0.8),
                            torch.tensor(torch.pi),
                            torch.tensor(-torch.pi / 2.0),
                        ).tolist()
                    )
                ),  # identity quaternion
                convention="ros",  # right-handed, X-forward, Z-up
            ),
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot_left/panda_leftfinger",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["/World/envs/env_.*/Cube_green"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot_left/panda_rightfinger",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["/World/envs/env_.*/Cube_green"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot_right/panda_leftfinger",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["/World/envs/env_.*/Cube_red"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot_right/panda_rightfinger",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["/World/envs/env_.*/Cube_red"],
        ),
    )

    tcp_marker_cfg = FRAME_MARKER_CFG.copy()
    tcp_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    tcp_marker_cfg.prim_path = "/Visuals/FrameTransformer"

    target_marker_cfg = CUBOID_MARKER_CFG.copy()
    target_marker_cfg.markers = {
        "cylinder": sim_utils.CylinderCfg(
            radius=GOAL_RADIUS,
            height=0.001,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
    target_marker_cfg.prim_path = "/Visuals/TargetMarker"

    tcp_left_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot_left/panda_link7",
        debug_vis=False,
        visualizer_cfg=tcp_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot_left/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1),
                ),
            ),
        ],
    )

    tcp_right_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot_right/panda_link7",
        debug_vis=False,
        visualizer_cfg=tcp_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot_right/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1),
                ),
            ),
        ],
    )

    left_finger_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot_left/panda_link7",
        debug_vis=False,
        visualizer_cfg=tcp_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot_left/panda_leftfinger",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.045),
                ),
            ),
        ],
    )

    right_finger_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot_left/panda_link7",
        debug_vis=False,
        visualizer_cfg=tcp_marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot_left/panda_rightfinger",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.045),
                ),
            ),
        ],
    )

    # Set Cube as object
    cube_green_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube_green",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.3, 0.0), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(DEX_CUBE_SCALE, DEX_CUBE_SCALE, DEX_CUBE_SCALE),
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

    # Set Cube as object
    cube_red_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube_red",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.3, 0.0), rot=(1, 0, 0, 0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(DEX_CUBE_SCALE, DEX_CUBE_SCALE, DEX_CUBE_SCALE),
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

    success_distance_threshold = 0.025

    cube_green_sample_range = [
        [
            -0.05,
            0.1,
            0.0,
            0.0,
            torch.pi / 2,
            -torch.pi,
        ],
        [
            0.05,
            0.2,
            0.0,
            0.0,
            torch.pi / 2,
            torch.pi,
        ],
    ]

    cube_red_sample_range = [
        [
            -0.05,
            -0.1,
            0.0,
            -torch.pi / 2,
            0.0,
            -torch.pi,
        ],
        [
            0.05,
            -0.2,
            0.0,
            -torch.pi / 2,
            0.0,
            torch.pi,
        ],
    ]

    target_sample_range = [
        [
            -0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            -torch.pi,
        ],
        [
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            torch.pi,
        ],
    ]


@configclass
class TwoRobotStackCubeCameraCfg(TwoRobotStackCubeCfg):

    obs_mode: str = "camera"


@configclass
class TwoRobotStackCubeTeleopCfg(TwoRobotStackCubeCfg):

    robot_controller = "task_space"

    robot_left_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot_left")
    robot_left_cfg.spawn.activate_contact_sensors = True
    robot_left_cfg.init_state.pos = (0, 0.75, 0)
    robot_left_cfg.init_state.rot = tuple(
        quat_from_euler_xyz(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(-torch.pi / 2.0),
        ).tolist()
    )  # identity quaternion
    robot_right_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot_right")
    robot_right_cfg.spawn.activate_contact_sensors = True
    robot_right_cfg.init_state.pos = (0, -0.75, 0)
    robot_right_cfg.init_state.rot = tuple(
        quat_from_euler_xyz(
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(torch.pi / 2.0),
        ).tolist()
    )

    viewer = ViewerCfg()
    viewer.eye = (-1.6, 0.0, 1.2)
    viewer.lookat = (0.0, 0.0, 0.4)
    viewer.origin_type = "env"
    viewer.env_index = 0
