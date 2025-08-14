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
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, FrameTransformerCfg, OffsetCfg
from isaaclab.sim import PhysxCfg, PinholeCameraCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class PegInsertionSideEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 4  # 4 / (decimation * dt) = 200 steps
    # - spaces definition
    action_space = 9
    observation_space = 43
    state_space = 0

    obs_mode: str = "state"
    robot_controller: str = "joint_space"

    TABLE_OFFSET = 0.55

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
        env_spacing=2.0,
        replicate_physics=False,
    )

    robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.activate_contact_sensors = True

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
                pos=(TABLE_OFFSET + 0.3, 0.0, 0.4),  # offset from center view
                rot=(
                    tuple(
                        quat_from_euler_xyz(
                            torch.tensor(0.6),
                            torch.tensor(torch.pi),
                            torch.tensor(-torch.pi / 2.0),
                        ).tolist()
                    )
                ),  # identity quaternion
                convention="ros",  # right-handed, X-forward, Z-up
            ),
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["/World/envs/env_.*/Peg"],  # only peg contacts
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=["/World/envs/env_.*/Peg"],
        ),
    )
    # Listens to the required transforms
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    tcp_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_link7",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1),
                ),
            ),
        ],
    )

    peg_sample_range = [
        [0.4, -0.3, 0.0, 0.0, 0.0, 1.0 * torch.pi / 6.0],
        [0.6, 0.0, 0.0, 0.0, 0.0, 5.0 * torch.pi / 6.0],
    ]

    box_sample_range = [
        [0.45, 0.2, 0.0, 0.0, 0.0, 3.0 * torch.pi / 8.0],
        [0.5, 0.4, 0.0, 0.0, 0.0, 5.0 * torch.pi / 8.0],
    ]

    asset_dir = "/home/johann/Downloads/peg_insertion_side"

    def get_multi_cfg(self, usd_paths: list[str], prim_path: str, kinematic_enabled) -> RigidObjectCfg:
        cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path=prim_path,
            spawn=sim_utils.MultiUsdFileCfg(
                usd_path=usd_paths,
                random_choice=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=kinematic_enabled,
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0025,
                    max_depenetration_velocity=1000.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.27807487, 0.20855615, 0.16934046),
                    emissive_color=(0.0, 0.0, 0.0),
                    roughness=0.5,
                    metallic=0.0,
                    opacity=1.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.05), rot=(0.0, 0.0, 0.0, 1.0)),
        )

        return cfg


@configclass
class PegInsertionSideCameraEnvCfg(PegInsertionSideEnvCfg):

    obs_mode: str = "camera"


@configclass
class PegInsertionSideTeleopCfg(PegInsertionSideEnvCfg):

    robot_controller = "task_space"

    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.activate_contact_sensors = True

    viewer = ViewerCfg()
    viewer.eye = (1.6, 0.0, 1.2)
    viewer.lookat = (0.0, 0.0, 0.2)
    viewer.origin_type = "env"
    viewer.env_index = 0
