# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import torch

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PhysxCfg, PinholeCameraCfg, RigidBodyPropertiesCfg, SimulationCfg, UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip




@configclass
class AssemblyKitEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 4  # 4 / (decimation * dt) = 200 steps
    # - spaces definition
    action_space = 9
    observation_space = 20
    state_space = 0

    obs_mode: str = "state"
    table_offset = 0.55

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
        replicate_physics=False,  # TODO set to True for optimization
    )

    # table_cfg: AssetBaseCfg = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Table",
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(table_offset, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)
    #     ),
    # )

    robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.init_state.joint_pos = {
        "panda_joint1": 00.0,
        "panda_joint2": 0.6,
        "panda_joint3": 0.0,
        "panda_joint4": -2.2,
        "panda_joint5": 0.0,
        "panda_joint6": 3.037,
        "panda_joint7": 0.741,
        "panda_finger_joint.*": 0.04,
    }

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
                pos=(table_offset + 0.3, 0.0, 0.4),  # offset from center view
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
    )


@configclass
class AssemblyKitCameraEnvCfg(AssemblyKitEnvCfg):

    obs_mode: str = "camera"
