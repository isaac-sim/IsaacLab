# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.kuka import KUKA_VICTOR_CFG  # isort: skip


@configclass
class KukaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Kuka as robot
        self.scene.robot = KUKA_VICTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        scene_offsets = [0.3, 0.2, 0.8]
        
        self.scene.table.init_state.pos = [0.5 + scene_offsets[0], scene_offsets[1], scene_offsets[2]]
        self.scene.plane.init_state.pos = [0., 0, 0.0]
        self.commands.object_pose.ranges.pos_z = [0.25 + scene_offsets[2], 0.75 + scene_offsets[2]]

        # Set actions for the specific robot type (Kuka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["victor_left_arm_joint.*"], scale=0.5, use_default_offset=True
        )
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_finger.*"],
        #     open_command_expr={"panda_finger_.*": 0.04},
        #     close_command_expr={"panda_finger_.*": 0.0},
        # )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "victor_left_tool0"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5+scene_offsets[0], scene_offsets[1], 0.055+scene_offsets[2]], rot=[1, 0, 0, 0]),
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
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.object.spawn.activate_contact_sensors = True
        # self.scene.table.spawn.activate_contact_sensors = True
        self.scene.object.spawn.mass_props = MassPropertiesCfg(mass=1)
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/victor_root",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/victor_left_tool0",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )
        self.scene.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object", 
            # filter_prim_paths_expr= ["{ENV_REGEX_NS}/Table"], 
            # filter_prim_paths_expr= ["{ENV_REGEX_NS}/Robot/.*finger"],
            filter_prim_paths_expr= ["{ENV_REGEX_NS}/Table", 
                                     "{ENV_REGEX_NS}/Robot/victor_left_finger_b_link_0", 
                                     "{ENV_REGEX_NS}/Robot/victor_left_finger_b_link_0"], 
            track_air_time=True, track_pose=True,
            update_period=0.0, debug_vis=True
        )


@configclass
class KukaCubeLiftEnvCfg_PLAY(KukaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
