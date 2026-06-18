# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import TerminationTermCfg 

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg

## Pre-defined configs
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG

@configclass
class EventCfg:
    """Configuration for events."""
    init_openarm_arm_pose = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

@configclass
class OpenarmCubeStackEnvCfg(StackEnvCfg):
    """Configuration for the OpenArm Cube Stack Environment."""

    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()

        # 1. 機器人與場景配置
        self.scene.robot = OPENARM_BI_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.robot.init_state.joint_pos = {
            # 左手：給它一個自然的預設姿勢（不要是全 0）
            "openarm_left_joint1": 0.0,
            "openarm_left_joint2": 0.0,
            "openarm_left_joint3": 0.0,
            "openarm_left_joint4": 0.0,
            "openarm_left_joint5": 0.0,
            "openarm_left_joint6": 0.0,
            "openarm_left_joint7": 0.0,
            "openarm_right_joint.*": 0.0, 
            "openarm_left_finger_joint.*": 0.0,
            "openarm_right_finger_joint.*": 0.0,
        }

        self.scene.robot.actuators["openarm_arm"].stiffness = 150.0 
        self.scene.robot.actuators["openarm_arm"].damping = 50.0
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.table.spawn.semantic_tags = [("class", "table")]
        self.scene.plane.semantic_tags = [("class", "ground")]

        # 2. 動作配置
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["openarm_left_joint[1-7]"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_left_finger_joint.*"],
            open_command_expr={"openarm_left_finger_joint.*": 0.044},
            close_command_expr={"openarm_left_finger_joint.*": 0.0},
        )
        
        # 3. 方塊配置
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, max_depenetration_velocity=5.0, disable_gravity=False,
        )
        for i, (name, pos, color) in enumerate([
            ("cube_1", [0.2, 0.08, 0.0203], "blue"),
            ("cube_2", [0.55, 0.05, 0.0203], "red"),
            ("cube_3", [0.60, -0.1, 0.0203], "green")
        ]):
            setattr(self.scene, name, RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name.capitalize()}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=[1, 0, 0, 0]),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/{color}_block.usd",
                    rigid_props=cube_properties,
                    semantic_tags=[("class", name)],
                ),
            ))

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_left_link1", 
            debug_vis=False,
            target_frames=[],
        )

        from isaaclab.managers import TerminationTermCfg
        from isaaclab.utils import configclass 

        @configclass
        class Terminations:
            time_out = TerminationTermCfg(func=mdp.time_out)

        self.terminations = Terminations()
        
        # 關掉不必要的觀察值與獎勵
        self.rewards = None
        if self.observations.policy is not None:
            # 不要一個一個 delattr，直接給它一個空的配置類
            from isaaclab.managers import ObservationGroupCfg
            self.observations.policy = ObservationGroupCfg() 
            
        # 確保子任務觀察值也是空的
        self.observations.subtask_terms = None