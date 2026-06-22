# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

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

        # Arm: stiffness high enough to hold pose under load
        self.scene.robot.actuators["openarm_arm"].stiffness = 400.0
        self.scene.robot.actuators["openarm_arm"].damping = 80.0
        # Gripper: compliant gains + effort cap so the finger-cube contact force
        # stays small and does not generate a wrist-twisting reaction torque.
        # effort_limit_sim=8 N caps grasp force; still enough to hold the cube.
        self.scene.robot.actuators["openarm_gripper"].stiffness = 200.0
        self.scene.robot.actuators["openarm_gripper"].damping = 20.0
        self.scene.robot.actuators["openarm_gripper"].effort_limit_sim = 8.0
        # Lower the robot body's PhysX depenetration velocity so that contact
        # corrections between fingers and cube are gentle impulses, not spikes.
        self.scene.robot.spawn.rigid_props.max_depenetration_velocity = 1.0  # type: ignore[union-attr]
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
        # ── Workspace pad ────────────────────────────────────────────────────
        # A 10 cm tall platform that raises the cubes to a height the arms can
        # comfortably grasp.  Top surface at Z = 0.10 m.
        # Covers x: [0.075, 0.725], y: [-0.275, 0.275] — fits all randomised
        # cube positions used in reach_red_cube_env_cfg.
        # To hide the pad: comment out self.scene.workspace_pad below.
        # To change height: edit size[2] and pad pos[2]=size[2]/2,
        #   then update CUBE_Z = size[2] + 0.02 (half the 4 cm block).
        PAD_HEIGHT = 0.13
        CUBE_Z = PAD_HEIGHT + 0.02   # pad top + half-block height
        self.scene.workspace_pad = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/workspace_pad",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, PAD_HEIGHT / 2)),
            spawn=sim_utils.CuboidCfg(
                size=(0.65, 0.55, PAD_HEIGHT),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.55, 0.55, 0.55), roughness=0.6
                ),
            ),
        )

        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, max_depenetration_velocity=1.0, disable_gravity=False,
        )
        for i, (name, pos, color) in enumerate([
            ("cube_1", [0.2,  0.08,  CUBE_Z], "blue"),
            ("cube_2", [0.55, 0.05,  CUBE_Z], "red"),
            ("cube_3", [0.60, -0.10, CUBE_Z], "green")
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

        @configclass
        class Terminations:
            time_out = TerminationTermCfg(func=mdp.time_out)

        self.terminations = Terminations()
        
        # 關掉不必要的觀察值與獎勵
        self.rewards = None
        if self.observations.policy is not None:
            from isaaclab.managers import ObservationGroupCfg
            empty_policy = ObservationGroupCfg()
            empty_policy.concatenate_terms = False
            self.observations.policy = empty_policy
            
        # 確保子任務觀察值也是空的
        self.observations.subtask_terms = None