# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab_tasks.manager_based.manipulation.cube_lift import mdp
from isaaclab_tasks.manager_based.manipulation.cube_lift.lift_env_cfg import CubeEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaDevEnvCfg(CubeEnvCfg):
    def __post_init__(self):
        # post init of parent
       
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"


        cube_properties = RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        )

        # Set each stacking cube deterministically
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/vial_rack.usd",
                scale=(0.75, 0.75, 1.5),
                rigid_props=cube_properties,
                semantic_tags=[("class", "object")],
            ),
        )

        self.scene.flask = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/flask",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.65, 0.4, 0.05],rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/conical_flask.usd",
                scale=(1, 1, 1),
                rigid_props=cube_properties,
                visible=True,
                copy_from_source = False,
                semantic_tags=[("class", "flask")],
            ),
        ) 

        self.scene.vial = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/vial",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.65, 0.3, 0.05],rot=[0, 0, 1, 0]),
            spawn=UsdFileCfg(
                usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/sample_vial_20ml.usd",
                scale=(1, 1, 1),
                rigid_props=cube_properties,
                visible=True,
                copy_from_source = False,
                semantic_tags=[("class", "vial")],
            ),
        ) 
        self.scene.beaker = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/beaker",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.65, 0.2, 0.05],rot=[0, 0, 1, 0]),
            spawn=UsdFileCfg(
                usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/beaker_500ml.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=cube_properties,
                semantic_tags=[("class", "beaker")],
            ),
        ) 
        self.scene.stirplate = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/stirplate",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40, -0.3, 0.05],rot=[0.707, 0, 0, -0.707]),
            spawn=UsdFileCfg(
                usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/lab_equipment/mag_hotplate.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=cube_properties,
                semantic_tags=[("class", "stirplate")],
            ),
        ) 

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
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
class FrankaDevEnvCfg_PLAY(FrankaDevEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
