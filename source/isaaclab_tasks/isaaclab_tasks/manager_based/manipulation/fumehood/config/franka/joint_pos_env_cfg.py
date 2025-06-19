# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.fumehood import mdp
from isaaclab_tasks.manager_based.manipulation.fumehood.lift_env_cfg import FumehoodEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
import random as rd


@configclass
class FrankaFumehoodEnvCfg(FumehoodEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.objects = ["glass_sample_vial","beaker_500ml","glass_conical_flask", "vial_rack"]
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
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object

        #what if its now a vial rack  ?

        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.02],rot=[1, 0, 0, 0]) ,#rot=[0, 0, 1, 0])
        #     spawn=UsdFileCfg(
        #         usd_path=f"/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/vial_rack.usd",
        #         #usd_path=f"/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/test_cube.usd",
        #         #usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        #         #usd_path = f"/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/glass_conical.usd",
        #         #usd_path = f"/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/glass_sample_vial.usd",
        #         scale=(0.8, 0.8, 2.0),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #     ),
        # )

        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

       
        self.scene.object = self.randomiseObject(self.objects)

        # self.scene.flask = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/flask",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.7, 0.2, 0.05],rot=[1, 0, 0, 0]),
        #     spawn=UsdFileCfg(
        #         usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/glass_conical_flask.usd",
        #         scale=(1, 1, 1),
        #         rigid_props=cube_properties,
        #         semantic_tags=[("class", "flask")],
        #     ),
        # ) 

        # self.scene.vial = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/vial",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.7, 0.3, 0.05],rot=[0, 0, 1, 0]),
        #     spawn=UsdFileCfg(
        #         usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/glass_sample_vial.usd",
        #         scale=(1, 1, 1),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #         semantic_tags=[("class", "vial")],
        #     ),
        # ) 
        # self.scene.beaker = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/beaker",
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.7, 0.4, 0.05],rot=[0, 0, 1, 0]),
        #     spawn=UsdFileCfg(
        #         usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/beaker_500ml.usd",
        #         scale=(0.5, 0.5, 0.5),
        #         rigid_props=RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=16,
        #             solver_velocity_iteration_count=1,
        #             max_angular_velocity=1000.0,
        #             max_linear_velocity=1000.0,
        #             max_depenetration_velocity=5.0,
        #             disable_gravity=False,
        #         ),
        #         semantic_tags=[("class", "beaker")],
        #     ),
        # ) 

        self.scene.object_collection = RigidObjectCollectionCfg(
            rigid_objects={
                "beaker": RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/beaker",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.7, 0.4, 0.05],rot=[0, 0, 1, 0]),
                    spawn=UsdFileCfg(
                        usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/beaker_500ml.usd",
                        scale=(0.5, 0.5, 0.5),
                        rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=16,
                            solver_velocity_iteration_count=1,
                            max_angular_velocity=1000.0,
                            max_linear_velocity=1000.0,
                            max_depenetration_velocity=5.0,
                            disable_gravity=False,
                        ),
                    semantic_tags=[("class", "beaker")],
                    ),
                ),
                "vial": RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/vial",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.7, 0.3, 0.05],rot=[0, 0, 1, 0]),
                    spawn=UsdFileCfg(
                        usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/glass_sample_vial.usd",
                        scale=(1, 1, 1),
                        rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=16,
                            solver_velocity_iteration_count=1,
                            max_angular_velocity=1000.0,
                            max_linear_velocity=1000.0,
                            max_depenetration_velocity=5.0,
                            disable_gravity=False,
                        ),
                        semantic_tags=[("class", "vial")],
                    ),
                ),
                "flask" : RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/flask",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.7, 0.2, 0.05],rot=[1, 0, 0, 0]),
                    spawn=UsdFileCfg(
                        usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/glass_conical_flask.usd",
                        scale=(1, 1, 1),
                        rigid_props=cube_properties,
                        semantic_tags=[("class", "flask")],
                    ),
                ) ,
                "stirplate" : RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/stirplate",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.7, 0.2, 0.05],rot=[1, 0, 0, 0]),
                    spawn=UsdFileCfg(
                        usd_path="/workspace/isaaclab/source/isaaclab_assets/data/Props/lab_equipment/mag_hotplate.usd",
                        scale=(1, 1, 1),
                        rigid_props=cube_properties,
                        semantic_tags=[("class", "flask")],
                    ),
                ) 

            }
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
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
    
    def randomiseObject(self, objects:list[str]) -> RigidObjectCfg:
        # spawn random number
        n = rd.randint(0,len(objects)-1)
        #some objects import upside down
        if n<2:
            rot = [0, 0, 1, 0]
        else :
            rot=[1, 0, 0, 0]
        print(f"Random spawn object : {objects[n]}")
        object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=rot),
            spawn=UsdFileCfg(
                usd_path=f"/workspace/isaaclab/source/isaaclab_assets/data/Props/glassware/{objects[n]}.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                semantic_tags=[("class", "object")],
            ),
        )
        return object


@configclass
class FrankaCubeEnvCfg_PLAY(FrankaFumehoodEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
