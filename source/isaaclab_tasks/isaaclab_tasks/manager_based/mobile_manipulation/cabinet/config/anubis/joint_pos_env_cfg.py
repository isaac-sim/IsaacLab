# Environment configuration for the Anubis robot in the Cabinet task for RL training.

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.mm.cabinet import mdp

from isaaclab_tasks.manager_based.mm.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
    CabinetEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.anubis import ANUBIS_CFG  # isort:skip

@configclass
class AnubisCabinetEnvCfg(CabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set franka as robot
        self.scene.robot = ANUBIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



        # Set Actions for the specific robot type (franka)
        self.actions.armR_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["link1.*", "arm1.*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.armL_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["link2.*","arm2.*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.gripperR_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper1.*"],
            open_command_expr={"gripper1.*": 0.04},
            close_command_expr={"gripper1.*": 0.0},
        )

        self.actions.gripperL_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper2.*"],
            open_command_expr={"gripper2.*": 0.04},
            close_command_expr={"gripper2.*": 0.0},
        )

        self.actions.base_action = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["dummy_base_.*"],
        )

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_R_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_footprint",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/RightEndEffectorFrameTransformer_R"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link1",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper1L",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper1R",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
        self.scene.ee_L_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_footprint",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LeftEndEffectorFrameTransformer_L"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link2",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper2L",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper2R",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )

        # # override rewards
        # self.rewards.approach_gripper_handle.params["offset"] = 0.04
        # self.rewards.grasp_handle.params["open_joint_pos"] = 0.04
        # self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["panda_finger_.*"]


@configclass
class AnubisCabinetEnvCfg_PLAY(AnubisCabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
~                                                                                                                                                                                                                                                                                             
~                                                                                                                                                                                                                                                                                             
~                                                                                                                                                                                                                                                                                             
~                                                                                                                                                                                                                                                                                             
~                                                                                                                                                                                                                                                                                             
~                                                                                                                                                                                                                                                                                             
~                                                                       