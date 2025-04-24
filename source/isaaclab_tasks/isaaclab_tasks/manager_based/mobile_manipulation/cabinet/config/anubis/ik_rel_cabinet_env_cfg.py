# Environment configuration for the Anubis robot in the Cabinet task for teleoperation.
  
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.anubis import ANUBIS_PD_CFG  # isort:skip


@configclass
class AnubisCabinetEnvCfg(joint_pos_env_cfg.AnubisCabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = ANUBIS_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.armR_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["link1.*", "arm1.*"],
            body_name="ee_link1",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.actions.armL_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["link2.*", "arm2.*"],
            body_name="ee_link2",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # self.


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