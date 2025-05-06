from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity_rma.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_assets.robots.unitree import *  # isort: skip


@configclass
class Go2RoughEnvCfg_rnn(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        }

        # self.scene.robot.actuators = {
        #     "legs": DelayedPDActuatorCfg(
        #         joint_names_expr=[".*"],
        #         effort_limit=23.5,
        #         velocity_limit=30.0,
        #         stiffness=25, 
        #         damping=0.5, 
        #         min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
        #         max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        #     )
        # }
        
        self.scene.terrain.terrain_generator.sub_terrains["flat"].proportion = 0.2
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.2
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.2
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.2
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.2

        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Observations------------------------------
        
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        self.observations.policy.actions = None

        # ------------------------------Actions------------------------------

        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------

        self.events.randomize_actuator_gains = None

        # ------------------------------Rewards------------------------------

        # ------------------------------Terminations------------------------------

        # self.terminations.base_contact = None

        # ------------------------------Commands------------------------------

        if self.__class__.__name__ == "Go2RoughEnvCfg_rnn":
            self.disable_zero_weight_rewards()
        
