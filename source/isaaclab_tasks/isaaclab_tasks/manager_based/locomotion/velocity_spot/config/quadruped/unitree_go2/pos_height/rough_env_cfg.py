from isaaclab.utils import configclass
from robot_lab.tasks.locomotion.velocity_spot.velocity_height_env_cfg import LocomotionVelocityHeightRoughEnvCfg
from robot_lab.assets.unitree import *  # isort: skip


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityHeightRoughEnvCfg):
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
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.2
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.2
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0

        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Observations------------------------------
        
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------

        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.joint_names = self.joint_names
        self.actions.joint_vel = None
        self.actions.joint_eff = None
        # self.actions.joint_stiffness = None
        # self.actions.joint_damping = None

        # ------------------------------Events------------------------------

        # self.events.randomize_reset_base.params = {
        #     "pose_range": {
        #         "x": (-0.5, 0.5),
        #         "y": (-0.5, 0.5),
        #         "z": (0.0, 0.2),
        #         "roll": (-3.14, 3.14),
        #         "pitch": (-3.14, 3.14),
        #         "yaw": (-3.14, 3.14),
        #     },
        #     "velocity_range": {
        #         "x": (-0.5, 0.5),
        #         "y": (-0.5, 0.5),
        #         "z": (-0.5, 0.5),
        #         "roll": (-0.5, 0.5),
        #         "pitch": (-0.5, 0.5),
        #         "yaw": (-0.5, 0.5),
        #     },
        # }

        # ------------------------------Rewards------------------------------

        self.rewards.base_height_l2.weight = -100.0
        # self.rewards.base_height_l2_exp.weight = 5.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------

        # self.terminations.base_contact = None
        self.terminations.hip_contact = None
        self.terminations.thigh_contact = None
        self.terminations.calf_contact = None
        self.terminations.bad_orientation = None

        # ------------------------------Commands------------------------------


        
