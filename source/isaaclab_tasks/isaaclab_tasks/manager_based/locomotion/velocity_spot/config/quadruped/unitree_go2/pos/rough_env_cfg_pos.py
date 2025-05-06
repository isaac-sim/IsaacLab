from isaaclab.utils import configclass
from robot_lab.tasks.locomotion.velocity_spot.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from robot_lab.assets.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2RoughEnvCfg_pos(LocomotionVelocityRoughEnvCfg):
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
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        
        self.scene.terrain.terrain_generator.sub_terrains["flat"].proportion = 0.5
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.5
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0

        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Observations------------------------------
        
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # self.observations.policy.actions = None
        # self.observations.critic.actions = None

        # ------------------------------Actions------------------------------

        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.joint_names = self.joint_names
        self.actions.joint_vel = None
        self.actions.joint_eff = None
        self.actions.joint_stiffness = None
        self.actions.joint_damping = None

        # ------------------------------Events------------------------------


        # ------------------------------Rewards------------------------------

        # self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
        #     self.base_link_name,
        #     '.*_hip', 
        #     '.*_thigh', 
        #     '.*_calf'
        # ]

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2RoughEnvCfg_pos":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------

        # self.terminations.base_contact = None
        self.terminations.hip_contact = None
        self.terminations.thigh_contact = None
        self.terminations.calf_contact = None

        # ------------------------------Commands------------------------------

@configclass
class UnitreeGo2RoughEnvCfg_pos_PLAY(UnitreeGo2RoughEnvCfg_pos):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 64
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


        
