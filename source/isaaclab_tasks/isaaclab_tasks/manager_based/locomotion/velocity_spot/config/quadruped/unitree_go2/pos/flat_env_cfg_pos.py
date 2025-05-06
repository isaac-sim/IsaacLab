from isaaclab.utils import configclass
from .rough_env_cfg_pos import UnitreeGo2RoughEnvCfg_pos


@configclass
class UnitreeGo2FlatEnvCfg_pos(UnitreeGo2RoughEnvCfg_pos):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2FlatEnvCfg_pos":
            self.disable_zero_weight_rewards()

@configclass
class UnitreeGo2FlatEnvCfg_pos_PLAY(UnitreeGo2FlatEnvCfg_pos):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 64
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        self.observations.policy.imu_ang_vel.noise = None
        self.observations.policy.imu_lin_acc.noise = None
        self.observations.policy.joint_pos.noise = None
        self.observations.policy.joint_vel.noise = None

        # self.events = None
        self.episode_length_s = 5
