from isaaclab.utils import configclass
from .rough_env_cfg_pos_rnn import UnitreeGo2RoughEnvCfg_pos_rnn


@configclass
class UnitreeGo2FlatEnvCfg_pos_rnn(UnitreeGo2RoughEnvCfg_pos_rnn):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        self.scene.height_scanner = None
        self.commands.base_height.sensor_name = None
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.base_height_l2_exp.params["sensor_cfg"] = None

        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2FlatEnvCfg_pos_rnn":
            self.disable_zero_weight_rewards()