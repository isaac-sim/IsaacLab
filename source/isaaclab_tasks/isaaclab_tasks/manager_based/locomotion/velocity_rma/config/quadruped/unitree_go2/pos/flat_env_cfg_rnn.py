from isaaclab.utils import configclass
from .rough_env_cfg_rnn import Go2RoughEnvCfg_rnn


@configclass
class Go2FlatEnvCfg_rnn(Go2RoughEnvCfg_rnn):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "Go2FlatEnvCfg_rnn":
            self.disable_zero_weight_rewards()
