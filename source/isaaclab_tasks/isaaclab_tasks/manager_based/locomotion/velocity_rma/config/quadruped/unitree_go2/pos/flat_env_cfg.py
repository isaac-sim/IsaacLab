from isaaclab.utils import configclass
from .rough_env_cfg import Go2RoughEnvCfg


@configclass
class Go2FlatEnvCfg(Go2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "Go2FlatEnvCfg":
            self.disable_zero_weight_rewards()