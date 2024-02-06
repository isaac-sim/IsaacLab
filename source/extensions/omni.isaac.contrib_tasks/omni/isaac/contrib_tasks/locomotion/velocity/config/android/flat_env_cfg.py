from omni.isaac.orbit.utils import configclass

from .rough_env_cfg import AndroidRoughEnvCfg


@configclass
class AndroidFlatEnvCfg(AndroidRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class AndroidFlatEnvCfg_PLAY(AndroidFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None