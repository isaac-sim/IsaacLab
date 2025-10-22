from isaaclab.envs import ManagerBasedRLEnv
from .locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg


class LocomanipulationG1ManagerBasedRLEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: LocomanipulationG1EnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        cfg.on_environment_initialized()
        print("[INFO]: Completed setting up the LocomanipulationG1ManagerBasedRLEnv...")
