import argparse
import torch
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from deformable_cube_env import DeformableCubeEnv, DeformableCubeEnvCfg

if __name__ == "__main__":
	env_cfg = DeformableCubeEnvCfg()
	env = DeformableCubeEnv(env_cfg)
	env.reset()
	while simulation_app.is_running():
		env.step(torch.randn(env_cfg.num_envs, env_cfg.action_space, device="cuda"))
