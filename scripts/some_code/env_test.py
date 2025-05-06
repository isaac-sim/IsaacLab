import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.locomotion.velocity_rma.config.quadruped.unitree_go2.pos.flat_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg

env_cfg = Go2FlatEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env = ManagerBasedRLEnv(cfg=env_cfg)

# simulate physics
count = 0
while simulation_app.is_running():
    with torch.inference_mode():
        action = torch.zeros_like(env.action_manager.action)
        obs, rew, terminated, truncated, info = env.step(action)
        # print(rew)
        count += 1

env.close()
simulation_app.close()
