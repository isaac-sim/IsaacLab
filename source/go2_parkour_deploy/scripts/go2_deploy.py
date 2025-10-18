import os
# os.environ["MUJOCO_GL"] = "osmesa"
# os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ["SDL_AUDIODRIVER"] = "dummy"

import sys
sys.path.append("/home/wang/IsaacLab/Isaaclab_Parkour")

import parkour_isaaclab
sys.path.append("/home/wang/IsaacLab/go2_parkour_deploy")
from scripts.utils import load_local_cfg
from core.deployment_player import DeploymentPlayer
# import multiprocessing as mp
import sys

def main(args):
    """Play with RSL-RL agent."""
    logs_path = '/'
    for path in parkour_isaaclab.__path__[0].split('/')[1:-1]:
        logs_path = os.path.join(logs_path, path)
    logs_path = os.path.join(logs_path,'logs',args.rl_lib,args.task, args.expid)
    # model_path = os.path.join(logs_path, f'{args.model_id}.pt')
    cfgs_path = os.path.join(logs_path, 'params')
    env_cfg = load_local_cfg(cfgs_path, 'env')
    agent_cfg = load_local_cfg(cfgs_path, 'agent')
    env_cfg.scene.num_envs = 1
    
    player = DeploymentPlayer(
        env_cfg=env_cfg,
        agent_cfg = agent_cfg, 
        network_interface= args.interface,
        logs_path = logs_path, 
    )
    
    player.reset(maximum_iteration = args.n_eval)
    while player.alive():
        _, terminated, timeout, extras = player.play()
        if terminated or timeout:
           player.reset(extras = extras)
    print('Eval Done')
    
    sys.exit()

if __name__ == "__main__":
    import argparse
    # mp.set_start_method("spawn")
    parser = argparse.ArgumentParser(description='sim_2_sim')
    parser.add_argument("--rl_lib", type=str, default='rsl_rl')
    parser.add_argument("--task", type=str, default='unitree_go2_parkour_student_ppo')
    parser.add_argument("--expid", type=str, default='2025-09-03_12-07-56')
    parser.add_argument("--interface", type=str, default='lo')
    parser.add_argument("--use_joystick", action='store_true', default=False)
    parser.add_argument("--n_eval", type=int, default=10)
    args = parser.parse_args()
    main(args)