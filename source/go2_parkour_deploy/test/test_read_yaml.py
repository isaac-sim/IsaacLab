import re
import os 
import parkour_isaaclab
import torch
import numpy as np 
from scripts.utils import load_local_cfg
from core.utils import isaac_to_mujoco, ISAAC_JOINT_NAMES

def main(args):
    """Play with RSL-RL agent."""
    logs_path = '/'
    for path in parkour_isaaclab.__path__[0].split('/')[1:-1]:
        logs_path = os.path.join(logs_path, path)
    logs_path = os.path.join(logs_path,'logs',args.rl_lib,args.task, args.expid)
    cfgs_path = os.path.join(logs_path, 'params')
    env_cfg = load_local_cfg(cfgs_path, 'env')
    default_joint_compiled = {re.compile(joint_name): value  for joint_name, value in env_cfg.scene.robot.init_state.joint_pos.items()}
    saturation_limit_compiled = {re.compile(joint_name): value  for joint_name, value in env_cfg.scene.robot.actuators.base_legs.saturation_effort.items()}
    velocity_limit_compiled = {re.compile(joint_name): value  for joint_name, value in env_cfg.scene.robot.actuators.base_legs.velocity_limit.items()}
    effort_limit_compiled = {re.compile(joint_name): value  for joint_name, value in env_cfg.scene.robot.actuators.base_legs.effort_limit.items()}
    default_joint_data = torch.zeros((1, len(ISAAC_JOINT_NAMES)), device='cuda:0')
    default_joint_dict = {}
    _saturation_effort = torch.zeros((1, 12), device='cuda:0')
    _saturation_effort_dict = {}
    _velocity_limit = torch.zeros((1, 12), device='cuda:0')
    _velocity_limit_dict = {}
    _effort_limit = torch.zeros((1, 12), device='cuda:0')
    _effort_limit_dict = {}




    for joint_name_compiled in default_joint_compiled.keys():
        for indx,real_joint_name in enumerate(ISAAC_JOINT_NAMES):
            if joint_name_compiled.match(real_joint_name):
                default_joint_data[0,indx] = default_joint_compiled[joint_name_compiled]
                default_joint_dict[real_joint_name] = default_joint_compiled[joint_name_compiled]

    for joint_name_compiled in saturation_limit_compiled.keys():
        for indx, real_joint_name in enumerate(ISAAC_JOINT_NAMES):
            if joint_name_compiled.match(real_joint_name):
                _saturation_effort[0,indx] = saturation_limit_compiled[joint_name_compiled]
                _saturation_effort_dict[real_joint_name] = saturation_limit_compiled[joint_name_compiled]

                _velocity_limit[0,indx] = velocity_limit_compiled[joint_name_compiled]
                _velocity_limit_dict[real_joint_name] = velocity_limit_compiled[joint_name_compiled]

                _effort_limit[0,indx] = effort_limit_compiled[joint_name_compiled]
                _effort_limit_dict[real_joint_name] = effort_limit_compiled[joint_name_compiled]

    print(f'[saturation_effort]:{_saturation_effort_dict}')
    print(f'[velocity_limit]:{_velocity_limit_dict}')
    print(f'[effort_limit]:{_effort_limit_dict}')
    print(f'[default_joint_data]:{default_joint_dict}')


if __name__ == "__main__":
    import argparse
    # mp.set_start_method("spawn")
    parser = argparse.ArgumentParser(description='sim_2_sim')
    parser.add_argument("--rl_lib", type=str, default='rsl_rl')
    parser.add_argument("--task", type=str, default='unitree_go2_parkour')
    parser.add_argument("--expid", type=str, default='2025-08-13_05-15-58')
    args = parser.parse_args()
    main(args)
