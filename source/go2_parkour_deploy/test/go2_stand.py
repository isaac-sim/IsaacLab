from mujoco_deploy.mujoco_wrapper import MujocoWrapper
import os ,core, copy
import parkour_isaaclab
from scripts.utils import load_local_cfg
import torch as th 
from core.utils import isaac_to_mujoco, mujoco_to_isaac
import time,mujoco
# stand_up_joint_pos = th.tensor([
#     0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
#     0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763
# ],
# dtype=float).to('cuda:0')[mujoco_to_isaac]

stand_down_joint_pos = th.tensor([
    0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
    1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
],
dtype=float).to('cuda:0')[mujoco_to_isaac]


class TESTRESET(MujocoWrapper):
    
    def reset(self):
        self.common_step_counter = 0
        self._mujoco_env.reset()
        for i in range(100):
            self._mujoco_env.step()
        self._get_depth_image(True)
        init_pose = self._mujoco_env.articulation.joint_pos
        runing_time = 0.0
        while runing_time<3.0: 
            runing_time += 0.1
            self.sensor_update()
            self.sensor_render()
            self._get_depth_image()
            phase = th.tanh(th.tensor([runing_time / 1.2]).to('cuda:0'))
            cur_pose = phase * stand_down_joint_pos + (1-phase) * init_pose
            self._processed_actions = self._mujoco_env.articulation.joint_stiffness*(cur_pose - self._mujoco_env.articulation.joint_pos) + \
                self._mujoco_env.articulation.joint_dampings * (self._mujoco_env.articulation.control_joint_velocities - self._mujoco_env.articulation.joint_vel)
            processed_action_np = self._processed_actions.detach().cpu().numpy()
            self._mujoco_env.step(processed_action_np)
        print('[INFO] setting stand down joint pos is done')
        runing_time = 0.0
        while runing_time<3.0: 
            runing_time += 0.002
            self.sensor_update()
            self.sensor_render()
            self._get_depth_image()
            phase = th.tanh(th.tensor([runing_time / 1.2]).to('cuda:0'))
            cur_pose = phase * self._mujoco_env.default_joint_pose + (1-phase) * stand_down_joint_pos
            self._processed_actions = self._mujoco_env.articulation.joint_stiffness*(cur_pose - self._mujoco_env.articulation.joint_pos) + \
                self._mujoco_env.articulation.joint_dampings * (self._mujoco_env.articulation.control_joint_velocities - self._mujoco_env.articulation.joint_vel)
            processed_action_np = self._processed_actions.detach().cpu().numpy()
            self._mujoco_env.step(processed_action_np)

        cur_pose = self._mujoco_env.default_joint_pose
        self._processed_actions = self._mujoco_env.articulation.joint_stiffness*(cur_pose - self._mujoco_env.articulation.joint_pos) + \
            self._mujoco_env.articulation.joint_dampings * (self._mujoco_env.articulation.control_joint_velocities - self._mujoco_env.articulation.joint_vel)
        processed_action_np = self._processed_actions.detach().cpu().numpy()
        self._mujoco_env.step(processed_action_np)

        print('[INFO] setting default joint pose is done')

def main(args):
    logs_path = '/'
    for path in parkour_isaaclab.__path__[0].split('/')[1:-1]:
        logs_path = os.path.join(logs_path, path)
    logs_path = os.path.join(logs_path,'logs',args.rl_lib,args.task, args.expid)
    cfgs_path = os.path.join(logs_path, 'params')
    env_cfg = load_local_cfg(cfgs_path, 'env')
    agent_cfg = load_local_cfg(cfgs_path, 'agent')

    env = TESTRESET(env_cfg, agent_cfg, os.path.join(core.__path__[0],'go2/scene_parkour.xml'), True)
    env.reset()

if __name__ == "__main__":
    import argparse
    # mp.set_start_method("spawn")
    parser = argparse.ArgumentParser(description='sim_2_sim')
    parser.add_argument("--rl_lib", type=str, default='rsl_rl')
    parser.add_argument("--task", type=str, default='unitree_go2_parkour')
    parser.add_argument("--expid", type=str, default='2025-09-03_12-07-56')
    parser.add_argument("--interface", type=str, default='lo')
    parser.add_argument("--use_joystick", action='store_true', default=False)
    parser.add_argument("--n_eval", type=int, default=10)
    args = parser.parse_args()
    main(args)
