import torch as th
from mujoco_deploy.mujoco_wrapper import MujocoWrapper
import os
import core
from typing import Dict

class DeploymentPlayer:
    def __init__(
        self,
        env_cfg,
        agent_cfg, 
        network_interface,
        logs_path,
    ):
        try: 
            env_cfg.scene.depth_camera
            use_camera = True 
        except: 
            use_camera = False 

        if network_interface.lower() =='lo':
            self.env = MujocoWrapper(env_cfg, agent_cfg, os.path.join(core.__path__[0],'go2/scene_parkour.xml'), use_camera)
        self._use_camera = use_camera
        if self._use_camera:
            self.policy = th.jit.load(os.path.join(logs_path,'exported_deploy','policy.pt'), map_location=self.env.device)
            self.policy.eval()

            self.depth_encoder = th.jit.load(os.path.join(logs_path,'exported_deploy','depth_latest.pt'), map_location=self.env.device)
            self.depth_encoder.eval()
            print("use camera ..............")

        else:
            self.policy = th.jit.load(os.path.join(logs_path,'exported_teacher','policy.pt'), map_location=self.env.device)
            self.policy.eval()
            self.depth_encoder = None
            print("not use camera ..............")

        self._clip_actions = agent_cfg['clip_actions']
        estimator_paras = agent_cfg["estimator"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.num_priv_explicit = estimator_paras["num_priv_explicit"]
        self.history_len = 10 
        self.cnt = 0 
        self._call_cnt = 0
        self._maximum_iteration = float('inf')
        print(self.policy)
        
    def play(self):
        """Advances the environment one time step after generating observations"""
        obs, extras = self.env.get_observations()
        with th.inference_mode():
            if not self._use_camera:
                actions = self.policy(obs , hist_encoding=True)
            else:
                if self.env.common_step_counter %5 == 0:
                    depth_image = extras["observations"]['depth_camera']
                    proprioception = obs[:, :self.num_prop].clone()
                    proprioception[:, 6:8] = 0
                    depth_latent_and_yaw = self.depth_encoder(depth_image , proprioception )
                    self.depth_latent = depth_latent_and_yaw[:, :-2]
                    self.yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5*self.yaw
                actions = self.policy(obs , scandots_latent=self.depth_latent)
        if self._clip_actions is not None:
            actions = th.clamp(actions, -self._clip_actions, self._clip_actions)
        obs, terminated, timeout, extras = self.env.step(actions)  # For HW, this internally just does forward

        self.cnt += 1
        return obs, terminated, timeout, extras
    
    def reset(self, maximum_iteration: int |None = None, extras: Dict[str, str] | None = None):
        self._call_cnt +=1 
        if type(maximum_iteration) == int:
            self.maximum_iteration = maximum_iteration
        if self.alive():
            self.env.reset() 
            print('[Current eval iter]: ', self._call_cnt, '[Left]: ', self.maximum_iteration-self._call_cnt)

    def alive(self):
        if self._call_cnt <= self.maximum_iteration:
            return True
        else:
            self.env.close()
            return False 
        
