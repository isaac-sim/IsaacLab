import torch as th 
import numpy as np 
from mujoco_deploy.mujoco_sensors.mujoco_raycaster import MujocoRaycaster
from mujoco_deploy.mujoco_sensors.mujoco_contact_sensor import MujocoContactSensor
from mujoco_deploy.mujoco_sensors.mujoco_depth_camera import MujocoDepthCamera
from isaaclab.utils.math  import euler_xyz_from_quat, wrap_to_pi
from mujoco_deploy.mujoco_env import MujocoEnv
from mujoco_deploy.mujoco_sensors.mujoco_joystick_controller import MujocoJoystick
import copy, cv2, torchvision
from core.utils import isaac_to_mujoco, mujoco_to_isaac, stand_down_joint_pos
import math
import matplotlib.pyplot as plt





from tqdm import tqdm
class MujocoWrapper():
    """
    Apply Reward, Oberservation, Termination in here 
    """
    def __init__(
        self, 
        env_cfg,
        agent_cfg,
        model_xml_path:str,
        use_camera: bool,
        ):
        self._mujoco_env = MujocoEnv(env_cfg, model_xml_path, use_camera)
        self._use_camera = use_camera
        self.device = self._mujoco_env.articulation.device
        self.common_step_counter = 0 
        self.decimation = env_cfg.decimation 
        self._history_length = env_cfg.observations.policy.extreme_parkour_observations.params.history_length
        self._clip = th.tensor([[*env_cfg.actions.joint_pos.clip['.*']]], device=self.device).repeat(
            1, self._mujoco_env.articulation.num_motor, 1
            )
        self._observation_clip = env_cfg.observations.policy.extreme_parkour_observations.clip[-1]
        self._agent_cfg = agent_cfg
        self._stand_down_joint_pos = th.tensor(stand_down_joint_pos,dtype=float).to('cuda:0')[mujoco_to_isaac]
        self._init_sensors()
        self._init_commands()
        self._init_action_buffers()

    def _init_commands(self):
        self._joystick = MujocoJoystick(self._mujoco_env.env_cfg, self.device)
        self._joystick.start_listening()

    def _init_sensors(self):
        self._sensor_term = []
        if not self._use_camera:
            self._height_scanner = MujocoRaycaster(self._mujoco_env.env_cfg, 
                            self._mujoco_env.articulation, 
                            self._mujoco_env.model, 
                            self._mujoco_env.data)
            self._sensor_term.append(self._height_scanner)
        else: 
            self._depth_camera = MujocoDepthCamera(self._mujoco_env.env_cfg, 
                    self._mujoco_env.articulation.device,
                    self._mujoco_env.model, 
                    self._mujoco_env.data)
            self._sensor_term.append(self._depth_camera)
            self.clipping_range = self._depth_camera.sensor_cfg.max_distance
            self.resize_transform = torchvision.transforms.Resize(
                            self._mujoco_env.env_cfg.observations.depth_camera.depth_cam.params.resize, 
                            interpolation=torchvision.transforms.InterpolationMode.BICUBIC).to(self.device)
            self.depth_buffer = th.zeros(self.num_envs,  
                                self._mujoco_env.env_cfg.observations.depth_camera.depth_cam.params.buffer_len, 
                                *self._mujoco_env.env_cfg.observations.depth_camera.depth_cam.params.resize).to(self.device)

        self._contact_sensor = MujocoContactSensor(self._mujoco_env.env_cfg, 
                            self._mujoco_env.articulation, 
                            self._mujoco_env.model, 
                            self._mujoco_env.data)
        
        self._sensor_term.append(self._contact_sensor)
        
    def _init_action_buffers(self):
        joint_pos_cfg = self._mujoco_env.env_cfg.actions.joint_pos
        self._action_history_length = joint_pos_cfg.history_length
        self._delay_update_global_steps = int(joint_pos_cfg.delay_update_global_steps)
        self._use_delay = joint_pos_cfg.use_delay
        self._action_delay_steps = joint_pos_cfg.action_delay_steps
        self._action_history_buf = th.zeros(1, self._action_history_length, self._mujoco_env.articulation.num_motor, device=self.device, dtype=th.float)
        self._actions = th.zeros(1, self._mujoco_env.articulation.num_motor, device=self.device)
        self._processed_actions = th.zeros(1, self._mujoco_env.articulation.num_motor, device=self.device)
        self._obs_history_buffer = th.zeros(1, self._history_length, self._agent_cfg.estimator.num_prop, device=self.device)
        self.episode_length_buf = th.zeros(1, device=self.device, dtype=th.long)
        self._delta_yaw = th.zeros(1,1).to(self.device)
        self._delta_next_yaw = th.zeros(1,1).to(self.device)
        self._priv_explicit = th.zeros(1, self._agent_cfg.estimator.num_priv_explicit, device=self.device, dtype=th.float32)
        self._priv_latent = th.zeros(1, self._agent_cfg.estimator.num_priv_latent, device=self.device, dtype=th.float32)
        self._dummy_scan = th.zeros(1, self._agent_cfg.estimator.num_scan, device=self.device, dtype=th.float32)

    def _init_pose_stand_up(self):
        runing_time = 0.0
        with tqdm(total=3.0, desc="[INFO] Setting up the initial posture ...") as pbar:   
            while runing_time < 3.0:
                self.sensor_update()
                self.sensor_render()
                runing_time += self._mujoco_env.env_cfg.sim.dt
                pbar.update(min(self._mujoco_env.env_cfg.sim.dt, 3.0 - pbar.n)) 
                phase = th.tanh(th.tensor([runing_time / 1.2]).to('cuda:0'))
                cur_pose = phase * self._mujoco_env.default_joint_pose + (1-phase) * self._stand_down_joint_pos
                self._init_actions = self._mujoco_env.articulation.joint_stiffness*(cur_pose - self._mujoco_env.articulation.joint_pos) + \
                    self._mujoco_env.articulation.joint_dampings * (self._mujoco_env.articulation.control_joint_velocities - self._mujoco_env.articulation.joint_vel)
                processed_action_np = self._init_actions.detach().cpu().numpy()
                self._mujoco_env.step(processed_action_np)

    def _init_pose_stand_down(self):
        runing_time = 0.0
        init_pose = self._mujoco_env.articulation.joint_pos
        with tqdm(total=3.0, desc="[INFO] Setting up the stand down posture ...") as pbar:   
            while runing_time < 3.0:
                self.sensor_update()
                self.sensor_render()
                runing_time += 0.1
                pbar.update(min(0.1, 3.0 - pbar.n)) 
                phase = th.tanh(th.tensor([runing_time / 1.2]).to('cuda:0'))
                cur_pose = phase * self._stand_down_joint_pos + (1-phase) * init_pose
                self._init_actions = self._mujoco_env.articulation.joint_stiffness*(cur_pose - self._mujoco_env.articulation.joint_pos) + \
                    self._mujoco_env.articulation.joint_dampings * (self._mujoco_env.articulation.control_joint_velocities - self._mujoco_env.articulation.joint_vel)
                processed_action_np = self._init_actions.detach().cpu().numpy()
                self._mujoco_env.step(processed_action_np)

    def reset(self):
        self.common_step_counter = 0
        self._mujoco_env.reset()
        self.episode_length_buf = th.zeros(1, device=self.device, dtype=th.long)
        for i in range(100):
            self._mujoco_env.step()
        self._init_pose_stand_down()
        self._init_pose_stand_up()
        self.sensor_update()
        self.sensor_render()
        print('[INFO] Initial posture setting complete')

    def get_observations(self):
        self.roll, self.pitch, yaw = euler_xyz_from_quat(self._mujoco_env.articulation.root_quat_w)
        imu_obs = th.stack((wrap_to_pi(self.roll), wrap_to_pi(self.pitch)), dim=1).to(self.device)
        height_scan = th.clip(self._height_scanner.sensor_data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.sensor_data.ray_hits_w[..., 2] - 0.3, -1, 1).to(self.device) \
                        if not self._use_camera  else\
                        self._dummy_scan
        height_scan = th.clip(height_scan, -1, 1).to(self.device)
        env_idx_tensor = th.tensor([True]).to(dtype = th.bool, device=self.device)
        invert_env_idx_tensor = ~env_idx_tensor
        commands = self._joystick.velocity_cmd
        print("commands:",commands)
        self._delta_next_yaw[:] = self._delta_yaw[:] = wrap_to_pi(yaw)[:,None]
        obs_buf = th.cat((
                            self._mujoco_env.articulation.root_ang_vel_b* 0.25,   #[1,4]
                            imu_obs,    #[1,2]
                            0*self._delta_yaw, 
                            self._delta_yaw,
                            self._delta_next_yaw,
                            0*commands[:, 0:2], 
                            commands[:, 0:1],  #[1,1]
                            env_idx_tensor.float()[:, None],
                            invert_env_idx_tensor.float()[:, None],
                            self._mujoco_env.articulation.joint_pos - self._mujoco_env.default_joint_pose,
                            self._mujoco_env.articulation.joint_vel* 0.05,
                            self._action_history_buf[:, -1],
                            self._get_contact_fill(),
                            ),dim=-1)
        
        observations = th.cat([obs_buf, #53 
                                height_scan, #132 
                                self._priv_explicit, # 9
                                self._priv_latent, #29
                                self._obs_history_buffer.view(1, -1)
                                ], dim=-1)
        obs_buf[:, 6:8] = 0
        self._obs_history_buffer = th.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            th.stack([obs_buf] * self._history_length, dim=1),
            th.cat([
                self._obs_history_buffer[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )
        depth_image = self._get_depth_image()
        observations = th.clip(observations, min = -self._observation_clip, max = self._observation_clip)
        extras = {'observations':{"policy":observations.to(th.float),
                                  'depth_camera':depth_image.to(th.float)}}
        return observations.to(th.float), extras
    
    def _process_depth_image(self, depth_image):
        depth_image = self._crop_depth_image(depth_image)
        depth_image = self.resize_transform(depth_image[None, :]).squeeze()
        depth_image = self._normalize_depth_image(depth_image)
        return depth_image
    
    def _crop_depth_image(self, depth_image):
        # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
        return depth_image[:-2, 4:-4]
    
    def _normalize_depth_image(self, depth_image):
        depth_image = (depth_image) / (self.clipping_range)  - 0.5
        return depth_image

    def _get_depth_image(
        self,
        is_reset: bool = False 
        ):
        depth_image = self._depth_camera.sensor_data.output["distance_to_camera"].squeeze(-1)[:]
        processed_image = self._process_depth_image(depth_image)
        if is_reset:
            self.depth_buffer[0] = th.stack([processed_image]* 2, dim=0)
        if self.common_step_counter % 5 ==0:
            self.depth_buffer[0] = th.cat([self.depth_buffer[0, 1:], 
                                    processed_image.to(self.device).unsqueeze(0)], dim=0)
            # cv2.imshow('processed_image',processed_image.detach().cpu().numpy())
            # cv2.waitKey(1)
        return self.depth_buffer[:, -2].to(self.device)
    
    def _get_contact_fill( 
        self,
        ):
        foot_ids = self._contact_sensor.sensor_data.foot_ids
        contact_forces = self._contact_sensor.sensor_data.net_forces_w_history[:, 0, foot_ids] #(N, 4, 3)
        previous_contact_forces = self._contact_sensor.sensor_data.net_forces_w_history[:, -1, foot_ids] # N, 4, 3
        contact = th.norm(contact_forces, dim=-1) > 2.
        last_contacts = th.norm(previous_contact_forces, dim=-1) > 2.
        contact_filt = th.logical_or(contact, last_contacts) 
        return (contact_filt.float()-0.5).to(self.device)

    def _apply_action(self):
        """Applies the current action to the robot."""
        """ class LocomotionEnv(DirectRLEnv)"""
        """ 1. make processed actions"""
        error_pos = self._processed_actions - self._mujoco_env.articulation.joint_pos
        error_vel = self._mujoco_env.articulation.control_joint_velocities - self._mujoco_env.articulation.joint_vel # type: ignore
        self.computed_effort = self._mujoco_env.articulation.joint_stiffness * error_pos \
                               + self._mujoco_env.articulation.joint_dampings * error_vel 
        

        # """DCMotor _clip_effort"""
        max_effort = self._mujoco_env.articulation.saturation_effort \
            * (1.0 - self._mujoco_env.articulation.joint_vel/ self._mujoco_env.articulation.velocity_limit)
        max_effort = th.clip(max_effort, min=self._mujoco_env.articulation.zeros_effort, max=self._mujoco_env.articulation.effort_limit)
        min_effort = self._mujoco_env.articulation.saturation_effort \
            * (-1.0 - self._mujoco_env.articulation.joint_vel / self._mujoco_env.articulation.velocity_limit)
        min_effort = th.clip(min_effort, min=-self._mujoco_env.articulation.effort_limit, max=self._mujoco_env.articulation.zeros_effort)
        self._applied_effort = th.clip(self.computed_effort, min=min_effort, max=max_effort)

    def _process_actions(self):
        if self.common_step_counter % self._delay_update_global_steps == 0:
            if len(self._action_delay_steps) != 0:
                self.delay = th.tensor(self._action_delay_steps.pop(0), device=self.device, dtype=th.float)
        self._action_history_buf = th.cat([self._action_history_buf[:, 1:].clone(), self._actions[:, None, :].clone()], dim=1)
        indices = -1 - self.delay
        if self._use_delay:
            self._actions = self._action_history_buf[:, indices.long()]

        if self._mujoco_env.env_cfg.actions.joint_pos.clip is not None:
            self._actions = th.clamp(
                self._actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        if self._mujoco_env.env_cfg.actions.joint_pos.use_default_offset:
            """ JointPositionAction """
            """ process_actions: self._raw_actions * self._scale + self._offset """
            self._processed_actions = self._actions * self._mujoco_env.env_cfg.actions.joint_pos.scale + self._mujoco_env.default_joint_pose

        else:
            self._processed_actions = self._actions * self._mujoco_env.env_cfg.actions.joint_pos.scale 

    def step(self, actions: th.Tensor | None = None):
        ### actions is isaacsim based
        self._actions = actions.clone()       
        self._process_actions()
        self.common_step_counter += 1
        self.episode_length_buf += 1  # step in current episode (per env)

        for _ in range(self.decimation):
            self._apply_action()
            applied_effort_np = self._applied_effort.detach().cpu().numpy()
            self._mujoco_env.step(applied_effort_np)
            self.sensor_render()
            self.sensor_update()
        obs, extras = self.get_observations()
        termination = self._termination()
        max_episode_length = math.ceil(self._mujoco_env.env_cfg.episode_length_s/(self._mujoco_env.env_cfg.sim.dt  * self.decimation))
        time_out_buf = self.episode_length_buf >= max_episode_length
        return obs , termination , time_out_buf, extras
    
    def _termination(self):
        reset_buf = th.zeros((1, ), dtype=th.bool, device=self._mujoco_env.articulation.device)
        roll_cutoff = th.abs(wrap_to_pi(self.roll)) > 1.5
        pitch_cutoff = th.abs(wrap_to_pi(self.pitch)) > 1.5
        height_cutoff = self._mujoco_env.articulation.root_state_w[:, 2] < -0.25
        reset_buf |= roll_cutoff
        reset_buf |= pitch_cutoff
        reset_buf |= height_cutoff  
        return reset_buf

    def sensor_render(self):
        for sensor_term in self._sensor_term:
            if hasattr(sensor_term, 'render'): 
                sensor_term.render(self._mujoco_env.viewer)
            else:
                continue

    def sensor_update(self):
        for sensor_term in self._sensor_term:
            sensor_term.update(self._mujoco_env.env_cfg.sim.dt)
        
    @property
    def sim(self):
        return self._mujoco_env

    @property
    def num_actions(self):
        return self._mujoco_env.articulation.num_motor

    @property
    def num_envs(self):
        return 1

    def close(self):
        self._mujoco_env.close()

