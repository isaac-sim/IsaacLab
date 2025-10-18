import pygame
import sys
import numpy as np
import pygame
from threading import Thread
from collections import deque
import torch as th 

class MujocoJoystick:
    """
    Reference from 
    """
    def __init__(self, env_cfg, device):
        self._init_joystick()
        self.x_vel = 0
        self.y_vel = 0
        self.yaw = 0
        self._device = device
        self.joystick = pygame.joystick.Joystick(0)
        commands_cfg = env_cfg.commands.base_velocity
        self._resampling_time_range = commands_cfg.resampling_time_range[-1]
        self._small_commands_to_zero = commands_cfg.small_commands_to_zero
        self._lin_vel_x_range = commands_cfg.ranges.lin_vel_x
        self._heading = commands_cfg.ranges.heading
        self._lin_vel_clip = commands_cfg.clips.lin_vel_clip
        self._ang_vel_clip = commands_cfg.clips.ang_vel_clip
        self._stopping = False
        self._listening_thread = None



    def _init_joystick(self, device_id=0):
        """
        We are only support gamepad joystick type
        """
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("No gamepad detected.")
            sys.exit()
        print(f"[INFO] Initialized {self.joystick.get_name()}")
        print(f"[INFO] Joystick power level {self.joystick.get_power_level()}")
        buffer_length = 10
        self._x_buffer = deque([0] * buffer_length, buffer_length)
        self._velocity_cmd = np.zeros((1,3))

    def start_listening(self):
        self._listening_thread = Thread(target=self.listen)
        self._listening_thread.start()

    def listen(self):
        while not self._stopping:
            pygame.event.pump()
            x_input = (self.joystick.get_axis(0))  * (self._lin_vel_x_range[1] - self._lin_vel_x_range[0]) + self._lin_vel_x_range[0]
            # print("joystick axis 0 (x):", self.joystick.get_axis(0))
            self._x_buffer.append(x_input)
            self.x_vel = np.median(self._x_buffer)
            self._velocity_cmd[:] = np.array([[self.x_vel, self.y_vel, self.yaw]])
            pygame.time.wait(10)


    def reset(self):
        self._x_buffer.clear()

    @property 
    def velocity_cmd(self): 
        if self._small_commands_to_zero:
            self._velocity_cmd[:,:2] *= np.abs(self._velocity_cmd[:, 0:1]) \
                                            > self._lin_vel_clip
        return th.tensor(self._velocity_cmd).to(self._device)
