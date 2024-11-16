import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ur5_basic_control import Ur5JointController


class UR5GymEnv(gym.Env):
    def __init__(self):
        super(UR5GymEnv, self).__init__()
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )
        # A ros2 node that connects to the UR5 robot driver
        self.ur5_controller = Ur5JointController()

    def step(self, action: list[float]):
        """Set Delta joint angles to the UR5 robot to move the arm. Values are normalized between -1 and 1.

        Args:
            action (list[float]): A list of 6 floats representing the delta joint angles to be applied to the UR5 robot. Order of the joints: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
        """
        # Execute the action
        self.ur5_controller.set_joint_delta(action)

    def _get_obs(self):
        pass

    def _is_done(self):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass
