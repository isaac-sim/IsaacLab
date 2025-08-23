from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import isaaclab.utils.math as math_utils
from .utils import torch_rand_float_tensor

if TYPE_CHECKING:
    from .controller_cfg import LeeControllerCfg
    from isaaclab.assets import Articulation

class BaseLeeController:
    """
    This class will operate as the base class for all controllers.
    It will be inherited by the specific controller classes.
    """

    cfg: LeeControllerCfg
    
    def __init__(self, cfg: LeeControllerCfg, env):
        self.cfg = cfg
        self.env = env
        self.robot: Articulation = env.scene["robot"]

    def init_tensors(self, global_tensor_dict):
        self.robot_position = global_tensor_dict["robot_position"]
        self.robot_orientation = global_tensor_dict["robot_orientation"]
        self.robot_linvel = global_tensor_dict["robot_linvel"]
        self.robot_angvel = global_tensor_dict["robot_angvel"]
        self.robot_vehicle_orientation = global_tensor_dict["robot_vehicle_orientation"]
        self.robot_vehicle_linvel = global_tensor_dict["robot_vehicle_linvel"]
        self.robot_body_angvel = global_tensor_dict["robot_body_angvel"]
        self.robot_body_linvel = global_tensor_dict["robot_body_linvel"]
        self.robot_euler_angles = global_tensor_dict["robot_euler_angles"]
        self.mass = global_tensor_dict["robot_mass"].unsqueeze(1)
        self.robot_inertia = global_tensor_dict["robot_inertia"]
        self.gravity = global_tensor_dict["gravity"]

        # Read from config and set the values for controller parameters
        self.K_pos_tensor_max = torch.tensor(self.cfg.K_pos_tensor_max, device=self.env.device, requires_grad=False).expand(self.env.num_envs, -1)
        self.K_pos_tensor_min = torch.tensor(self.cfg.K_pos_tensor_min, device=self.env.device, requires_grad=False).expand(self.env.num_envs, -1)
        self.K_linvel_tensor_max = torch.tensor(self.cfg.K_vel_tensor_max, device=self.env.device, requires_grad=False).expand(self.env.num_envs, -1)
        self.K_linvel_tensor_min = torch.tensor(self.cfg.K_vel_tensor_min, device=self.env.device, requires_grad=False).expand(self.env.num_envs, -1)
        self.K_rot_tensor_max = torch.tensor(self.cfg.K_rot_tensor_max, device=self.env.device, requires_grad=False).expand(self.env.num_envs, -1)
        self.K_rot_tensor_min = torch.tensor(self.cfg.K_rot_tensor_min, device=self.env.device, requires_grad=False).expand(self.env.num_envs, -1)
        self.K_angvel_tensor_max = torch.tensor(self.cfg.K_angvel_tensor_max, device=self.env.device, requires_grad=False).expand(self.env.num_envs, -1)
        self.K_angvel_tensor_min = torch.tensor(self.cfg.K_angvel_tensor_min, device=self.env.device, requires_grad=False).expand(self.env.num_envs, -1)

        # Set the current values of the controller parameters
        self.K_pos_tensor_current = (self.K_pos_tensor_max + self.K_pos_tensor_min) / 2.0
        self.K_linvel_tensor_current = (self.K_linvel_tensor_max + self.K_linvel_tensor_min) / 2.0
        self.K_rot_tensor_current = (self.K_rot_tensor_max + self.K_rot_tensor_min) / 2.0
        self.K_angvel_tensor_current = (self.K_angvel_tensor_max + self.K_angvel_tensor_min) / 2.0

        # tensors that are needed later in the controller are predefined here
        self.accel = torch.zeros((self.env.num_envs, 3), device=self.env.device)
        self.wrench_command = torch.zeros((self.env.num_envs, 6), device=self.env.device)  # [fx, fy, fz, tx, ty, tz]
        # tensors that are needed later in the controller are predefined here
        self.desired_quat = torch.zeros_like(self.robot.data.root_quat_w)
        self.desired_body_angvel = torch.zeros_like(self.robot.data.root_ang_vel_b)
        self.euler_angle_rates = torch.zeros_like(self.robot.data.root_ang_vel_b)

        # buffer tensor to be used by torch.jit functions for various purposes
        self.buffer_tensor = torch.zeros((self.env.num_envs, 3, 3), device=self.env.device)

    def __call__(self, command_actions):
        self.wrench_command[:] = 0.0
        self.wrench_command[:, 2] = (command_actions[:, 0] + 1.0) * self.mass.squeeze(1) * torch.norm(self.gravity, dim=1)
        self.euler_angle_rates[:, :2] = 0.0
        self.euler_angle_rates[:, 2] = command_actions[:, 3]
        self.desired_body_angvel[:] = euler_rates_to_body_rates(self.robot_euler_angles, self.euler_angle_rates, self.buffer_tensor)

        # quaternion desired
        # desired euler angle is equal to commanded roll, commanded pitch, and current yaw
        quat_desired = math_utils.quat_from_euler_xyz(command_actions[:, 1], command_actions[:, 2], self.robot_euler_angles[:, 2])
        self.wrench_command[:, 3:6] = self.compute_body_torque(quat_desired, self.desired_body_angvel)

        return self.wrench_command

    def reset(self):
        self.reset_idx(env_ids=None)

    def reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs)
        self.randomize_params(env_ids)

    def randomize_params(self, env_ids):
        if self.cfg.randomize_params == False:
            return
        self.K_pos_tensor_current[env_ids] = torch_rand_float_tensor(self.K_pos_tensor_min[env_ids], self.K_pos_tensor_max[env_ids])
        self.K_linvel_tensor_current[env_ids] = torch_rand_float_tensor(self.K_linvel_tensor_min[env_ids], self.K_linvel_tensor_max[env_ids])
        self.K_rot_tensor_current[env_ids] = torch_rand_float_tensor(self.K_rot_tensor_min[env_ids], self.K_rot_tensor_max[env_ids])
        self.K_angvel_tensor_current[env_ids] = torch_rand_float_tensor(self.K_angvel_tensor_min[env_ids], self.K_angvel_tensor_max[env_ids])

    def compute_acceleration(self, setpoint_position, setpoint_velocity):
        position_error_world_frame = setpoint_position - self.robot.data.root_pos_w
        setpoint_velocity_world_frame = math_utils.quat_apply(math_utils.yaw_quat(self.robot.data.root_quat_w), setpoint_velocity)
        velocity_error = setpoint_velocity_world_frame - self.robot.data.root_vel_w
        accel_command = self.K_pos_tensor_current * position_error_world_frame + self.K_linvel_tensor_current * velocity_error
        return accel_command

    def compute_body_torque(self, setpoint_orientation, setpoint_angvel):
        setpoint_angvel[:, 2] = torch.clamp(setpoint_angvel[:, 2], -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)
        RT_Rd_quat = math_utils.quat_mul(math_utils.quat_inv(self.robot.data.root_quat_w), setpoint_orientation)      # (N,4) wxyz
        R_err = math_utils.matrix_from_quat(RT_Rd_quat)
        skew_matrix = R_err.transpose(-1, -2) - R_err
        rotation_error = 0.5 * torch.stack([-skew_matrix[:, 1, 2], skew_matrix[:, 0, 2], -skew_matrix[:, 0, 1]], dim=1)
        angvel_error = self.robot.data.root_ang_vel_b - math_utils.quat_apply(RT_Rd_quat, setpoint_angvel)
        feed_forward_body_rates = torch.cross(self.robot.data.root_ang_vel_b, torch.bmm(self.robot_inertia, self.robot.data.root_ang_vel_b.unsqueeze(2)).squeeze(2), dim=1)
        torque = -self.K_rot_tensor_current * rotation_error - self.K_angvel_tensor_current * angvel_error + feed_forward_body_rates
        return torque

@torch.jit.script
def euler_rates_to_body_rates(robot_euler_angles, desired_euler_rates, rotmat_euler_to_body_rates):
    s_pitch = torch.sin(robot_euler_angles[:, 1])
    c_pitch = torch.cos(robot_euler_angles[:, 1])

    s_roll = torch.sin(robot_euler_angles[:, 0])
    c_roll = torch.cos(robot_euler_angles[:, 0])

    rotmat_euler_to_body_rates[:, 0, 0] = 1.0
    rotmat_euler_to_body_rates[:, 1, 1] = c_roll
    rotmat_euler_to_body_rates[:, 0, 2] = -s_pitch
    rotmat_euler_to_body_rates[:, 2, 1] = -s_roll
    rotmat_euler_to_body_rates[:, 1, 2] = s_roll * c_pitch
    rotmat_euler_to_body_rates[:, 2, 2] = c_roll * c_pitch

    return torch.bmm(rotmat_euler_to_body_rates, desired_euler_rates.unsqueeze(2)).squeeze(2)
