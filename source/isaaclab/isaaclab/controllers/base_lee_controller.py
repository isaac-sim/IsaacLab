import torch

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("base_lee_controller")

logger.setLevel("DEBUG")


import pytorch3d.transforms as p3d_transforms

from aerial_gym.control.controllers.base_controller import *


class BaseLeeController(BaseController):
    """
    This class will operate as the base class for all controllers.
    It will be inherited by the specific controller classes.
    """

    def __init__(self, control_config, num_envs, device, mode="robot"):
        super().__init__(control_config, num_envs, device, mode)
        self.cfg = control_config
        self.num_envs = num_envs
        self.device = device

    def init_tensors(self, global_tensor_dict):
        super().init_tensors(global_tensor_dict)

        # Read from config and set the values for controller parameters
        self.K_pos_tensor_max = torch.tensor(
            self.cfg.K_pos_tensor_max, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.K_pos_tensor_min = torch.tensor(
            self.cfg.K_pos_tensor_min, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.K_linvel_tensor_max = torch.tensor(
            self.cfg.K_vel_tensor_max, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.K_linvel_tensor_min = torch.tensor(
            self.cfg.K_vel_tensor_min, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.K_rot_tensor_max = torch.tensor(
            self.cfg.K_rot_tensor_max, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.K_rot_tensor_min = torch.tensor(
            self.cfg.K_rot_tensor_min, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.K_angvel_tensor_max = torch.tensor(
            self.cfg.K_angvel_tensor_max, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)
        self.K_angvel_tensor_min = torch.tensor(
            self.cfg.K_angvel_tensor_min, device=self.device, requires_grad=False
        ).expand(self.num_envs, -1)

        # Set the current values of the controller parameters
        self.K_pos_tensor_current = (self.K_pos_tensor_max + self.K_pos_tensor_min) / 2.0
        self.K_linvel_tensor_current = (self.K_linvel_tensor_max + self.K_linvel_tensor_min) / 2.0
        self.K_rot_tensor_current = (self.K_rot_tensor_max + self.K_rot_tensor_min) / 2.0
        self.K_angvel_tensor_current = (self.K_angvel_tensor_max + self.K_angvel_tensor_min) / 2.0

        # tensors that are needed later in the controller are predefined here
        self.accel = torch.zeros((self.num_envs, 3), device=self.device)
        self.wrench_command = torch.zeros(
            (self.num_envs, 6), device=self.device
        )  # [fx, fy, fz, tx, ty, tz]

        # tensors that are needed later in the controller are predefined here
        self.desired_quat = torch.zeros_like(self.robot_orientation)
        self.desired_body_angvel = torch.zeros_like(self.robot_body_angvel)
        self.euler_angle_rates = torch.zeros_like(self.robot_body_angvel)

        # buffer tensor to be used by torch.jit functions for various purposes
        self.buffer_tensor = torch.zeros((self.num_envs, 3, 3), device=self.device)

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def reset_commands(self):
        self.wrench_command[:] = 0.0

    def reset(self):
        self.reset_idx(env_ids=None)

    def reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.K_rot_tensor.shape[0])
        self.randomize_params(env_ids)

    def randomize_params(self, env_ids):
        if self.cfg.randomize_params == False:
            # logger.debug(
            #     "Randomization of controller parameters is disabled based on config setting."
            # )
            return
        self.K_pos_tensor_current[env_ids] = torch_rand_float_tensor(
            self.K_pos_tensor_min[env_ids], self.K_pos_tensor_max[env_ids]
        )
        self.K_linvel_tensor_current[env_ids] = torch_rand_float_tensor(
            self.K_linvel_tensor_min[env_ids], self.K_linvel_tensor_max[env_ids]
        )
        self.K_rot_tensor_current[env_ids] = torch_rand_float_tensor(
            self.K_rot_tensor_min[env_ids], self.K_rot_tensor_max[env_ids]
        )
        self.K_angvel_tensor_current[env_ids] = torch_rand_float_tensor(
            self.K_angvel_tensor_min[env_ids], self.K_angvel_tensor_max[env_ids]
        )

    def compute_acceleration(self, setpoint_position, setpoint_velocity):
        position_error_world_frame = setpoint_position - self.robot_position
        # logger.debug(
        #     f"position_error_world_frame: {position_error_world_frame}, setpoint_position: {setpoint_position}, robot_position: {self.robot_position}"
        # )
        setpoint_velocity_world_frame = quat_rotate(
            self.robot_vehicle_orientation, setpoint_velocity
        )
        velocity_error = setpoint_velocity_world_frame - self.robot_linvel

        accel_command = (
            self.K_pos_tensor_current * position_error_world_frame
            + self.K_linvel_tensor_current * velocity_error
        )
        return accel_command

    def compute_body_torque(self, setpoint_orientation, setpoint_angvel):
        setpoint_angvel[:, 2] = torch.clamp(
            setpoint_angvel[:, 2], -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate
        )
        RT_Rd_quat = quat_mul(quat_inverse(self.robot_orientation), setpoint_orientation)
        RT_Rd = quat_to_rotation_matrix(RT_Rd_quat)
        rotation_error = 0.5 * compute_vee_map(torch.transpose(RT_Rd, -2, -1) - RT_Rd)
        angvel_error = self.robot_body_angvel - quat_rotate(RT_Rd_quat, setpoint_angvel)
        feed_forward_body_rates = torch.cross(
            self.robot_body_angvel,
            torch.bmm(self.robot_inertia, self.robot_body_angvel.unsqueeze(2)).squeeze(2),
            dim=1,
        )
        torque = (
            -self.K_rot_tensor_current * rotation_error
            - self.K_angvel_tensor_current * angvel_error
            + feed_forward_body_rates
        )
        return torque


@torch.jit.script
def calculate_desired_orientation_from_forces_and_yaw(forces_command, yaw_setpoint):
    c_phi_s_theta = forces_command[:, 0]
    s_phi = -forces_command[:, 1]
    c_phi_c_theta = forces_command[:, 2]

    # Calculate orientation setpoint
    pitch_setpoint = torch.atan2(c_phi_s_theta, c_phi_c_theta)
    roll_setpoint = torch.atan2(s_phi, torch.sqrt(c_phi_c_theta**2 + c_phi_s_theta**2))
    quat_desired = quat_from_euler_xyz_tensor(
        torch.stack((roll_setpoint, pitch_setpoint, yaw_setpoint), dim=1)
    )
    return quat_desired


# @torch.jit.script
def calculate_desired_orientation_for_position_velocity_control(
    forces_command, yaw_setpoint, rotation_matrix_desired
):
    b3_c = torch.div(forces_command, torch.norm(forces_command, dim=1).unsqueeze(1))
    temp_dir = torch.zeros_like(forces_command)
    temp_dir[:, 0] = torch.cos(yaw_setpoint)
    temp_dir[:, 1] = torch.sin(yaw_setpoint)

    b2_c = torch.cross(b3_c, temp_dir, dim=1)
    b2_c = torch.div(b2_c, torch.norm(b2_c, dim=1).unsqueeze(1))
    b1_c = torch.cross(b2_c, b3_c, dim=1)

    rotation_matrix_desired[:, :, 0] = b1_c
    rotation_matrix_desired[:, :, 1] = b2_c
    rotation_matrix_desired[:, :, 2] = b3_c
    q = p3d_transforms.matrix_to_quaternion(rotation_matrix_desired)
    quat_desired = torch.stack((q[:, 1], q[:, 2], q[:, 3], q[:, 0]), dim=1)

    sign_qw = torch.sign(quat_desired[:, 3])
    # quat_desired = quat_desired * sign_qw.unsqueeze(1)

    return quat_desired


# quat_from_rotation_matrix(rotation_matrix_desired)


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
