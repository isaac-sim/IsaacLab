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
        self.mass, self.robot_inertia, _ = aggregate_inertia_about_robot_com(self.robot.root_physx_view, self.env.device)
        self.gravity = torch.tensor(self.cfg.gravity, device=self.env.device).expand(self.env.num_envs, -1)

        # Read from config and set the values for controller parameters
        self.K_pos_tensor_max = torch.tensor(self.cfg.K_pos_tensor_max, device=self.env.device).expand(self.env.num_envs, -1)
        self.K_pos_tensor_min = torch.tensor(self.cfg.K_pos_tensor_min, device=self.env.device).expand(self.env.num_envs, -1)
        self.K_linvel_tensor_max = torch.tensor(self.cfg.K_vel_tensor_max, device=self.env.device).expand(self.env.num_envs, -1)
        self.K_linvel_tensor_min = torch.tensor(self.cfg.K_vel_tensor_min, device=self.env.device).expand(self.env.num_envs, -1)
        self.K_rot_tensor_max = torch.tensor(self.cfg.K_rot_tensor_max, device=self.env.device).expand(self.env.num_envs, -1)
        self.K_rot_tensor_min = torch.tensor(self.cfg.K_rot_tensor_min, device=self.env.device).expand(self.env.num_envs, -1)
        self.K_angvel_tensor_max = torch.tensor(self.cfg.K_angvel_tensor_max, device=self.env.device).expand(self.env.num_envs, -1)
        self.K_angvel_tensor_min = torch.tensor(self.cfg.K_angvel_tensor_min, device=self.env.device).expand(self.env.num_envs, -1)

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
        robot_euler_angles = torch.stack(math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w), dim=-1)
        robot_euler_angles = math_utils.wrap_to_pi(robot_euler_angles)
        self.wrench_command[:] = 0.0
        self.wrench_command[:, 2] = (command_actions[:, 0] + 1.0) * self.mass * torch.norm(self.gravity, dim=1)
        self.euler_angle_rates[:, :2] = 0.0
        self.euler_angle_rates[:, 2] = command_actions[:, 3]
        self.desired_body_angvel[:] = euler_rates_to_body_rates(robot_euler_angles, self.euler_angle_rates, self.buffer_tensor)

        # quaternion desired
        # desired euler angle is equal to commanded roll, commanded pitch, and current yaw
        quat_desired = math_utils.quat_from_euler_xyz(command_actions[:, 1], command_actions[:, 2], robot_euler_angles[:, 2])
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
        velocity_error = setpoint_velocity_world_frame - self.robot.data.root_lin_vel_w
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


@torch.no_grad()
def aggregate_inertia_about_robot_com(root_view, device, eps=1e-12):
    """
    root_view: ArticulationView

    Returns:
      total_mass: (E,)
      I_total:    (E,3,3) inertia about robot COM, expressed in world axes
      com_robot:  (E,3)
    """
    body_pose_w = root_view.get_link_transforms().to(device)
    body_pos_w = body_pose_w[..., :3]
    body_quat_w = math_utils.convert_quat(body_pose_w[..., 3:7], to='wxyz')

    # Inertia in mass frame (local to COM) -> (E,B,3,3)
    I_local_any = root_view.get_inertias().to(device)
    E, B, _ = I_local_any.shape
    I_local = I_local_any.view(E, B, 3, 3)

    # COM local pose (massLocalPose): [x,y,z,qx,qy,qz,qw]
    com_local_pose = root_view.get_coms().to(device)
    q_mass_wxyz = math_utils.convert_quat(com_local_pose[..., 3:7], to='wxyz')

    # Masses
    inv_m = root_view.get_inv_masses().to(device)
    m = torch.where(inv_m > 0, 1.0 / inv_m, torch.zeros_like(inv_m))
    m_sum = m.sum(dim=1, keepdim=True)
    valid = (m > 0).float().unsqueeze(-1)

    # World COM of each link: p_link + body_rot_matrix * com_pos_local
    body_rot_matrix = math_utils.matrix_from_quat(body_quat_w)
    com_world = body_pos_w + (body_rot_matrix @ com_local_pose[..., :3][..., :, None]).squeeze(-1)

    # Robot COM (mass-weighted)
    com_robot = (m.unsqueeze(-1) * com_world).sum(dim=1) / (m_sum + eps)

    # Rotate inertia from mass frame to world: R = body_rot_matrix * R_mass
    R_mass = math_utils.matrix_from_quat(q_mass_wxyz)
    R = body_rot_matrix @ R_mass
    I_world = R @ I_local @ R.transpose(-1, -2)

    # Parallel-axis to robot COM
    r = com_world - com_robot[:, None, :]
    rrT = r[..., :, None] @ r[..., None, :]
    r2 = (r * r).sum(dim=-1, keepdim=True)
    I3 = torch.eye(3, device=device).reshape(1,1,3,3).expand(E, B, 3, 3)
    I_pa = m[..., None, None] * (r2[..., None] * I3 - rrT)

    # Sum over links (ignore zero-mass pads)
    I_total = ((I_world + I_pa) * valid[..., None]).sum(dim=1)
    I_total = 0.5 * (I_total + I_total.transpose(-1, -2))
    total_mass = m.sum(dim=1)

    return total_mass, I_total, com_robot