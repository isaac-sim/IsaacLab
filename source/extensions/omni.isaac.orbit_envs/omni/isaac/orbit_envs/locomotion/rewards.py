import torch

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from omni.isaac.orbit_envs.locomotion.velocity.locomotion_env import LocomotionEnv


def lin_vel_z_l2(env: "LocomotionEnv"):
    """Penalize z-axis base linear velocity using L2-kernel."""
    return torch.square(env.robot.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: "LocomotionEnv"):
    """Penalize xy-axii base angular velocity using L2-kernel."""
    return torch.sum(torch.square(env.robot.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: "LocomotionEnv"):
    """Penalize non-float base orientation."""
    return torch.sum(torch.square(env.robot.data.projected_gravity_b[:, :2]), dim=1)


def dof_torques_l2(env: "LocomotionEnv"):
    """Penalize torques applied on the robot."""
    return torch.sum(torch.square(env.robot.data.applied_torques), dim=1)


def dof_vel_l2(env: "LocomotionEnv"):
    """Penalize dof velocities on the robot."""
    return torch.sum(torch.square(env.robot.data.dof_vel), dim=1)


def dof_acc_l2(env: "LocomotionEnv"):
    """Penalize dof accelerations on the robot."""
    return torch.sum(torch.square(env.robot.data.dof_acc), dim=1)


def dof_pos_limits(env: "LocomotionEnv"):
    """Penalize dof positions too close to the limit."""
    out_of_limits = -(env.robot.data.dof_pos - env.robot.data.soft_dof_pos_limits[..., 0]).clip(max=0.0)
    out_of_limits += (env.robot.data.dof_pos - env.robot.data.soft_dof_pos_limits[..., 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def dof_vel_limits(env: "LocomotionEnv", soft_ratio: float):
    """Penalize dof velocities too close to the limit.

    Args:
        soft_ratio (float): Defines the soft limit as a percentage of the hard limit.
    """
    out_of_limits = torch.abs(env.robot.data.dof_vel) - env.robot.data.soft_dof_vel_limits * soft_ratio
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: "LocomotionEnv"):
    """Penalize changes in actions."""
    return torch.sum(torch.square(env.previous_actions - env.actions), dim=1)


def applied_torque_limits(env: "LocomotionEnv"):
    """Penalize applied torques that are too close to the actuator limits."""
    out_of_limits = torch.abs(env.robot.data.applied_torques - env.robot.data.computed_torques)
    return torch.sum(out_of_limits, dim=1)


def base_height_l2(env: "LocomotionEnv", target_height: float):
    """Penalize base height from its target."""
    # TODO: Fix this for rough-terrain.
    base_height = env.robot.data.root_pos_w[:, 2]
    return torch.square(base_height - target_height)


def track_lin_vel_xy_exp(env: "LocomotionEnv", scale: float):
    """Tracking of linear velocity commands (xy axes).

    Args:
        std (float): Defines the width of the bell-curve.
    """
    lin_vel_error = torch.sum(
        torch.square(env._command_manager.command[:, :2] - env.robot.data.root_lin_vel_b[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / scale)


def track_ang_vel_z_exp(env: "LocomotionEnv", scale):
    """Tracking of angular velocity commands (yaw).

    Args:
        std (float): Defines the width of the bell-curve.
    """
    ang_vel_error = torch.square(env._command_manager.command[:, 2] - env.robot.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / scale)


def feet_air_time(env: "LocomotionEnv", time_threshold: float, sensor_name: str):
    """Reward long steps taken by the feet."""
    last_air_time = env.sensors[sensor_name].data.last_air_time
    first_contact = last_air_time > 0.0
    reward = torch.sum((last_air_time - time_threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env._command_manager.command[:, :2], dim=1) > 0.1
    return reward


def undesired_contacts(env: "LocomotionEnv", body_ids: List[int], threshold: float, sensor_name: str):
    """Penalize undesired contacts."""
    # check if contact force is above threshold
    net_contact_forces = env.sensors[sensor_name].data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def contact_forces(env: "LocomotionEnv", body_ids: List[int], max_contact_force: float, sensor_name: str):
    # net_contact_forces = env._foot_contact.get_net_contact_forces(dt=env.cfg.sim.dt, clone=False).view(-1, 4, 3)
    net_contact_forces = env.sensors[sensor_name].data.net_forces_w_history
    return torch.sum(
        (torch.max(torch.norm(net_contact_forces[:, :, body_ids], dim=-1), dim=1)[0] - max_contact_force).clip(min=0.0),
        dim=1,
    )
