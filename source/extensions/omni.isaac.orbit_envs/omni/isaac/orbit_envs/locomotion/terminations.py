import torch

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from omni.isaac.orbit_envs.locomotion.velocity.locomotion_env import LocomotionEnv


def time_out(env: "LocomotionEnv"):
    return env.episode_length_buf >= env.max_episode_length


def command_resample(env: "LocomotionEnv", num_commands):
    return torch.logical_and((env.command_time_left <= 0.0), (env.num_commands == num_commands))


def illegal_contact(env: "LocomotionEnv", body_ids: List[int], force_threshold, sensor_name: str):
    net_contact_forces = env.sensors[sensor_name].data.net_forces_w_history
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, body_ids], dim=-1), dim=1)[0] > force_threshold, dim=1
    )


def bad_orientation(env: "LocomotionEnv", limit_angle):
    return torch.acos(-env.robot.data.projected_gravity_b[:, 2]).abs() > limit_angle


def base_height(env: "LocomotionEnv", minimum_height):
    return env.robot.data.root_pos_w[:, 2] < minimum_height


def dof_torque_limit(env: "LocomotionEnv"):
    return torch.any(torch.isclose(env.robot.data.computed_torques, env.robot.data.applied_torques), dim=1)


def dof_velocity_limit(env: "LocomotionEnv", max_velocity):
    # TODO read max velocities per joint from robot
    return torch.any(env.robot.data.dof_vel > max_velocity, dim=1)


def dof_pos_limit(env: "LocomotionEnv"):
    out_of_limits = -(env.robot.data.dof_pos - env.robot.data.soft_dof_pos_limits[:, 0]).clip(max=0.0) + (
        env.robot.data.dof_pos - env.robot.data.soft_dof_pos_limits[:, 1]
    ).clip(min=0.0)
    return torch.any(out_of_limits > 1.0e-6, dim=1)
