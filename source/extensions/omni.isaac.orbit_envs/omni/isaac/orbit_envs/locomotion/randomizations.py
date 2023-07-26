import torch

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

from omni.isaac.orbit.utils.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
    from omni.isaac.orbit_envs.locomotion.velocity.locomotion_env import LocomotionEnv


def physics_material(
    env: "LocomotionEnv",
    env_ids,
    body_ids: list,
    static_friction_range,
    dynamic_friction_range,
    restitution_range,
    num_buckets,
):
    body_view = env.robot._body_view
    # get bucketed materials
    material_buckets = torch.zeros(num_buckets, 3)
    material_buckets[:, 0].uniform_(*static_friction_range)
    material_buckets[:, 1].uniform_(*dynamic_friction_range)
    material_buckets[:, 2].uniform_(*restitution_range)

    material_ids = torch.randint(0, num_buckets, (body_view.count, body_view.num_shapes))

    materials = material_buckets[material_ids]

    # create global body indices from env_ids and env_body_ids
    if env_ids is None:
        env_ids = torch.arange(env.num_envs)
    bodies_per_env = body_view.count // env.num_envs
    indices = torch.tensor(body_ids, dtype=torch.int).repeat(len(env_ids), 1)
    indices += env_ids.unsqueeze(1) * bodies_per_env

    body_view._physics_view.set_material_properties(
        materials, indices
    )  # Must be CPU tensors right now. TODO check if changes in new release


def add_body_mass(env: "LocomotionEnv", env_ids, body_ids: list, mass_range):
    body_view = env.robot._body_view

    masses = body_view._physics_view.get_masses()
    masses += sample_uniform(*mass_range, masses.shape, device=masses.device)

    # create global body indices from env_ids and env_body_ids
    if env_ids is None:
        env_ids = torch.arange(env.num_envs)
    bodies_per_env = body_view.count // env.num_envs
    indices = torch.tensor(body_ids, dtype=torch.int).repeat(len(env_ids), 1)
    indices += env_ids.unsqueeze(1) * bodies_per_env

    body_view._physics_view.set_masses(
        masses, indices
    )  # Must be CPU tensors right now. TODO check if changes in new release


def apply_external_force_torqe(env: "LocomotionEnv", env_ids: torch.Tensor, body_ids: list, force_range, torque_range):
    body_count = env.num_envs * len(body_ids)
    forces = sample_uniform(*force_range, (body_count, 3), env.device)
    torques = sample_uniform(*torque_range, (body_count, 3), env.device)
    env.robot.set_external_force_and_torque(forces, torques, env_ids, body_ids)


def push_robot(env: "LocomotionEnv", env_ids, velocity_range):
    velocities = env.robot.articulations.get_velocities(env_ids, clone=False)
    velocities[:, 0].uniform_(*velocity_range.get("x", (0.0, 0.0)))
    velocities[:, 1].uniform_(*velocity_range.get("y", (0.0, 0.0)))
    velocities[:, 2].uniform_(*velocity_range.get("z", (0.0, 0.0)))
    velocities[:, 3].uniform_(*velocity_range.get("roll", (0.0, 0.0)))
    velocities[:, 4].uniform_(*velocity_range.get("pitch", (0.0, 0.0)))
    velocities[:, 5].uniform_(*velocity_range.get("yaw", (0.0, 0.0)))

    env.robot.articulations.set_velocities(velocities, env_ids)


def reset_robot_root(env: "LocomotionEnv", env_ids, pose_range, velocity_range):
    # pose
    root_states = env.robot.get_default_root_state(env_ids, clone=False)
    pos_offset = torch.zeros_like(root_states[:, 0:3])
    pos_offset[:, 0].uniform_(*pose_range.get("x", (0.0, 0.0)))
    pos_offset[:, 1].uniform_(*pose_range.get("y", (0.0, 0.0)))
    pos_offset[:, 2].uniform_(*pose_range.get("z", (0.0, 0.0)))
    positions = root_states[:, 0:3] + env.envs_positions[env_ids] + pos_offset

    euler_angles = torch.zeros_like(positions)
    euler_angles[:, 0].uniform_(*pose_range.get("roll", (0.0, 0.0)))
    euler_angles[:, 1].uniform_(*pose_range.get("pitch", (0.0, 0.0)))
    euler_angles[:, 2].uniform_(*pose_range.get("yaw", (0.0, 0.0)))

    orientations = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])

    env.robot.articulations.set_world_poses(positions, orientations, env_ids)
    # velocities
    velocities = root_states[:, 7:13]
    velocities[:, 0].uniform_(*velocity_range.get("x", (0.0, 0.0)))
    velocities[:, 1].uniform_(*velocity_range.get("y", (0.0, 0.0)))
    velocities[:, 2].uniform_(*velocity_range.get("z", (0.0, 0.0)))
    velocities[:, 3].uniform_(*velocity_range.get("roll", (0.0, 0.0)))
    velocities[:, 4].uniform_(*velocity_range.get("pitch", (0.0, 0.0)))
    velocities[:, 5].uniform_(*velocity_range.get("yaw", (0.0, 0.0)))
    env.robot.articulations.set_velocities(velocities, env_ids)


def reset_robot_joints_scale_defaults(env: "LocomotionEnv", env_ids, position_range, velocity_range):
    """
    Reset the robot joints to a random position and velocity within the given ranges.
    Scales the default position and velocity by the given ranges.
    """
    positions = env.robot.data.default_dof_pos[env_ids]
    velocities = env.robot.data.default_dof_vel[env_ids]
    positions *= sample_uniform(*position_range, positions.shape, positions.device)
    velocities *= sample_uniform(*velocity_range, velocities.shape, velocities.device)

    env.robot.set_dof_state(positions, velocities, env_ids)
