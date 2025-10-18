

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal
import omni.usd
from isaaclab.assets import RigidObject,Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.actuators import DCMotor
from parkour_isaaclab.actuators import ParkourDCMotor
from isaaclab.sensors import RayCasterCamera
from isaaclab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import  ManagerBasedEnv
    from isaaclab.managers import EventTermCfg

def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)
    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_root_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset: float = 3.0
):
    asset: Articulation = env.scene[asset_cfg.name]
    terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    root_states = asset.data.default_root_state[env_ids].clone()
    origin = env.scene.env_origins[env_ids].clone()
    origin[:,-1] = 0
    positions = root_states[:, 0:3] + origin - \
        torch.tensor((terrain_gen_cfg.size[1] + offset, 0, 0)).to(env.device)
    asset.write_root_pose_to_sim(torch.cat([positions, root_states[:, 3:7]], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:13] , env_ids=env_ids) ## it mush need for init vel

def randomize_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float] | None = None,
    damping_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    def randomize(data: torch.Tensor, params: tuple[float, float]) -> torch.Tensor:
        return _randomize_prop_by_op(
            data, params, dim_0_ids=None, dim_1_ids=actuator_indices, operation=operation, distribution=distribution
        )

    for actuator in asset.actuators.values():
        if isinstance(asset_cfg.joint_ids, slice):
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                global_indices = slice(None)
            else:
                global_indices = torch.tensor(actuator.joint_indices, device=asset.device)
        elif isinstance(actuator.joint_indices, slice):
            global_indices = actuator_indices = torch.tensor(asset_cfg.joint_ids, device=asset.device)
        else:
            actuator_joint_indices = torch.tensor(actuator.joint_indices, device=asset.device)
            asset_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)
            # the indices of the joints in the actuator that have to be randomized
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
            if len(actuator_indices) == 0:
                continue
            global_indices = actuator_joint_indices[actuator_indices]
        if stiffness_distribution_params is not None:
            stiffness = actuator.stiffness[env_ids].clone()
            stiffness[:, actuator_indices] = asset.data.default_joint_stiffness[env_ids][:, global_indices].clone()
            randomize(stiffness, stiffness_distribution_params)
            actuator.stiffness[env_ids] = stiffness
            if isinstance(actuator, DCMotor) or isinstance(actuator, ParkourDCMotor):
                asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)
        # Randomize damping
        if damping_distribution_params is not None:
            damping = actuator.damping[env_ids].clone()
            damping[:, actuator_indices] = asset.data.default_joint_damping[env_ids][:, global_indices].clone()
            randomize(damping, damping_distribution_params)
            actuator.damping[env_ids] = damping
            if isinstance(actuator, DCMotor) or isinstance(actuator, ParkourDCMotor):
                asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)

def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()
    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples
    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)

def push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):  
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    vel_w = asset.data.root_vel_w[env_ids]
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    random_noise = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    vel_w[:,:2] = random_noise[:,:2]
    vel_w[:,2:] += random_noise[:,2:]
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

def random_camera_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    pos_noise_range: dict[str,tuple[float,float]] | None = None,
    rot_noise_range: dict[str,tuple[float,float]] | None = None,
    convention: str = 'ros',
):
    """
    prestartup
    """
    camera_sensor: RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    init_rot = torch.tensor(camera_sensor.cfg.offset.rot).repeat(env.num_envs,1).to(env.device)

    if pos_noise_range is not None: 
        pos_range_list = [pos_noise_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        pos_ranges = torch.tensor(pos_range_list, device=env.device)
        random_pose = math_utils.sample_uniform(pos_ranges[:,0], pos_ranges[:,1], (env.num_envs,1), device=env.device)
    else:
        random_pose = None
    if rot_noise_range is not None:
        rot_range_list = [rot_noise_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
        rot_ranges = torch.deg2rad(torch.tensor(rot_range_list)).to(env.device)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(init_rot)
        init_rot = torch.stack([roll, pitch, yaw], dim=-1).to(env.device)
        init_rot += math_utils.sample_uniform(rot_ranges[:,0], rot_ranges[:,1], (env.num_envs,1), device=env.device)
        random_rot = math_utils.quat_from_euler_xyz(init_rot[:,0],init_rot[:,1],init_rot[:,2])
    else:
        random_rot = init_rot 

    camera_sensor.set_world_poses(
        positions=random_pose,
        orientations=random_rot,
        convention=convention,
        env_ids=torch.arange(env.num_envs, dtype=torch.int64, device=env.device),
    )
    
class randomize_rigid_body_material(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )
        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            self.num_shapes_per_body = None

        # obtain parameters for sampling friction and restitution values
        friction_range = cfg.params.get("friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.,0.))
        num_buckets = int(cfg.params.get("num_buckets", 1))
        range_list = [friction_range, (0,0), restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")
        self.material_buckets[:,1] = self.material_buckets[:,0]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        friction_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        bucket_ids = torch.randint(0, num_buckets, (len(env_ids),), device="cpu")
        material_samples = self.material_buckets[bucket_ids]
        total_num_shapes = self.asset.root_physx_view.max_shapes
        material_samples = material_samples.unsqueeze(1).repeat(1,total_num_shapes,1)
        # retrieve material buffer from the physics simulation
        materials = self.asset.root_physx_view.get_material_properties()
        # update material buffer with new samples
        if self.num_shapes_per_body is not None:
            # sample material properties from the given ranges
            for body_id in self.asset_cfg.body_ids:
                # obtain indices of shapes for the body
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                # assign the new materials
                # material samples are of shape: num_env_ids x total_num_shapes 
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            # assign all the materials
            materials[env_ids] = material_samples[:]

        # apply to simulation
        self.asset.root_physx_view.set_material_properties(materials, env_ids)
