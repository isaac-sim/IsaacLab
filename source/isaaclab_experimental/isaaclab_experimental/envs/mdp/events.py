# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import torch
from typing import TYPE_CHECKING, Literal

import warp as wp
from newton.solvers import SolverNotifyFlags

import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.warp.utils import resolve_asset_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


#def randomize_rigid_body_scale(
#    env: ManagerBasedEnv,
#    env_ids: torch.Tensor | None,
#    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
#    asset_cfg: SceneEntityCfg,
#    relative_child_path: str | None = None,
#):
#    raise NotImplementedError("Not implemented")


#class randomize_rigid_body_material(ManagerTermBase):
#    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
#        """Initialize the term.
#
#        Args:
#            cfg: The configuration of the event term.
#            env: The environment instance.
#
#        Raises:
#            ValueError: If the asset is not an Articulation.
#        """
#        super().__init__(cfg, env)
#
#        # extract the used quantities (to enable type-hinting)
#        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
#        self.asset: Articulation = env.scene[self.asset_cfg.name]
#
#        if not isinstance(self.asset, (Articulation)):
#            raise ValueError(
#                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
#                f" with type: '{type(self.asset)}'."
#            )
#
#        # compute prefix scan for efficient indexing (shape counts come from Articulation)
#        self.shape_start_indices = None
#        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
#            # get shapes per body from Articulation class
#            num_shapes_per_body = self.asset.num_shapes_per_body
#
#            # compute prefix scan for faster lookup during randomization
#            self.shape_start_indices = [0]  # First body starts at index 0
#            for i in range(len(num_shapes_per_body) - 1):
#                self.shape_start_indices.append(self.shape_start_indices[-1] + num_shapes_per_body[i])
#
#        # cache default material properties for consistent randomization baseline
#        self.default_material_mu = None
#
#        # cache mask tensor for efficient reuse
#        self.env_mask = torch.zeros((env.scene.num_envs,), dtype=torch.bool, device=env.device)
#
#    def __call__(
#        self,
#        env: ManagerBasedEnv,
#        env_ids: torch.Tensor | None,
#        static_friction_range: tuple[float, float],
#        dynamic_friction_range: tuple[float, float],
#        restitution_range: tuple[float, float],
#        num_buckets: int,
#        asset_cfg: SceneEntityCfg,
#        make_consistent: bool = False,
#    ):
#        """Randomize the material properties of rigid bodies.
#
#        This function randomizes the friction coefficient of rigid body shapes. Due to Newton physics
#        (mjwarp) limitations, only the static friction coefficient is actually applied to the simulation.
#
#        Args:
#            env: The environment instance.
#            env_ids: The environment indices to apply the randomization to. If None, all environments are used.
#            static_friction_range: Range for static friction coefficient. This is the only parameter that
#                actually affects the simulation in Newton physics.
#            dynamic_friction_range: Range for dynamic friction coefficient. **NOT USED** - Newton physics
#                only supports a single friction coefficient per shape.
#            restitution_range: Range for restitution coefficient. **NOT USED** - Newton physics does not
#                currently support restitution randomization through this interface.
#            num_buckets: Number of material buckets. **NOT USED** - materials are sampled dynamically.
#            asset_cfg: The asset configuration specifying which asset and bodies to randomize.
#            make_consistent: Whether to ensure dynamic friction <= static friction. **NOT USED** - only
#                static friction is applied.
#
#        Note:
#            Newton physics only supports setting a single friction coefficient (mu) per shape. The dynamic_friction_range and restitution_range parameters
#            are kept for API consistency but do not affect the simulation.
#        """
#
#        # resolve environment ids
#        if env_ids is None:
#            env_ids = torch.arange(env.scene.num_envs, device="cpu")
#        else:
#            env_ids = env_ids.cpu()
#
#        # env_ids is guaranteed to be a tensor at this point
#        num_env_ids = env_ids.shape[0]
#
#        # cache default material properties on first call for consistent randomization baseline
#        if self.default_material_mu is None:
#            self.default_material_mu = wp.to_torch(
#                self.asset.root_newton_view.get_attribute("shape_material_mu", self.asset.root_newton_model)
#            ).clone()
#
#        # start with default values and clone for safe modification
#        material_mu = self.default_material_mu.clone()
#
#        # sample friction coefficients dynamically for each call
#        # Newton physics uses a single friction coefficient (mu) per shape
#        # Note: Only static_friction_range is used; dynamic_friction_range and restitution_range are ignored
#        total_num_shapes = material_mu.shape[1]
#
#        # sample friction values directly for the environments and shapes that need updating
#        if self.shape_start_indices is not None:
#            # sample friction coefficients for specific body shapes using efficient indexing
#            for body_id in self.asset_cfg.body_ids:
#                # use precomputed indices for fast lookup
#                start_idx = self.shape_start_indices[body_id]
#                end_idx = start_idx + self.asset.num_shapes_per_body[body_id]
#                num_shapes_in_body = end_idx - start_idx
#
#                # sample friction coefficients using only static_friction_range
#                friction_samples = math_utils.sample_uniform(
#                    static_friction_range[0],
#                    static_friction_range[1],
#                    (num_env_ids, num_shapes_in_body),
#                    device=self.asset.device,
#                )
#
#                # assign the new friction coefficients
#                material_mu[env_ids, start_idx:end_idx] = friction_samples
#        else:
#            # sample friction coefficients for all shapes
#            friction_samples = math_utils.sample_uniform(
#                static_friction_range[0],
#                static_friction_range[1],
#                (num_env_ids, total_num_shapes),
#                device=self.asset.device,
#            )
#
#            # assign all the friction coefficients
#            material_mu[env_ids, :] = friction_samples
#
#        # apply to simulation using cached mask
#        self.env_mask.fill_(False)  # reset all to False
#        self.env_mask[env_ids] = True
#        self.asset.root_newton_view.set_attribute(
#            "shape_material_mu", self.asset.root_newton_model, wp.from_torch(material_mu), mask=self.env_mask
#        )
#        NewtonManager._solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)


#def randomize_rigid_body_mass(
#    env: ManagerBasedEnv,
#    env_ids: torch.Tensor | None,
#    asset_cfg: SceneEntityCfg,
#    mass_distribution_params: tuple[float, float],
#    operation: Literal["add", "scale", "abs"],
#    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
#    recompute_inertia: bool = True,
#):
#    """Randomize the mass of the bodies by adding, scaling, or setting random values.
#
#    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
#    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.
#
#    If the ``recompute_inertia`` flag is set to ``True``, the function recomputes the inertia tensor of the bodies
#    after setting the mass. This is useful when the mass is changed significantly, as the inertia tensor depends
#    on the mass. It assumes the body is a uniform density object. If the body is not a uniform density object,
#    the inertia tensor may not be accurate.
#
#    .. tip::
#        This function uses CPU tensors to assign the body masses. It is recommended to use this function
#        only during the initialization of the environment.
#    """
#    # extract the used quantities (to enable type-hinting)
#    asset: Articulation = env.scene[asset_cfg.name]
#
#    # resolve environment ids
#    if env_ids is None:
#        env_ids = torch.arange(env.scene.num_envs, device="cpu")
#    else:
#        env_ids = env_ids.cpu()
#
#    # resolve body indices
#    if asset_cfg.body_ids == slice(None):
#        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
#    else:
#        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
#
#    # get the current masses of the bodies (num_assets, num_bodies)
#    masses = wp.to_torch(asset.root_newton_view.get_attribute("body_mass", asset.root_newton_model)).clone()
#    # apply randomization on default values
#    # this is to make sure when calling the function multiple times, the randomization is applied on the
#    # default values and not the previously randomized values
#    masses[env_ids[:, None], body_ids] = asset.data.default_mass[env_ids[:, None], body_ids].clone()
#
#    # sample from the given range
#    # note: we modify the masses in-place for all environments
#    #   however, the setter takes care that only the masses of the specified environments are modified
#    masses = _randomize_prop_by_op(
#        masses, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
#    )
#
#    # set the mass into the physics simulation
#    mask = torch.zeros((env.scene.num_envs,), dtype=torch.bool, device=env.device)
#    mask[env_ids] = True
#    asset.root_newton_view.set_attribute("body_mass", asset.root_newton_model, wp.from_torch(masses), mask=mask)
#
#    # recompute inertia tensors if needed
#    if recompute_inertia:
#        # compute the ratios of the new masses to the initial masses
#        ratios = masses[env_ids[:, None], body_ids] / asset.data.default_mass[env_ids[:, None], body_ids]
#        # scale the inertia tensors by the the ratios
#        # since mass randomization is done on default values, we can use the default inertia tensors
#        inertias = (
#            wp.to_torch(asset.root_newton_view.get_attribute("body_inertia", asset.root_newton_model))
#            .clone()
#            .reshape(env.scene.num_envs, asset.num_bodies, 9)
#        )
#        if isinstance(asset, Articulation):
#            # inertia has shape: (num_envs, num_bodies, 9) for articulation
#            inertias[env_ids[:, None], body_ids] = (
#                asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
#            )
#        else:
#            # inertia has shape: (num_envs, 9) for rigid object
#            inertias[env_ids] = asset.data.default_inertia[env_ids] * ratios
#        # set the inertia tensors into the physics simulation
#        asset.root_newton_view.set_attribute(
#            "body_inertia",
#            asset.root_newton_model,
#            wp.from_torch(
#                inertias.reshape(env.scene.num_envs, asset.num_bodies, 3, 3), dtype=wp.mat33, requires_grad=False
#            ),
#            mask=mask,
#        )
#
#        NewtonManager._solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)


#def randomize_rigid_body_com(
#    env: ManagerBasedEnv,
#    env_ids: torch.Tensor | None,
#    com_range: dict[str, tuple[float, float]],
#    asset_cfg: SceneEntityCfg,
#):
#    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.
#
#    .. note::
#        This function uses CPU tensors to assign the CoM. It is recommended to use this function
#        only during the initialization of the environment.
#    """
#    # extract the used quantities (to enable type-hinting)
#    asset: Articulation = env.scene[asset_cfg.name]
#    # resolve environment ids
#    if env_ids is None:
#        env_ids = torch.arange(env.scene.num_envs, device=env.device)
#
#    # resolve body indices
#    if asset_cfg.body_ids == slice(None):
#        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device=env.device)
#    else:
#        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device=env.device)
#
#    # get the current com of the bodies (num_assets, num_bodies)
#    coms = wp.to_torch(asset.root_newton_view.get_attribute("body_com", asset.root_newton_model)).clone()
#
#    # sample random CoM values
#    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
#    ranges = torch.tensor(range_list, device=env.device)
#    rand_samples = math_utils.sample_uniform(
#        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=env.device
#    ).unsqueeze(1)
#
#    # Randomize the com in range
#    coms[:, body_ids, :3] += rand_samples
#
#    # Set the new coms
#    mask = torch.zeros((env.scene.num_envs,), dtype=torch.bool, device=env.device)
#    mask[env_ids] = True
#    asset.root_newton_view.set_attribute(
#        "body_com", asset.root_newton_model, wp.from_torch(coms, dtype=wp.vec3), mask=mask
#    )
#    NewtonManager._solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)


#def randomize_rigid_body_collider_offsets(
#    env: ManagerBasedEnv,
#    env_ids: torch.Tensor | None,
#    asset_cfg: SceneEntityCfg,
#    rest_offset_distribution_params: tuple[float, float] | None = None,
#    contact_offset_distribution_params: tuple[float, float] | None = None,
#    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
#):
#    raise NotImplementedError("Not implemented")


#def randomize_physics_scene_gravity(
#    env: ManagerBasedEnv,
#    env_ids: torch.Tensor | None,
#    gravity_distribution_params: tuple[list[float], list[float]],
#    operation: Literal["add", "scale", "abs"],
#    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
#):
#    raise NotImplementedError("Not implemented")


#def randomize_actuator_gains(
#    env: ManagerBasedEnv,
#    env_ids: torch.Tensor | None,
#    asset_cfg: SceneEntityCfg,
#    stiffness_distribution_params: tuple[float, float] | None = None,
#    damping_distribution_params: tuple[float, float] | None = None,
#    operation: Literal["add", "scale", "abs"] = "abs",
#    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
#):
#    """Randomize the actuator gains in an articulation by adding, scaling, or setting random values.
#
#    This function allows randomizing the actuator stiffness and damping gains.
#
#    The function samples random values from the given distribution parameters and applies the operation to the joint properties.
#    It then sets the values into the actuator models. If the distribution parameters are not provided for a particular property,
#    the function does not modify the property.
#
#    .. tip::
#        For implicit actuators, this function uses CPU tensors to assign the actuator gains into the simulation.
#        In such cases, it is recommended to use this function only during the initialization of the environment.
#    """
#    # Extract the used quantities (to enable type-hinting)
#    asset: Articulation = env.scene[asset_cfg.name]
#
#    # Resolve environment ids
#    if env_ids is None:
#        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
#
#    def randomize(data: torch.Tensor, params: tuple[float, float]) -> torch.Tensor:
#        return _randomize_prop_by_op(
#            data, params, dim_0_ids=None, dim_1_ids=actuator_indices, operation=operation, distribution=distribution
#        )
#
#    # Loop through actuators and randomize gains
#    for actuator in asset.actuators.values():
#        if isinstance(asset_cfg.joint_ids, slice):
#            # we take all the joints of the actuator
#            actuator_indices = slice(None)
#            if isinstance(actuator.joint_indices, slice):
#                global_indices = slice(None)
#            else:
#                global_indices = torch.tensor(actuator.joint_indices, device=asset.device)
#        elif isinstance(actuator.joint_indices, slice):
#            # we take the joints defined in the asset config
#            global_indices = actuator_indices = torch.tensor(asset_cfg.joint_ids, device=asset.device)
#        else:
#            # we take the intersection of the actuator joints and the asset config joints
#            actuator_joint_indices = torch.tensor(actuator.joint_indices, device=asset.device)
#            asset_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)
#            # the indices of the joints in the actuator that have to be randomized
#            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
#            if len(actuator_indices) == 0:
#                continue
#            # maps actuator indices that have to be randomized to global joint indices
#            global_indices = actuator_joint_indices[actuator_indices]
#        # Randomize stiffness
#        if stiffness_distribution_params is not None:
#            stiffness = actuator.stiffness[env_ids].clone()
#            stiffness[:, actuator_indices] = asset.data.default_joint_stiffness[env_ids][:, global_indices].clone()
#            randomize(stiffness, stiffness_distribution_params)
#            actuator.stiffness[env_ids] = stiffness
#            if isinstance(actuator, ImplicitActuator):
#                asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)
#        # Randomize damping
#        if damping_distribution_params is not None:
#            damping = actuator.damping[env_ids].clone()
#            damping[:, actuator_indices] = asset.data.default_joint_damping[env_ids][:, global_indices].clone()
#            randomize(damping, damping_distribution_params)
#            actuator.damping[env_ids] = damping
#            if isinstance(actuator, ImplicitActuator):
#                asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


#def randomize_joint_parameters(
#    env: ManagerBasedEnv,
#    env_ids: torch.Tensor | None,
#    asset_cfg: SceneEntityCfg,
#    friction_distribution_params: tuple[float, float] | None = None,
#    armature_distribution_params: tuple[float, float] | None = None,
#    lower_limit_distribution_params: tuple[float, float] | None = None,
#    upper_limit_distribution_params: tuple[float, float] | None = None,
#    operation: Literal["add", "scale", "abs"] = "abs",
#    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
#):
#    """Randomize the simulated joint parameters of an articulation by adding, scaling, or setting random values.
#
#    This function allows randomizing the joint parameters of the asset. These correspond to the physics engine
#    joint properties that affect the joint behavior. The properties include the joint friction coefficient, armature,
#    and joint position limits.
#
#    The function samples random values from the given distribution parameters and applies the operation to the
#    joint properties. It then sets the values into the physics simulation. If the distribution parameters are
#    not provided for a particular property, the function does not modify the property.
#
#    .. tip::
#        This function uses CPU tensors to assign the joint properties. It is recommended to use this function
#        only during the initialization of the environment.
#    """
#    # extract the used quantities (to enable type-hinting)
#    asset: Articulation = env.scene[asset_cfg.name]
#
#    # resolve environment ids
#    if env_ids is None:
#        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
#
#    # resolve joint indices
#    if asset_cfg.joint_ids == slice(None):
#        joint_ids = slice(None)  # for optimization purposes
#    else:
#        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)
#
#    # sample joint properties from the given ranges and set into the physics simulation
#    # joint friction coefficient
#    if friction_distribution_params is not None:
#        friction_coeff = _randomize_prop_by_op(
#            asset.data.default_joint_friction_coeff.clone(),
#            friction_distribution_params,
#            env_ids,
#            joint_ids,
#            operation=operation,
#            distribution=distribution,
#        )
#        asset.write_joint_friction_coefficient_to_sim(
#            friction_coeff[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids
#        )
#
#    # joint armature
#    if armature_distribution_params is not None:
#        armature = _randomize_prop_by_op(
#            asset.data.default_joint_armature.clone(),
#            armature_distribution_params,
#            env_ids,
#            joint_ids,
#            operation=operation,
#            distribution=distribution,
#        )
#        asset.write_joint_armature_to_sim(armature[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids)
#
#    # joint position limits
#    if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
#        joint_pos_limits = asset.data.default_joint_pos_limits.clone()
#        # -- randomize the lower limits
#        if lower_limit_distribution_params is not None:
#            joint_pos_limits[..., 0] = _randomize_prop_by_op(
#                joint_pos_limits[..., 0],
#                lower_limit_distribution_params,
#                env_ids,
#                joint_ids,
#                operation=operation,
#                distribution=distribution,
#            )
#        # -- randomize the upper limits
#        if upper_limit_distribution_params is not None:
#            joint_pos_limits[..., 1] = _randomize_prop_by_op(
#                joint_pos_limits[..., 1],
#                upper_limit_distribution_params,
#                env_ids,
#                joint_ids,
#                operation=operation,
#                distribution=distribution,
#            )
#
#        # extract the position limits for the concerned joints
#        joint_pos_limits = joint_pos_limits[env_ids[:, None], joint_ids]
#        if (joint_pos_limits[..., 0] > joint_pos_limits[..., 1]).any():
#            raise ValueError(
#                "Randomization term 'randomize_joint_parameters' is setting lower joint limits that are greater than"
#                " upper joint limits. Please check the distribution parameters for the joint position limits."
#            )
#        # set the position limits into the physics simulation
#        asset.write_joint_position_limit_to_sim(
#            joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
#        )


#def randomize_fixed_tendon_parameters(
#    env: ManagerBasedEnv,
#    env_ids: torch.Tensor | None,
#    asset_cfg: SceneEntityCfg,
#    stiffness_distribution_params: tuple[float, float] | None = None,
#    damping_distribution_params: tuple[float, float] | None = None,
#    limit_stiffness_distribution_params: tuple[float, float] | None = None,
#    lower_limit_distribution_params: tuple[float, float] | None = None,
#    upper_limit_distribution_params: tuple[float, float] | None = None,
#    rest_length_distribution_params: tuple[float, float] | None = None,
#    offset_distribution_params: tuple[float, float] | None = None,
#    operation: Literal["add", "scale", "abs"] = "abs",
#    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
#):
#    raise NotImplementedError("Not implemented")


#def apply_external_force_torque(
#    env: ManagerBasedEnv,
#    env_ids: torch.Tensor,
#    force_range: tuple[float, float],
#    torque_range: tuple[float, float],
#    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#):
#    """Randomize the external forces and torques applied to the bodies.
#
#    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
#    and torques is equal to the number of bodies times the number of environments. The forces and torques are
#    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
#    applied when ``asset.write_data_to_sim()`` is called in the environment.
#    """
#    # extract the used quantities (to enable type-hinting)
#    asset: Articulation = env.scene[asset_cfg.name]
#    # resolve environment ids
#    if env_ids is None:
#        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
#    # resolve number of bodies
#    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
#
#    # sample random forces and torques
#    size = (len(env_ids), num_bodies, 3)
#    forces = math_utils.sample_uniform(*force_range, size, asset.device)
#    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
#    # set the forces and torques into the buffers
#    # note: these are only applied when you call: `asset.write_data_to_sim()`
#    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


#def push_by_setting_velocity(
#    env: ManagerBasedEnv,
#    env_ids: torch.Tensor,
#    velocity_range: dict[str, tuple[float, float]],
#    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#):
#    """Push the asset by setting the root velocity to a random value within the given ranges.
#
#    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
#    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.
#
#    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
#    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
#    If the dictionary does not contain a key, the velocity is set to zero for that axis.
#    """
#    # extract the used quantities (to enable type-hinting)
#    asset: Articulation = env.scene[asset_cfg.name]
#
#    # velocities
#    vel_w = asset.data.root_vel_w[env_ids]
#    # sample random velocities
#    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
#    ranges = torch.tensor(range_list, device=asset.device)
#    vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
#    # set the velocities into the physics simulation
#    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)

@wp.struct
class range_struct:
    low: float
    high: float

@wp.struct
class position_range_struct:
    x: range_struct
    y: range_struct
    z: range_struct

@wp.struct
class orientation_range_struct:
    roll: range_struct
    pitch: range_struct
    yaw: range_struct

@wp.struct
class linear_velocity_range_struct:
    vx: range_struct
    vy: range_struct
    vz: range_struct

@wp.struct
class angular_velocity_range_struct:
    wx: range_struct
    wy: range_struct
    wz: range_struct

@wp.struct
class pose_range_struct:
    position: position_range_struct
    orientation: orientation_range_struct

@wp.struct
class velocity_range_struct:
    linear: linear_velocity_range_struct
    angular: angular_velocity_range_struct

@wp.func
def sample_uniform_pose(state:wp.uint32, pose_range:pose_range_struct) -> wp.transformf:
    x = wp.randf(state, pose_range.position.x.low, pose_range.position.x.high)
    y = wp.randf(state + wp.uint32(1), pose_range.position.y.low, pose_range.position.y.high)
    z = wp.randf(state + wp.uint32(2), pose_range.position.z.low, pose_range.position.z.high)
    roll = wp.randf(state + wp.uint32(3), pose_range.orientation.roll.low, pose_range.orientation.roll.high)
    pitch = wp.randf(state + wp.uint32(4), pose_range.orientation.pitch.low, pose_range.orientation.pitch.high)
    yaw = wp.randf(state + wp.uint32(5), pose_range.orientation.yaw.low, pose_range.orientation.yaw.high)
    return wp.transform(wp.vec3f(x, y, z), wp.quat_rpy(roll, pitch, yaw))

@wp.func
def sample_uniform_velocity(state:wp.uint32, velocity_range:velocity_range_struct) -> wp.spatial_vectorf:
    vx = wp.randf(state, velocity_range.linear.vx.low, velocity_range.linear.vx.high)
    vy = wp.randf(state + wp.uint32(1), velocity_range.linear.vy.low, velocity_range.linear.vy.high)
    vz = wp.randf(state + wp.uint32(2), velocity_range.linear.vz.low, velocity_range.linear.vz.high)
    wx = wp.randf(state + wp.uint32(3), velocity_range.angular.wx.low, velocity_range.angular.wx.high)
    wy = wp.randf(state + wp.uint32(4), velocity_range.angular.wy.low, velocity_range.angular.wy.high)
    wz = wp.randf(state + wp.uint32(5), velocity_range.angular.wz.low, velocity_range.angular.wz.high)
    return wp.spatial_vectorf(vx, vy, vz, wx, wy, wz)

@wp.kernel
def reset_root_state_uniform_kernel(
    pose_range: pose_range_struct,
    velocity_range: velocity_range_struct,
    default_pose_buffer: wp.array(dtype=wp.transformf),
    default_velocity_buffer: wp.array(dtype=wp.spatial_vectorf),
    root_pose_buffer: wp.array(dtype=wp.transformf),
    root_vel_buffer: wp.array(dtype=wp.spatial_vectorf),
    mask: wp.array(dtype=wp.bool),
    state: wp.uint32,
):
    index = wp.tid()
    if mask[index]:
        root_pose_buffer[index] = sample_uniform_pose(state + wp.uint32(index*6), pose_range) + default_pose_buffer[index]
        root_vel_buffer[index] = sample_uniform_velocity(state + wp.uint32(mask.shape[0]*6 + index*6), velocity_range) + default_velocity_buffer[index]

class reset_root_state_uniform(ManagerTermBase):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # resolve the asset configuration
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        # create structs for position and velocity ranges
        self._pose_range = pose_range_struct()
        self._velocity_range = velocity_range_struct()
        # update the configuration
        self.update_config(
            cfg.params.get("pose_range", {}),
            cfg.params.get("velocity_range", {}),
        )
        # initialize the random state
        seed = np.random.randint(0, 1000000)
        self._state = wp.rand_init(seed)
        self._ALL_ENV_MASK = wp.ones((self.num_envs,), dtype=wp.bool, device=env.device)

    def update_config(
        self,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg | None = None,
    ) -> None:
        # update pose range
        self._pose_range.x.low = pose_range.get("x", (0.0, 0.0))[0]
        self._pose_range.x.high = pose_range.get("x", (0.0, 0.0))[1]
        self._pose_range.y.low = pose_range.get("y", (0.0, 0.0))[0]
        self._pose_range.y.high = pose_range.get("y", (0.0, 0.0))[1]
        self._pose_range.z.low = pose_range.get("z", (0.0, 0.0))[0]
        self._pose_range.z.high = pose_range.get("z", (0.0, 0.0))[1]
        self._pose_range.roll.low = pose_range.get("roll", (0.0, 0.0))[0]
        self._pose_range.roll.high = pose_range.get("roll", (0.0, 0.0))[1]
        self._pose_range.pitch.low = pose_range.get("pitch", (0.0, 0.0))[0]
        self._pose_range.pitch.high = pose_range.get("pitch", (0.0, 0.0))[1]
        self._pose_range.yaw.low = pose_range.get("yaw", (0.0, 0.0))[0]
        self._pose_range.yaw.high = pose_range.get("yaw", (0.0, 0.0))[1]
        # update velocity range
        self._velocity_range.vx.low = velocity_range.get("vx", (0.0, 0.0))[0]
        self._velocity_range.vx.high = velocity_range.get("vx", (0.0, 0.0))[1]
        self._velocity_range.vy.low = velocity_range.get("vy", (0.0, 0.0))[0]
        self._velocity_range.vy.high = velocity_range.get("vy", (0.0, 0.0))[1]
        self._velocity_range.vz.low = velocity_range.get("vz", (0.0, 0.0))[0]
        self._velocity_range.vz.high = velocity_range.get("vz", (0.0, 0.0))[1]
        self._velocity_range.wx.low = velocity_range.get("wx", (0.0, 0.0))[0]
        self._velocity_range.wx.high = velocity_range.get("wx", (0.0, 0.0))[1]
        self._velocity_range.wy.low = velocity_range.get("wy", (0.0, 0.0))[0]
        self._velocity_range.wy.high = velocity_range.get("wy", (0.0, 0.0))[1]
        self._velocity_range.wz.low = velocity_range.get("wz", (0.0, 0.0))[0]
        self._velocity_range.wz.high = velocity_range.get("wz", (0.0, 0.0))[1]
    
    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        mask: wp.array(dtype=wp.bool) | None = None,
        **kwargs,
    ) -> None:
        if mask is None:
            mask = self._ALL_ENV_MASK

        wp.launch(
            reset_root_state_uniform_kernel,
            dim=self.num_envs,
            inputs=[
                self._pose_range,
                self._velocity_range,
                self._asset.data.default_root_pose,
                self._asset.data.default_root_vel,
                self._asset.data.root_pose_w,
                self._asset.data.root_vel_w,
                self._state,
            ],
        )
        self._state += self.num_envs*12

@wp.func
def sample_uniform_pose_so3quat(state:wp.uint32, position_range:position_range_struct) -> wp.transformf:
    x = wp.randf(state, position_range.x.low, position_range.x.high)
    y = wp.randf(state + wp.uint32(1), position_range.y.low, position_range.y.high)
    z = wp.randf(state + wp.uint32(2), position_range.z.low, position_range.z.high)

    qx = wp.randn(state + wp.uint32(3))
    qy = wp.randn(state + wp.uint32(4))
    qz = wp.randn(state + wp.uint32(5))
    qw = wp.randn(state + wp.uint32(6))
    quat = wp.quatf(qx, qy, qz, qw)
    quat = wp.normalize(quat)

    return wp.transform(wp.vec3f(x, y, z), quat)

@wp.kernel
def reset_root_state_with_random_orientation_kernel(
    position_range: position_range_struct,
    velocity_range: velocity_range_struct,
    default_pose_buffer: wp.array(dtype=wp.transformf),
    default_velocity_buffer: wp.array(dtype=wp.spatial_vectorf),
    root_pose_buffer: wp.array(dtype=wp.transformf),
    root_vel_buffer: wp.array(dtype=wp.spatial_vectorf),
    mask: wp.array(dtype=wp.bool),
    state: wp.uint32,
):
    index = wp.tid()
    if mask[index]:
        root_pose_buffer[index] = sample_uniform_pose_so3quat(state + wp.uint32(index*7), position_range) + default_pose_buffer[index]
        root_vel_buffer[index] = sample_uniform_velocity(state + wp.uint32(mask.shape[0]*7 + index*6), velocity_range) + default_velocity_buffer[index]

class reset_root_state_with_random_orientation(ManagerTermBase):
    """Reset the asset root position and velocities sampled randomly within the given ranges
    and the asset root orientation sampled randomly from the SO(3).

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation uniformly from the SO(3) and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`position_range` - a dictionary of position ranges for each axis. The keys of the dictionary are ``x``,
      ``y``, and ``z``. The orientation is sampled uniformly from the SO(3).
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.
    """
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # resolve the asset configuration
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        # create structs for position and velocity ranges
        self._position_range = position_range_struct()
        self._velocity_range = velocity_range_struct()
        # update the configuration
        self.update_config(
            cfg.params.get("position_range", {}),
            cfg.params.get("velocity_range", {}),
        )
        # initialize the random state
        seed = np.random.randint(0, 1000000)
        self._state = wp.rand_init(seed)
        self._ALL_ENV_MASK = wp.ones((self.num_envs,), dtype=wp.bool, device=env.device)

    def update_config(self,
        position_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg | None = None,
    ) -> None:
        # update pose range
        self._position_range.x.low = position_range.get("x", (0.0, 0.0))[0]
        self._position_range.x.high = position_range.get("x", (0.0, 0.0))[1]
        self._position_range.y.low = position_range.get("y", (0.0, 0.0))[0]
        self._position_range.y.high = position_range.get("y", (0.0, 0.0))[1]
        self._position_range.z.low = position_range.get("z", (0.0, 0.0))[0]
        self._position_range.z.high = position_range.get("z", (0.0, 0.0))[1]
        # update velocity range
        self._velocity_range.vx.low = velocity_range.get("vx", (0.0, 0.0))[0]
        self._velocity_range.vx.high = velocity_range.get("vx", (0.0, 0.0))[1]
        self._velocity_range.vy.low = velocity_range.get("vy", (0.0, 0.0))[0]
        self._velocity_range.vy.high = velocity_range.get("vy", (0.0, 0.0))[1]
        self._velocity_range.vz.low = velocity_range.get("vz", (0.0, 0.0))[0]
        self._velocity_range.vz.high = velocity_range.get("vz", (0.0, 0.0))[1]
        self._velocity_range.wx.low = velocity_range.get("wx", (0.0, 0.0))[0]
        self._velocity_range.wx.high = velocity_range.get("wx", (0.0, 0.0))[1]
        self._velocity_range.wy.low = velocity_range.get("wy", (0.0, 0.0))[0]
        self._velocity_range.wy.high = velocity_range.get("wy", (0.0, 0.0))[1]
        self._velocity_range.wz.low = velocity_range.get("wz", (0.0, 0.0))[0]
        self._velocity_range.wz.high = velocity_range.get("wz", (0.0, 0.0))[1]
    
    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        mask: wp.array(dtype=wp.bool) | None = None,
        **kwargs,
    ) -> None:
        if mask is None:
            mask = self._ALL_ENV_MASK

        wp.launch(
            reset_root_state_with_random_orientation_kernel,
            dim=self.num_envs,
            inputs=[
                self._position_range,
                self._velocity_range,
                self._asset.data.default_root_pose,
                self._asset.data.default_root_vel,
                self._asset.data.root_pose_w,
                self._asset.data.root_vel_w,
                self._state,
            ],
        )
        self._state += self.num_envs*13


#def reset_root_state_from_terrain(
#    env: ManagerBasedEnv,
#    env_ids: torch.Tensor,
#    pose_range: dict[str, tuple[float, float]],
#    velocity_range: dict[str, tuple[float, float]],
#    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#):
#    """Reset the asset root state by sampling a random valid pose from the terrain.
#
#    This function samples a random valid pose(based on flat patches) from the terrain and sets the root state
#    of the asset to this position. The function also samples random velocities from the given ranges and sets them
#    into the physics simulation.
#
#    The function takes a dictionary of position and velocity ranges for each axis and rotation:
#
#    * :attr:`pose_range` - a dictionary of pose ranges for each axis. The keys of the dictionary are ``roll``,
#      ``pitch``, and ``yaw``. The position is sampled from the flat patches of the terrain.
#    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
#      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.
#
#    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
#    the position is set to zero for that axis.
#
#    Note:
#        The function expects the terrain to have valid flat patches under the key "init_pos". The flat patches
#        are used to sample the random pose for the robot.
#
#    Raises:
#        ValueError: If the terrain does not have valid flat patches under the key "init_pos".
#    """
#    # access the used quantities (to enable type-hinting)
#    asset: Articulation = env.scene[asset_cfg.name]
#    terrain: TerrainImporter = env.scene.terrain
#
#    # obtain all flat patches corresponding to the valid poses
#    valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
#    if valid_positions is None:
#        raise ValueError(
#            "The event term 'reset_root_state_from_terrain' requires valid flat patches under 'init_pos'."
#            f" Found: {list(terrain.flat_patches.keys())}"
#        )
#
#    # sample random valid poses
#    ids = torch.randint(0, valid_positions.shape[2], size=(len(env_ids),), device=env.device)
#    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
#    positions += asset.data.default_root_state[env_ids, :3]
#
#    # sample random orientations
#    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
#    ranges = torch.tensor(range_list, device=asset.device)
#    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)
#
#    # convert to quaternions
#    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])
#
#    # sample random velocities
#    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
#    ranges = torch.tensor(range_list, device=asset.device)
#    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
#
#    velocities = asset.data.default_root_state[env_ids, 7:13] + rand_samples
#
#    # set into the physics simulation
#    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
#    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

@wp.kernel
def reset_joints_by_scale_kernel(
    pos_range: wp.vec2f,
    vel_range: wp.vec2f,
    default_joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_vel: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    joint_vel_limits: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    mask: wp.array(dtype=wp.bool),
    joint_masks: wp.array(dtype=wp.bool),
    state: wp.uint32,
):
    i, j = wp.tid()
    if mask[i] and joint_masks[j]:
        joint_pos[i, j] = wp.clamp(
            default_joint_pos[i, j] * wp.randf(
                state + wp.uint32(i*mask.shape[0] + j),
                pos_range[0],
                pos_range[1],
            ),
            soft_joint_pos_limits[i, j][0],
            soft_joint_pos_limits[i, j][1]
        )
        joint_vel[i, j] = wp.clamp(
            default_joint_vel[i, j] * wp.randf(
                state + wp.uint32(mask.shape[0]*mask.shape[1] + i*mask.shape[0] + j),
                vel_range[0],
                vel_range[1],
            ),
            -joint_vel_limits[i, j],
            joint_vel_limits[i, j],
        )


class reset_joints_by_scale(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # resolve the asset configuration
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._launch_shape = (self.num_envs, len(asset_cfg.joint_ids))
        self._joint_masks = wp.array(asset_cfg.joint_masks, dtype=wp.bool, device=env.device)
        # create structs for position and velocity ranges
        self._pos_range = wp.vec2f(0.0, 0.0)
        self._vel_range = wp.vec2f(0.0, 0.0)
        # update the configuration
        self.update_config(
            cfg.params.get("position_range", (0.0, 0.0)),
            cfg.params.get("velocity_range", (0.0, 0.0)),
        )
        # initialize the random state
        seed = np.random.randint(0, 1000000)
        self._state = wp.rand_init(seed)
        self._ALL_ENV_MASK = wp.ones((self.num_envs,), dtype=wp.bool, device=env.device)

    def update_config(
        self,
        position_range: tuple[float, float],
        velocity_range: tuple[float, float],
        asset_cfg: SceneEntityCfg | None = None,
    ) -> None:
        # update position range
        self._pos_range = wp.vec2f(position_range[0], position_range[1])
        # update velocity range
        self._vel_range = wp.vec2f(velocity_range[0], velocity_range[1])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        if mask is None:
            mask = self._ALL_ENV_MASK
        wp.launch(
            reset_joints_by_scale_kernel,
            dim=self._launch_shape,
            inputs=[
                self._pos_range,
                self._vel_range,
                self._asset.data.default_joint_pos,
                self._asset.data.default_joint_vel,
                self._asset.data.soft_joint_pos_limits,
                self._asset.data.joint_vel_limits,
                self._asset.data.joint_pos,
                self._asset.data.joint_vel,
                mask,
                self._joint_masks,
                self._state,
            ],
        )
        self._state += self._launch_shape[0]*self._launch_shape[1]*2

@wp.kernel
def reset_joints_by_offset_kernel(
    pos_range: wp.vec2f,
    vel_range: wp.vec2f,
    default_joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_vel: wp.array2d(dtype=wp.float32),
    soft_joint_pos_limits: wp.array2d(dtype=wp.vec2f),
    joint_vel_limits: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    mask: wp.array(dtype=wp.bool),
    joint_masks: wp.array(dtype=wp.bool),
    state: wp.uint32,
):
    i, j = wp.tid()
    if mask[i] and joint_masks[j]:
        joint_pos[i, j] = wp.clamp(
            default_joint_pos[i, j] + wp.randf(
                state + wp.uint32(i*mask.shape[0] + j),
                pos_range[0],
                pos_range[1]
            ),
            soft_joint_pos_limits[i, j][0],
            soft_joint_pos_limits[i, j][1]
        )
        joint_vel[i, j] = wp.clamp(
            default_joint_vel[i, j] + wp.randf(
                state + wp.uint32(mask.shape[0]*mask.shape[1] + i*mask.shape[0] + j),
                vel_range[0],
                vel_range[1]
            ),
            -joint_vel_limits[i, j],
            joint_vel_limits[i, j],
        )

class reset_joints_by_offset(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # resolve the asset configuration
        asset_cfg: SceneEntityCfg = resolve_asset_cfg(cfg.params, env)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._launch_shape = (self.num_envs, len(asset_cfg.joint_ids))
        self._joint_masks = wp.array(asset_cfg.joint_masks, dtype=wp.bool, device=env.device)
        # create structs for position and velocity ranges
        self._pos_range = wp.vec2f(0.0, 0.0)
        self._vel_range = wp.vec2f(0.0, 0.0)
        # update the configuration
        self.update_config(
            cfg.params.get("position_range", (0.0, 0.0)),
            cfg.params.get("velocity_range", (0.0, 0.0)),
        )
        # initialize the random state
        seed = np.random.randint(0, 1000000)
        self._state = wp.rand_init(seed)
        self._ALL_ENV_MASK = wp.ones((self.num_envs,), dtype=wp.bool, device=env.device)

    def update_config(
        self,
        position_range: tuple[float, float],
        velocity_range: tuple[float, float],
        asset_cfg: SceneEntityCfg | None = None,
    ) -> None:
        # update position range
        self._pos_range = wp.vec2f(position_range[0], position_range[1])
        # update velocity range
        self._vel_range = wp.vec2f(velocity_range[0], velocity_range[1])
    
    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        mask: wp.array(dtype=wp.bool) | None = None,
        **kwargs,
    ) -> None:
        if mask is None:
            mask = self._ALL_ENV_MASK
        wp.launch(
            reset_joints_by_offset_kernel,
            dim=self._launch_shape,
            inputs=[
                self._pos_range,
                self._vel_range,
                self._asset.data.default_joint_pos,
                self._asset.data.default_joint_vel,
                self._asset.data.soft_joint_pos_limits,
                self._asset.data.joint_vel_limits,
                self._asset.data.joint_pos,
                self._asset.data.joint_vel,
                mask,
                self._joint_masks,
                self._state,
            ],
        )
        self._state += self._launch_shape[0]*self._launch_shape[1]*2


@wp.kernel
def reset_rigif_object_to_default_kernel(
    default_root_pose: wp.array(dtype=wp.transformf),
    default_root_vel: wp.array(dtype=wp.spatial_vectorf),
    env_origins: wp.array(dtype=wp.vec3f),
    root_pose: wp.array(dtype=wp.transformf),
    root_vel: wp.array(dtype=wp.spatial_vectorf),
    mask: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    if mask[i]:
        root_pose[i] = default_root_pose[i]
        wp.transform_set_translation(root_pose[i], wp.transform_get_translation(root_pose[i]) + env_origins[i])
        root_vel[i] = default_root_vel[i]

# FIXME: Add targets to match default state
@wp.kernel
def reset_articulation_root_to_default_kernel(
    default_root_pose: wp.array(dtype=wp.transformf),
    default_root_vel: wp.array(dtype=wp.spatial_vectorf),
    env_origins: wp.array(dtype=wp.vec3f),
    root_pose: wp.array(dtype=wp.transformf),
    root_vel: wp.array(dtype=wp.spatial_vectorf),
    mask: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    if mask[i]:
        root_pose[i] = default_root_pose[i]
        wp.transform_set_translation(root_pose[i], wp.transform_get_translation(root_pose[i]) + env_origins[i])
        root_vel[i] = default_root_vel[i]

@wp.kernel
def reset_articulation_joints_to_default_kernel(
    default_joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_vel: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    mask: wp.array(dtype=wp.bool),
):
    i, j = wp.tid()
    if mask[i]:
        joint_pos[i, j] = default_joint_pos[i, j]
        joint_vel[i, j] = default_joint_vel[i, j]

# FIXME: Add deformable support
def reset_scene_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mask: wp.array(dtype=wp.bool) | None = None,
    **kwargs,
) -> None:
    """Reset the scene to the default state specified in the scene configuration."""
    if mask is None:
        mask = wp.ones((env.num_envs,), dtype=wp.bool, device=env.device)
    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        wp.launch(
            reset_rigif_object_to_default_kernel,
            dim=env.num_envs,
            inputs=[
                rigid_object.data.default_root_pose,
                rigid_object.data.default_root_vel,
                env.scene.env_origins,
                rigid_object.data.root_pose_w,
                rigid_object.data.root_vel_w,
                mask,
            ],
        )
    # articulations
    for articulation_asset in env.scene.articulations.values():
        wp.launch(
            reset_articulation_root_to_default_kernel,
            dim=env.num_envs,
            inputs=[
                articulation_asset.data.default_root_pose,
                articulation_asset.data.default_root_vel,
                env.scene.env_origins,
                articulation_asset.data.root_pose_w,
                articulation_asset.data.root_vel_w,
                mask,
            ],
        )
        wp.launch(
            reset_articulation_joints_to_default_kernel,
            dim=(env.num_envs, articulation_asset.num_joints),
            inputs=[
                articulation_asset.data.default_joint_pos,
                articulation_asset.data.default_joint_vel,
                articulation_asset.data.joint_pos,
                articulation_asset.data.joint_vel,
                mask,
            ],
        )
    # deformable objects
    #for deformable_object in env.scene.deformable_objects.values():
    #    # obtain default and set into the physics simulation
    #    nodal_state = deformable_object.data.default_nodal_state_w[env_ids].clone()
    #    deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


#def reset_nodal_state_uniform(
#    env: ManagerBasedEnv,
#    env_ids: torch.Tensor,
#    position_range: dict[str, tuple[float, float]],
#    velocity_range: dict[str, tuple[float, float]],
#    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#):
#    raise NotImplementedError("Not implemented")


#class randomize_visual_texture_material(ManagerTermBase):
#    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
#        raise NotImplementedError("Not implemented")
#
#    def __call__(
#        self,
#        env: ManagerBasedEnv,
#        env_ids: torch.Tensor,
#        event_name: str,
#        asset_cfg: SceneEntityCfg,
#        texture_paths: list[str],
#        texture_rotation: tuple[float, float] = (0.0, 0.0),
#    ):
#        raise NotImplementedError("Not implemented")


#class randomize_visual_color(ManagerTermBase):
#    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
#        raise NotImplementedError("Not implemented")
#
#    def __call__(
#        self,
#        env: ManagerBasedEnv,
#        env_ids: torch.Tensor,
#        event_name: str,
#        asset_cfg: SceneEntityCfg,
#        colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
#        mesh_name: str = "",
#    ):
#        raise NotImplementedError("Not implemented")


"""
Internal helper functions.
"""


#def _randomize_prop_by_op(
#    data: torch.Tensor,
#    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
#    dim_0_ids: torch.Tensor | None,
#    dim_1_ids: torch.Tensor | slice,
#    operation: Literal["add", "scale", "abs"],
#    distribution: Literal["uniform", "log_uniform", "gaussian"],
#) -> torch.Tensor:
#    """Perform data randomization based on the given operation and distribution.
#
#    Args:
#        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
#        distribution_parameters: The parameters for the distribution to sample values from.
#        dim_0_ids: The indices of the first dimension to randomize.
#        dim_1_ids: The indices of the second dimension to randomize.
#        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
#        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.
#
#    Returns:
#        The data tensor after randomization. Shape is (dim_0, dim_1).
#
#    Raises:
#        NotImplementedError: If the operation or distribution is not supported.
#    """
#    # resolve shape
#    # -- dim 0
#    if dim_0_ids is None:
#        n_dim_0 = data.shape[0]
#        dim_0_ids = slice(None)
#    else:
#        n_dim_0 = len(dim_0_ids)
#        if not isinstance(dim_1_ids, slice):
#            dim_0_ids = dim_0_ids[:, None]
#    # -- dim 1
#    if isinstance(dim_1_ids, slice):
#        n_dim_1 = data.shape[1]
#    else:
#        n_dim_1 = len(dim_1_ids)
#
#    # resolve the distribution
#    if distribution == "uniform":
#        dist_fn = math_utils.sample_uniform
#    elif distribution == "log_uniform":
#        dist_fn = math_utils.sample_log_uniform
#    elif distribution == "gaussian":
#        dist_fn = math_utils.sample_gaussian
#    else:
#        raise NotImplementedError(
#            f"Unknown distribution: '{distribution}' for joint properties randomization."
#            " Please use 'uniform', 'log_uniform', 'gaussian'."
#        )
#    # perform the operation
#    if operation == "add":
#        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
#    elif operation == "scale":
#        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
#    elif operation == "abs":
#        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
#    else:
#        raise NotImplementedError(
#            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
#        )
#    return data
