# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX backend implementations for MDP event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

import carb
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import BaseArticulation
from isaaclab.envs.mdp.physics_events import randomize_prop_by_op, validate_scale_range

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg, SceneEntityCfg


class RandomizeRigidBodyMaterial:
    """PhysX backend implementation for material randomization.

    Uses the bucket-based approach required by PhysX's 64000 unique material limit.
    Materials are pre-sampled into buckets and randomly assigned to shapes.
    """

    def __init__(
        self, cfg: EventTermCfg, env: ManagerBasedEnv, asset: RigidObject | Articulation, asset_cfg: SceneEntityCfg
    ):
        # obtain parameters for sampling friction and restitution values
        static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(cfg.params.get("num_buckets", 1))

        # sample material properties from the given ranges
        # note: we only sample the materials once during initialization
        #   afterwards these are randomly assigned to the geometries of the asset
        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")

        # ensure dynamic friction is always less than static friction
        make_consistent = cfg.params.get("make_consistent", False)
        if make_consistent:
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

        self.asset = asset
        self.asset_cfg = asset_cfg

        # obtain number of shapes per body (needed for indexing the material properties correctly)
        # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
        #  per body. We use the physics simulation view to obtain the number of shapes per body.
        if isinstance(asset, BaseArticulation) and asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in asset.root_view.link_paths[0]:
                link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            # ensure the parsing is correct
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = asset.root_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            # in this case, we don't need to do special indexing
            self.num_shapes_per_body = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu", dtype=torch.int32)
        else:
            env_ids = env_ids.cpu()

        # randomly assign material IDs to the geometries
        total_num_shapes = self.asset.root_view.max_shapes
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        # retrieve material buffer from the physics simulation
        materials = wp.to_torch(self.asset.root_view.get_material_properties())

        # update material buffer with new samples
        if self.num_shapes_per_body is not None:
            # sample material properties from the given ranges
            for body_id in self.asset_cfg.body_ids:
                # obtain indices of shapes for the body
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                # assign the new materials
                # material samples are of shape: num_env_ids x total_num_shapes x 3
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            # assign all the materials
            materials[env_ids] = material_samples[:]

        # apply to simulation
        self.asset.root_view.set_material_properties(
            wp.from_torch(materials, dtype=wp.float32), wp.from_torch(env_ids, dtype=wp.int32)
        )


class RandomizeRigidBodyColliderOffsets:
    """PhysX backend implementation for collider offset randomization.

    Uses rest offset and contact offset directly via the PhysX tensor API.
    """

    def __init__(self, asset: RigidObject | Articulation):
        self.asset = asset
        self.default_rest_offsets = wp.to_torch(asset.root_view.get_rest_offsets()).clone()
        self.default_contact_offsets = wp.to_torch(asset.root_view.get_contact_offsets()).clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        rest_offset_distribution_params: tuple[float, float] | None = None,
        contact_offset_distribution_params: tuple[float, float] | None = None,
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu", dtype=torch.int32)
        else:
            env_ids = env_ids.to(device="cpu", dtype=torch.int32)
        wp_env_ids = wp.from_torch(env_ids, dtype=wp.int32)

        if rest_offset_distribution_params is not None:
            rest_offset = self.default_rest_offsets.clone()
            rest_offset = randomize_prop_by_op(
                rest_offset,
                rest_offset_distribution_params,
                None,
                slice(None),
                operation="abs",
                distribution=distribution,
            )
            self.asset.root_view.set_rest_offsets(wp.from_torch(rest_offset), wp_env_ids)

        if contact_offset_distribution_params is not None:
            contact_offset = self.default_contact_offsets.clone()
            contact_offset = randomize_prop_by_op(
                contact_offset,
                contact_offset_distribution_params,
                None,
                slice(None),
                operation="abs",
                distribution=distribution,
            )
            self.asset.root_view.set_contact_offsets(wp.from_torch(contact_offset), wp_env_ids)


class RandomizeRigidBodyCom:
    """PhysX backend implementation for center-of-mass randomization.

    Passes the full CoM pose (position + quaternion) to ``set_coms_index``.
    """

    def __init__(self, asset: RigidObject | Articulation, asset_cfg: SceneEntityCfg):
        self.asset = asset
        self.asset_cfg = asset_cfg
        self.default_com = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        com_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg,
    ):
        # store default CoM on first call for repeatable randomization
        if self.default_com is None:
            self.default_com = self.asset.data.body_com_pose_b.torch.clone()

        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        else:
            env_ids = env_ids.to(self.asset.device)

        # resolve body indices
        if self.asset_cfg.body_ids == slice(None):
            body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device=self.asset.device)
        else:
            body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device=self.asset.device)

        # sample random CoM values
        range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device=self.asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=self.asset.device
        ).unsqueeze(1)

        # start from defaults and add random offsets
        coms = self.default_com.clone()
        coms[env_ids[:, None], body_ids, :3] += rand_samples

        # PhysX expects the full CoM pose (position + quaternion)
        # note: pass partial data of shape (len(env_ids), len(body_ids), ...) to match the API
        self.asset.set_coms_index(coms=coms[env_ids[:, None], body_ids], body_ids=body_ids, env_ids=env_ids)


class RandomizePhysicsSceneGravity:
    """PhysX backend implementation for scene gravity randomization.

    Samples a single gravity vector and sets it scene-wide via the PhysX simulation view;
    all environments share the same gravity.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        self._carb = carb
        self._physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view

        distribution = cfg.params.get("distribution", "uniform")
        if distribution not in ("uniform", "log_uniform", "gaussian"):
            raise NotImplementedError(
                f"Unknown distribution: '{distribution}' for gravity randomization."
                " Please use 'uniform', 'log_uniform', or 'gaussian'."
            )

        operation = cfg.params["operation"]
        if operation not in ("add", "scale", "abs"):
            raise NotImplementedError(
                f"Unknown operation: '{operation}' for gravity randomization. Please use 'add', 'scale', or 'abs'."
            )

        gravity_distribution_params = cfg.params["gravity_distribution_params"]
        self._dist_param_0 = torch.tensor(gravity_distribution_params[0], device=env.device, dtype=torch.float32)
        self._dist_param_1 = torch.tensor(gravity_distribution_params[1], device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        gravity_distribution_params: tuple[list[float], list[float]],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        self._dist_param_0[0] = gravity_distribution_params[0][0]
        self._dist_param_1[0] = gravity_distribution_params[1][0]
        self._dist_param_0[1] = gravity_distribution_params[0][1]
        self._dist_param_1[1] = gravity_distribution_params[1][1]
        self._dist_param_0[2] = gravity_distribution_params[0][2]
        self._dist_param_1[2] = gravity_distribution_params[1][2]

        # PhysX applies a single gravity vector scene-wide via the simulation view
        gravity = torch.tensor(env.sim.cfg.gravity, device="cpu").unsqueeze(0)
        gravity = randomize_prop_by_op(
            gravity,
            (self._dist_param_0.cpu(), self._dist_param_1.cpu()),
            None,
            slice(None),
            operation=operation,
            # distribution is cached/validated at init; PhysX applies a single scene-wide vector
            # using uniform sampling (matches the original unified implementation).
            distribution="uniform",
        )
        gravity = gravity[0].tolist()
        self._physics_sim_view.set_gravity(self._carb.Float3(*gravity))


class RandomizeJointParameters:
    """PhysX backend implementation for joint-parameter randomization.

    Randomizes static, dynamic, and viscous joint friction coefficients, joint armature, and
    joint position limits.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv, asset: Articulation, asset_cfg: SceneEntityCfg):
        self.asset = asset
        self.asset_cfg = asset_cfg

        # cache default values
        self.default_joint_friction_coeff = self.asset.data.joint_friction_coeff.torch.clone()
        self.default_joint_armature = self.asset.data.joint_armature.torch.clone()
        self.default_joint_pos_limits = self.asset.data.joint_pos_limits.torch.clone()
        self.default_dynamic_joint_friction_coeff = self.asset.data.joint_dynamic_friction_coeff.torch.clone()
        self.default_viscous_joint_friction_coeff = self.asset.data.joint_viscous_friction_coeff.torch.clone()

        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "friction_distribution_params" in cfg.params:
                validate_scale_range(cfg.params["friction_distribution_params"], "friction_distribution_params")
            if "armature_distribution_params" in cfg.params:
                validate_scale_range(cfg.params["armature_distribution_params"], "armature_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_joint_parameters' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        friction_distribution_params: tuple[float, float] | None = None,
        armature_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        # resolve joint indices
        if self.asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)  # for optimization purposes
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids_for_slice = env_ids[:, None]
        else:
            env_ids_for_slice = env_ids

        # joint friction coefficient (static + dynamic + viscous)
        if friction_distribution_params is not None:
            friction_coeff = randomize_prop_by_op(
                self.default_joint_friction_coeff.clone(),
                friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            # ensure the friction coefficient is non-negative
            friction_coeff = torch.clamp(friction_coeff, min=0.0)
            static_friction_coeff = friction_coeff[env_ids_for_slice, joint_ids]

            dynamic_friction_coeff = randomize_prop_by_op(
                self.default_dynamic_joint_friction_coeff.clone(),
                friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            viscous_friction_coeff = randomize_prop_by_op(
                self.default_viscous_joint_friction_coeff.clone(),
                friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )

            # clamp to non-negative
            dynamic_friction_coeff = torch.clamp(dynamic_friction_coeff, min=0.0)
            viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)

            # ensure dynamic <= static (same shape before indexing)
            dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, friction_coeff)

            # index once at the end
            dynamic_friction_coeff = dynamic_friction_coeff[env_ids_for_slice, joint_ids]
            viscous_friction_coeff = viscous_friction_coeff[env_ids_for_slice, joint_ids]

            self.asset.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=static_friction_coeff,
                joint_dynamic_friction_coeff=dynamic_friction_coeff,
                joint_viscous_friction_coeff=viscous_friction_coeff,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )

        # joint armature
        if armature_distribution_params is not None:
            armature = randomize_prop_by_op(
                self.asset.data.default_joint_armature.torch.clone(),
                armature_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_joint_armature_to_sim(
                armature[env_ids_for_slice, joint_ids], joint_ids=joint_ids, env_ids=env_ids
            )

        # joint position limits
        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            joint_pos_limits = self.default_joint_pos_limits.clone()
            # -- randomize the lower limits
            if lower_limit_distribution_params is not None:
                joint_pos_limits[..., 0] = randomize_prop_by_op(
                    joint_pos_limits[..., 0],
                    lower_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
            # -- randomize the upper limits
            if upper_limit_distribution_params is not None:
                joint_pos_limits[..., 1] = randomize_prop_by_op(
                    joint_pos_limits[..., 1],
                    upper_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )

            # extract the position limits for the concerned joints
            joint_pos_limits = joint_pos_limits[env_ids_for_slice, joint_ids]
            if (joint_pos_limits[..., 0] > joint_pos_limits[..., 1]).any():
                raise ValueError(
                    "Randomization term 'randomize_joint_parameters' is setting lower joint limits that are greater"
                    " than upper joint limits. Please check the distribution parameters for the joint position limits."
                )
            # set the position limits into the physics simulation
            self.asset.write_joint_position_limit_to_sim_index(
                limits=joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
            )


class RandomizeFixedTendonParameters:
    """PhysX backend implementation for fixed-tendon parameter randomization.

    Randomizes tendon stiffness, damping, limit stiffness, position limits, rest length, and
    offset.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv, asset: Articulation, asset_cfg: SceneEntityCfg):
        self.asset = asset
        self.asset_cfg = asset_cfg

        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "stiffness_distribution_params" in cfg.params:
                validate_scale_range(
                    cfg.params["stiffness_distribution_params"], "stiffness_distribution_params", allow_zero=False
                )
            if "damping_distribution_params" in cfg.params:
                validate_scale_range(cfg.params["damping_distribution_params"], "damping_distribution_params")
            if "limit_stiffness_distribution_params" in cfg.params:
                validate_scale_range(
                    cfg.params["limit_stiffness_distribution_params"], "limit_stiffness_distribution_params"
                )
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_fixed_tendon_parameters' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        limit_stiffness_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        rest_length_distribution_params: tuple[float, float] | None = None,
        offset_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        # resolve tendon indices
        if self.asset_cfg.fixed_tendon_ids == slice(None):
            tendon_ids = slice(None)  # for optimization purposes
        else:
            tendon_ids = torch.tensor(self.asset_cfg.fixed_tendon_ids, dtype=torch.int, device=self.asset.device)

        # stiffness
        if stiffness_distribution_params is not None:
            stiffness = randomize_prop_by_op(
                self.asset.data.fixed_tendon_stiffness.torch.clone(),
                stiffness_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_stiffness_index(
                stiffness=stiffness[env_ids[:, None], tendon_ids], fixed_tendon_ids=tendon_ids, env_ids=env_ids
            )

        # damping
        if damping_distribution_params is not None:
            damping = randomize_prop_by_op(
                self.asset.data.fixed_tendon_damping.torch.clone(),
                damping_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_damping_index(
                damping=damping[env_ids[:, None], tendon_ids], fixed_tendon_ids=tendon_ids, env_ids=env_ids
            )

        # limit stiffness
        if limit_stiffness_distribution_params is not None:
            limit_stiffness = randomize_prop_by_op(
                self.asset.data.fixed_tendon_limit_stiffness.torch.clone(),
                limit_stiffness_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_limit_stiffness(
                limit_stiffness[env_ids[:, None], tendon_ids], tendon_ids, env_ids
            )

        # position limits
        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            limit = self.asset.data.fixed_tendon_pos_limits.torch.clone()
            # -- lower limit
            if lower_limit_distribution_params is not None:
                limit[..., 0] = randomize_prop_by_op(
                    limit[..., 0],
                    lower_limit_distribution_params,
                    env_ids,
                    tendon_ids,
                    operation=operation,
                    distribution=distribution,
                )
            # -- upper limit
            if upper_limit_distribution_params is not None:
                limit[..., 1] = randomize_prop_by_op(
                    limit[..., 1],
                    upper_limit_distribution_params,
                    env_ids,
                    tendon_ids,
                    operation=operation,
                    distribution=distribution,
                )

            # check if the limits are valid
            tendon_limits = limit[env_ids[:, None], tendon_ids]
            if (tendon_limits[..., 0] > tendon_limits[..., 1]).any():
                raise ValueError(
                    "Randomization term 'randomize_fixed_tendon_parameters' is setting lower tendon limits that are"
                    " greater than upper tendon limits."
                )
            self.asset.set_fixed_tendon_position_limit_index(
                limit=tendon_limits, fixed_tendon_ids=tendon_ids, env_ids=env_ids
            )

        # rest length
        if rest_length_distribution_params is not None:
            rest_length = randomize_prop_by_op(
                self.asset.data.fixed_tendon_rest_length.torch.clone(),
                rest_length_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_rest_length_index(
                rest_length=rest_length[env_ids[:, None], tendon_ids], fixed_tendon_ids=tendon_ids, env_ids=env_ids
            )

        # offset
        if offset_distribution_params is not None:
            offset = randomize_prop_by_op(
                self.asset.data.fixed_tendon_offset.torch.clone(),
                offset_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.set_fixed_tendon_offset_index(
                offset=offset[env_ids[:, None], tendon_ids], fixed_tendon_ids=tendon_ids, env_ids=env_ids
            )

        # write the fixed tendon properties into the simulation
        self.asset.write_fixed_tendon_properties_to_sim_index(env_ids=env_ids)


class RandomizeActuatorGains:
    """PhysX backend implementation for actuator-gain randomization."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv, asset: Articulation, asset_cfg: SceneEntityCfg):
        self.asset = asset
        self.asset_cfg = asset_cfg

        self.default_joint_stiffness = self.asset.data.joint_stiffness.torch.clone()
        self.default_joint_damping = self.asset.data.joint_damping.torch.clone()

        # For explicit Lab actuators the sim-level stiffness/damping is zeroed out,
        # so patch the defaults with the actual actuator PD gains.
        for actuator in self.asset.actuators.values():
            if not isinstance(actuator, ImplicitActuator):
                joint_ids = actuator.joint_indices
                self.default_joint_stiffness[:, joint_ids] = actuator.stiffness
                self.default_joint_damping[:, joint_ids] = actuator.damping

        # Explicit Newton actuators keep their kp/kd on the per-actuator controller arrays
        # (not on a Lab Actuator object), so the asset exposes a per-articulation snapshot
        # taken at articulation init time. This applies on either backend.
        newton_default_stiffness = getattr(self.asset, "newton_default_stiffness", None)
        if newton_default_stiffness is not None:
            joint_ids = self.asset.newton_managed_local_joints
            self.default_joint_stiffness[:, joint_ids] = newton_default_stiffness[:, joint_ids]
            self.default_joint_damping[:, joint_ids] = self.asset.newton_default_damping[:, joint_ids]

        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "stiffness_distribution_params" in cfg.params:
                validate_scale_range(
                    cfg.params["stiffness_distribution_params"], "stiffness_distribution_params", allow_zero=False
                )
            if "damping_distribution_params" in cfg.params:
                validate_scale_range(cfg.params["damping_distribution_params"], "damping_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_actuator_gains' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # Resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        def randomize(data: torch.Tensor, params: tuple[float, float]) -> torch.Tensor:
            return randomize_prop_by_op(
                data, params, dim_0_ids=None, dim_1_ids=actuator_indices, operation=operation, distribution=distribution
            )

        # Loop through actuators and randomize gains
        for actuator in self.asset.actuators.values():
            if isinstance(self.asset_cfg.joint_ids, slice):
                # we take all the joints of the actuator
                actuator_indices = slice(None)
                if isinstance(actuator.joint_indices, slice):
                    global_indices = slice(None)
                elif isinstance(actuator.joint_indices, torch.Tensor):
                    global_indices = actuator.joint_indices.to(self.asset.device)
                else:
                    raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")
            elif isinstance(actuator.joint_indices, slice):
                # we take the joints defined in the asset config
                global_indices = actuator_indices = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
            else:
                # we take the intersection of the actuator joints and the asset config joints
                actuator_joint_indices = actuator.joint_indices
                asset_joint_ids = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
                # the indices of the joints in the actuator that have to be randomized
                actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
                if len(actuator_indices) == 0:
                    continue
                # maps actuator indices that have to be randomized to global joint indices
                global_indices = actuator_joint_indices[actuator_indices]
            # Randomize stiffness
            if stiffness_distribution_params is not None:
                stiffness = actuator.stiffness[env_ids].clone()
                stiffness[:, actuator_indices] = self.default_joint_stiffness[env_ids][:, global_indices].clone()
                randomize(stiffness, stiffness_distribution_params)
                actuator.stiffness[env_ids] = stiffness
                if isinstance(actuator, ImplicitActuator):
                    self.asset.write_joint_stiffness_to_sim_index(
                        stiffness=stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids
                    )
            # Randomize damping
            if damping_distribution_params is not None:
                damping = actuator.damping[env_ids].clone()
                damping[:, actuator_indices] = self.default_joint_damping[env_ids][:, global_indices].clone()
                randomize(damping, damping_distribution_params)
                actuator.damping[env_ids] = damping
                if isinstance(actuator, ImplicitActuator):
                    self.asset.write_joint_damping_to_sim_index(
                        damping=damping, joint_ids=actuator.joint_indices, env_ids=env_ids
                    )

        # Push DR updates to explicit actuator controllers via the asset's own write methods.
        if not hasattr(self.asset, "write_actuator_stiffness_to_sim"):
            return

        if isinstance(self.asset_cfg.joint_ids, slice):
            joint_ids = torch.arange(self.asset.num_joints, device=self.asset.device, dtype=torch.long)
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device, dtype=torch.long)

        if stiffness_distribution_params is not None:
            new_stiffness = self.default_joint_stiffness[env_ids][:, joint_ids].clone()
            randomize_prop_by_op(
                new_stiffness,
                stiffness_distribution_params,
                dim_0_ids=None,
                dim_1_ids=slice(None),
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_actuator_stiffness_to_sim(
                stiffness=new_stiffness,
                env_ids=env_ids,
                joint_ids=joint_ids,
            )
        if damping_distribution_params is not None:
            new_damping = self.default_joint_damping[env_ids][:, joint_ids].clone()
            randomize_prop_by_op(
                new_damping,
                damping_distribution_params,
                dim_0_ids=None,
                dim_1_ids=slice(None),
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_actuator_damping_to_sim(
                damping=new_damping,
                env_ids=env_ids,
                joint_ids=joint_ids,
            )


class RandomizeRigidBodyScale:
    """PhysX backend implementation for rigid-body scale randomization via USD authoring.

    Modifies the ``xformOp:scale`` property of all prims corresponding to the asset. This must
    run before the simulation starts playing (event mode ``"usd"``).
    """

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        scale_range: tuple[float, float] | dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg,
        relative_child_path: str | None = None,
    ):
        # check if sim is running
        if env.sim.is_playing():
            raise RuntimeError(
                "Randomizing scale while simulation is running leads to unpredictable behaviors."
                " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
            )

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        if any(cls.__name__ == "Articulation" for cls in type(asset).__mro__):
            raise ValueError(
                "Scaling an articulation randomly is not supported, as it affects joint attributes and can cause"
                " unexpected behavior. To achieve different scales, we recommend generating separate USD files for"
                " each version of the articulation and using multi-asset spawning. For more details, refer to:"
                " https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html"
            )

        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # acquire stage
        stage = env.sim.stage
        # resolve prim paths for spawning and cloning
        prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

        # sample scale values
        if isinstance(scale_range, dict):
            range_list = [scale_range.get(key, (1.0, 1.0)) for key in ["x", "y", "z"]]
            ranges = torch.tensor(range_list, device="cpu")
            rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")
        else:
            rand_samples = math_utils.sample_uniform(*scale_range, (len(env_ids), 1), device="cpu")
            rand_samples = rand_samples.repeat(1, 3)
        # convert to list for the for loop
        rand_samples = rand_samples.tolist()

        # apply the randomization to the parent if no relative child path is provided
        # this might be useful if user wants to randomize a particular mesh in the prim hierarchy
        if relative_child_path is None:
            relative_child_path = ""
        elif not relative_child_path.startswith("/"):
            relative_child_path = "/" + relative_child_path

        # use sdf changeblock for faster processing of USD properties
        with Sdf.ChangeBlock():
            for i, env_id in enumerate(env_ids):
                # path to prim to randomize
                prim_path = prim_paths[env_id] + relative_child_path
                # spawn single instance
                prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

                # get the attribute to randomize
                scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
                # if the scale attribute does not exist, create it
                has_scale_attr = scale_spec is not None
                if not has_scale_attr:
                    scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

                # set the new scale
                scale_spec.default = Gf.Vec3f(*rand_samples[i])

                # ensure the operation is done in the right ordering if we created the scale attribute.
                # otherwise, we assume the scale attribute is already in the right order.
                # note: by default isaac sim follows this ordering for the transform stack so any asset
                #   created through it will have the correct ordering
                if not has_scale_attr:
                    op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                    if op_order_spec is None:
                        op_order_spec = Sdf.AttributeSpec(
                            prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                        )
                    op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
