# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton backend implementations for MDP event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import warp as wp
from newton.solvers import SolverNotifyFlags

import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.envs.mdp.physics_events import randomize_prop_by_op, validate_scale_range

from isaaclab_newton.assets import Articulation as NewtonArticulation
from isaaclab_newton.physics.newton_manager import NewtonManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    from isaaclab_newton.assets import Articulation, RigidObject


class RandomizeRigidBodyMaterial:
    """Newton backend implementation for material randomization.

    Newton can assign arbitrary friction/restitution per shape (no bucket limitation).
    Samples friction (mu) and restitution continuously from the given ranges.
    Newton uses a single friction coefficient (mu), so ``dynamic_friction_range``
    and ``num_buckets`` are ignored.
    """

    def __init__(
        self, cfg: EventTermCfg, env: ManagerBasedEnv, asset: RigidObject | Articulation, asset_cfg: SceneEntityCfg
    ):
        self.asset = asset
        self.asset_cfg = asset_cfg
        self._notify_shape_properties = SolverNotifyFlags.SHAPE_PROPERTIES

        # cache friction/restitution ranges for continuous per-shape sampling
        self._static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        self._restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))

        # get friction/restitution view-level bindings
        model = NewtonManager.get_model()
        self._friction_binding = asset._root_view.get_attribute("shape_material_mu", model)[:, 0]  # type: ignore
        self._restitution_binding = asset._root_view.get_attribute("shape_material_restitution", model)[:, 0]  # type: ignore

        # compute shape indices for body-specific randomization
        if isinstance(asset, NewtonArticulation) and asset_cfg.body_ids != slice(None):
            num_shapes_per_body = asset.num_shapes_per_body
            shape_indices_list = []
            for body_id in asset_cfg.body_ids:
                start_idx = sum(num_shapes_per_body[:body_id])
                end_idx = start_idx + num_shapes_per_body[body_id]
                shape_indices_list.extend(range(start_idx, end_idx))
            self._shape_indices = torch.tensor(shape_indices_list, dtype=torch.long)
        else:
            self._shape_indices = torch.arange(self._friction_binding.shape[1], dtype=torch.long)

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
        device = env.device
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.int32)
        else:
            env_ids = env_ids.to(device)

        num_shapes = len(self._shape_indices)
        shape_idx = self._shape_indices.to(device)

        # sample friction (mu) and restitution continuously per shape
        friction_range = torch.tensor(self._static_friction_range, device=device)
        restitution_range_t = torch.tensor(self._restitution_range, device=device)
        friction_samples = math_utils.sample_uniform(
            friction_range[0], friction_range[1], (len(env_ids), num_shapes), device=device
        )
        restitution_samples = math_utils.sample_uniform(
            restitution_range_t[0], restitution_range_t[1], (len(env_ids), num_shapes), device=device
        )

        # write only the affected env_ids to the warp binding
        friction_view = wp.to_torch(self._friction_binding)
        restitution_view = wp.to_torch(self._restitution_binding)
        friction_view[env_ids[:, None], shape_idx] = friction_samples
        restitution_view[env_ids[:, None], shape_idx] = restitution_samples

        # notify the physics engine
        NewtonManager.add_model_change(self._notify_shape_properties)


class RandomizeRigidBodyColliderOffsets:
    """Newton backend implementation for collider offset randomization.

    Maps PhysX concepts to Newton's geometry properties:

    - ``rest_offset`` -> ``shape_margin`` (Newton margin)
    - ``contact_offset`` -> ``shape_gap`` (Newton gap = contact_offset - margin)

    See the `Newton collision schema`_ for details.

    .. _Newton collision schema: https://newton-physics.github.io/newton/latest/concepts/collisions.html
    """

    def __init__(self, asset: RigidObject | Articulation):
        self.asset = asset
        self._notify_shape_properties = SolverNotifyFlags.SHAPE_PROPERTIES

        model = NewtonManager.get_model()
        self._sim_bind_shape_margin = asset._root_view.get_attribute("shape_margin", model)[:, 0]  # type: ignore
        self._sim_bind_shape_gap = asset._root_view.get_attribute("shape_gap", model)[:, 0]  # type: ignore

        self.default_margin = wp.to_torch(self._sim_bind_shape_margin).clone()
        self.default_gap = wp.to_torch(self._sim_bind_shape_gap).clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        rest_offset_distribution_params: tuple[float, float] | None = None,
        contact_offset_distribution_params: tuple[float, float] | None = None,
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        device = env.device
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.int32)
        else:
            env_ids = env_ids.to(device)

        margin_view = wp.to_torch(self._sim_bind_shape_margin)

        if rest_offset_distribution_params is not None:
            margin = self.default_margin.clone()
            margin = randomize_prop_by_op(
                margin,
                rest_offset_distribution_params,
                None,
                slice(None),
                operation="abs",
                distribution=distribution,
            )
            self.default_margin[env_ids] = margin[env_ids]
            margin_view[env_ids] = margin[env_ids]

        if contact_offset_distribution_params is not None:
            current_margin = self.default_margin
            contact_offset = torch.zeros_like(self.default_gap)
            contact_offset = randomize_prop_by_op(
                contact_offset,
                contact_offset_distribution_params,
                None,
                slice(None),
                operation="abs",
                distribution=distribution,
            )
            gap = torch.clamp(contact_offset - current_margin, min=0.0)
            self.default_gap[env_ids] = gap[env_ids]
            gap_view = wp.to_torch(self._sim_bind_shape_gap)
            gap_view[env_ids] = gap[env_ids]

        if rest_offset_distribution_params is not None or contact_offset_distribution_params is not None:
            NewtonManager.add_model_change(self._notify_shape_properties)


class RandomizeRigidBodyCom:
    """Newton backend implementation for center-of-mass randomization.

    Passes position-only (vec3) to ``set_coms_index``. Note that on Newton (MuJoCo Warp),
    runtime CoM changes may cause simulation instability because
    ``notify_model_changed(BODY_INERTIAL_PROPERTIES)`` does not fully recompute the mass
    matrix after ``body_ipos`` changes. Use with caution until this is fixed upstream.
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

        # Newton expects position-only (vec3f)
        # note: pass partial data of shape (len(env_ids), len(body_ids), 3) to match the API
        self.asset.set_coms_index(coms=coms[env_ids[:, None], body_ids, :3], body_ids=body_ids, env_ids=env_ids)


class RandomizePhysicsSceneGravity:
    """Newton backend implementation for scene gravity randomization.

    Samples per-environment gravity vectors and writes them in-place to the Newton model's
    per-world gravity array on GPU.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        self._notify_model_properties = SolverNotifyFlags.MODEL_PROPERTIES

        distribution = cfg.params.get("distribution", "uniform")
        if distribution == "uniform":
            self._dist_fn = math_utils.sample_uniform
        elif distribution == "log_uniform":
            self._dist_fn = math_utils.sample_log_uniform
        elif distribution == "gaussian":
            self._dist_fn = math_utils.sample_gaussian
        else:
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

        # Newton applies per-environment gravity via the model's per-world gravity array on GPU
        model = NewtonManager.get_model()
        if model is None or model.gravity is None:
            raise RuntimeError("Newton model is not initialized. Cannot randomize gravity.")

        gravity = wp.to_torch(model.gravity)

        if env_ids is None:
            env_ids = env.scene._ALL_INDICES
        if len(env_ids) == 0:
            return

        num = len(env_ids)
        random_values = self._dist_fn(
            self._dist_param_0.unsqueeze(0).expand(num, -1),
            self._dist_param_1.unsqueeze(0).expand(num, -1),
            (num, 3),
            device=env.device,
        )

        if operation == "abs":
            gravity[env_ids] = random_values
        elif operation == "add":
            gravity[env_ids] += random_values
        elif operation == "scale":
            gravity[env_ids] *= random_values

        NewtonManager.add_model_change(self._notify_model_properties)


class RandomizeJointParameters:
    """Newton backend implementation for joint-parameter randomization.

    Newton supports a single (static) joint friction coefficient, so dynamic and viscous
    friction are not randomized. Joint armature and joint position limits are randomized as
    on PhysX.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv, asset: Articulation, asset_cfg: SceneEntityCfg):
        self.asset = asset
        self.asset_cfg = asset_cfg

        # cache default values
        self.default_joint_friction_coeff = self.asset.data.joint_friction_coeff.torch.clone()
        self.default_joint_armature = self.asset.data.joint_armature.torch.clone()
        self.default_joint_pos_limits = self.asset.data.joint_pos_limits.torch.clone()

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

        # joint friction coefficient (Newton only supports the static coefficient)
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

            self.asset.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=static_friction_coeff,
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
    """Newton backend implementation for fixed-tendon parameter randomization.

    Newton supports randomizing tendon stiffness and damping. Limit stiffness, position limits,
    rest length, and offset are not yet supported and raise :class:`NotImplementedError` when
    their distribution parameters are provided.
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

        # the following tendon properties are not yet supported on Newton
        if limit_stiffness_distribution_params is not None:
            raise NotImplementedError("Limit stiffness is not support in Newton.")
        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            raise NotImplementedError("Position limits is not yet implemented with Newton.")
        if rest_length_distribution_params is not None:
            raise NotImplementedError("Rest length is not yet implemented with Newton.")
        if offset_distribution_params is not None:
            raise NotImplementedError("Offset is not supported in Newton.")

        # write the fixed tendon properties into the simulation
        self.asset.write_fixed_tendon_properties_to_sim_index(env_ids=env_ids)


class RandomizeActuatorGains:
    """Newton backend implementation for actuator-gain randomization."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv, asset: Articulation, asset_cfg: SceneEntityCfg):
        self.asset: Articulation = asset
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
        # taken at articulation init time.
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

        # Push DR updates to explicit Newton-actuator controllers via the asset's own
        # write methods (Newton ``ArticulationView``).
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
    """Newton backend implementation for rigid-body scale randomization (unsupported).

    Scale randomization is not supported on the Newton backend.
    """

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        scale_range: tuple[float, float] | dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg,
        relative_child_path: str | None = None,
    ):
        raise NotImplementedError(
            "'randomize_rigid_body_scale' is not supported on the Newton backend. Prefer multi-asset"
            " spawning (MultiAssetSpawnerCfg) with per-scale USD variants instead."
        )
