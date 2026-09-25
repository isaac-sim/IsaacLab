# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton backend implementations for MDP event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import warp as wp
from newton import ModelFlags

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.physics_events import randomize_prop_by_op

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

    The Kamino solver deduplicates contact materials globally by ``(mu, restitution)`` and
    shares them across environments, so it cannot accept per-shape or per-env overrides. When
    Kamino is active, one value is sampled per build-time material group and broadcast to every
    environment (no per-env variation). All other Newton solvers keep the per-shape sampling.
    """

    def __init__(
        self, cfg: EventTermCfg, env: ManagerBasedEnv, asset: RigidObject | Articulation, asset_cfg: SceneEntityCfg
    ):
        from newton import ModelFlags  # noqa: PLC0415
        from newton.solvers import SolverKamino  # noqa: PLC0415

        import isaaclab_newton.physics.newton_manager as newton_manager_module  # noqa: PLC0415

        self.asset = asset
        self.asset_cfg = asset_cfg
        self._newton_manager = newton_manager_module.NewtonManager
        self._notify_shape_properties = ModelFlags.SHAPE_PROPERTIES
        # Kamino deduplicates contact materials globally by (mu, restitution) at build time and
        # shares them across environments, so its in-place material update rejects per-shape /
        # per-env overrides. When Kamino is active we instead sample one value per build-time
        # material group and broadcast it to every environment. The grouping is derived lazily on
        # the first call, when the shape bindings still hold their build-time values.
        self._solver_kamino_cls = SolverKamino
        self._kamino_group_inverse: torch.Tensor | None = None
        self._kamino_num_groups = 0

        # cache friction/restitution ranges for continuous per-shape sampling
        self._static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        self._restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))

        # get friction/restitution view-level bindings
        model = self._newton_manager.get_model()
        self._friction_binding = asset._root_view.get_attribute("shape_material_mu", model)[:, 0]  # type: ignore
        self._restitution_binding = asset._root_view.get_attribute("shape_material_restitution", model)[:, 0]  # type: ignore

        # compute shape indices for body-specific randomization
        if isinstance(asset, NewtonArticulation) and asset_cfg.body_ids != slice(None):
            # ``body_ids`` are public IDs, while shape bindings use backend order; convert the
            # selected IDs once and index the backend-ordered shape counts directly.
            num_shapes_per_body = asset.backend_num_shapes_per_body
            shape_indices_list = []
            backend_body_ids = asset.map_body_ids_to_backend(asset_cfg.body_ids)
            for body_id in backend_body_ids:
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

        friction_range = torch.tensor(self._static_friction_range, device=device)
        restitution_range_t = torch.tensor(self._restitution_range, device=device)
        friction_view = wp.to_torch(self._friction_binding)
        restitution_view = wp.to_torch(self._restitution_binding)

        if isinstance(self._newton_manager._solver, self._solver_kamino_cls):
            # Kamino: sample one value per build-time material group and broadcast across every
            # environment. Per-shape / per-env variation is impossible because Kamino shares each
            # contact material across all shapes and environments that were built with identical
            # (mu, restitution).
            if self._kamino_group_inverse is None:
                build_keys = torch.stack((friction_view[0, shape_idx], restitution_view[0, shape_idx]), dim=-1)
                _, inverse = torch.unique(build_keys, dim=0, return_inverse=True)
                self._kamino_group_inverse = inverse
                self._kamino_num_groups = int(inverse.max().item()) + 1 if inverse.numel() else 0
            inverse = self._kamino_group_inverse
            friction_groups = math_utils.sample_uniform(
                friction_range[0], friction_range[1], (self._kamino_num_groups,), device=device
            )
            restitution_groups = math_utils.sample_uniform(
                restitution_range_t[0], restitution_range_t[1], (self._kamino_num_groups,), device=device
            )
            friction_view[:, shape_idx] = friction_groups[inverse]
            restitution_view[:, shape_idx] = restitution_groups[inverse]
        else:
            # sample friction (mu) and restitution continuously per shape
            friction_samples = math_utils.sample_uniform(
                friction_range[0], friction_range[1], (len(env_ids), num_shapes), device=device
            )
            restitution_samples = math_utils.sample_uniform(
                restitution_range_t[0], restitution_range_t[1], (len(env_ids), num_shapes), device=device
            )
            # write only the affected env_ids to the warp binding
            friction_view[env_ids[:, None], shape_idx] = friction_samples
            restitution_view[env_ids[:, None], shape_idx] = restitution_samples

        # notify the physics engine
        self._newton_manager.add_model_change(self._notify_shape_properties)


class RandomizeRigidBodyColliderOffsets:
    """Newton backend implementation for collider offset randomization.

    Maps PhysX concepts to Newton's geometry properties:

    - ``rest_offset`` -> ``shape_margin`` (Newton margin)
    - ``contact_offset`` -> ``shape_gap`` (Newton gap = contact_offset - margin)

    See the `Newton collision schema`_ for details.

    .. _Newton collision schema: https://newton-physics.github.io/newton/latest/concepts/collisions.html
    """

    def __init__(self, asset: RigidObject | Articulation):
        from newton import ModelFlags  # noqa: PLC0415

        import isaaclab_newton.physics.newton_manager as newton_manager_module  # noqa: PLC0415

        self.asset = asset
        self._newton_manager = newton_manager_module.NewtonManager
        self._notify_shape_properties = ModelFlags.SHAPE_PROPERTIES

        model = self._newton_manager.get_model()
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
            self._newton_manager.add_model_change(self._notify_shape_properties)


class RandomizePhysicsSceneGravity:
    """Newton backend implementation for scene gravity randomization.

    Samples per-environment gravity vectors and writes them in-place to the Newton model's
    per-world gravity array on GPU.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        self._notify_model_properties = ModelFlags.MODEL_PROPERTIES

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
