# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit physics randomization terms for Newton."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

from isaaclab.envs.mdp._randomization import _GravityRandomization, _randomize_prop_by_op
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


class randomize_rigid_body_material(ManagerTermBase):
    """Newton backend implementation for material randomization.

    Newton can assign arbitrary friction/restitution per shape (no bucket limitation).
    Samples friction (mu) and restitution continuously from the given ranges.
    ``friction_range`` samples the single friction coefficient (mu).

    The Kamino solver deduplicates contact materials globally by ``(mu, restitution)`` and
    shares them across environments, so it cannot accept per-shape or per-env overrides. When
    Kamino is active, one value is sampled per build-time material group and broadcast to every
    environment (no per-env variation). All other Newton solvers keep the per-shape sampling.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Bind this term to the active simulation and capture term-local state.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        from newton import ModelFlags  # noqa: PLC0415
        from newton.solvers import SolverKamino  # noqa: PLC0415

        from ...assets import Articulation as NewtonArticulation  # noqa: PLC0415

        self.asset = asset
        self.asset_cfg = asset_cfg
        self._newton_manager = env.sim.physics_manager
        self._notify_shape_properties = ModelFlags.SHAPE_PROPERTIES
        # Kamino deduplicates contact materials globally by (mu, restitution) at build time and
        # shares them across environments, so its in-place material update rejects per-shape /
        # per-env overrides. When Kamino is active we instead sample one value per build-time
        # material group and broadcast it to every environment. The grouping is derived lazily on
        # the first call, when the shape bindings still hold their build-time values.
        self._solver_kamino_cls = SolverKamino
        self._kamino_group_inverse: torch.Tensor | None = None
        self._kamino_num_groups = 0

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
        env_ids: torch.Tensor | slice | None,
        friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        asset_cfg: SceneEntityCfg,
    ):
        """Apply the configured randomization.

        Args:
            env: Environment owning this term.
            env_ids: Environment selection; None selects all environments.
            friction_range: Bounds for the single Newton friction coefficient.
            restitution_range: Restitution bounds; bucket-based backends use the construction-time range.
            asset_cfg: Asset and body selection resolved at construction.
        """
        device = env.device
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        env_rows = env_ids if isinstance(env_ids, slice) else env_ids[:, None]

        num_shapes = len(self._shape_indices)
        shape_idx = self._shape_indices.to(device)

        friction_range = torch.tensor(friction_range, device=device)
        restitution_range_t = torch.tensor(restitution_range, device=device)
        friction_view = wp.to_torch(self._friction_binding)
        restitution_view = wp.to_torch(self._restitution_binding)

        num_envs = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
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
                friction_range[0], friction_range[1], (num_envs, num_shapes), device=device
            )
            restitution_samples = math_utils.sample_uniform(
                restitution_range_t[0], restitution_range_t[1], (num_envs, num_shapes), device=device
            )
            # write only the affected env_ids to the warp binding
            friction_view[env_rows, shape_idx] = friction_samples
            restitution_view[env_rows, shape_idx] = restitution_samples

        # notify the physics engine
        self._newton_manager.add_model_change(self._notify_shape_properties)


class randomize_rigid_body_collider_parameters(ManagerTermBase):
    """Randomize Newton shape margins and gaps independently [m].

    Every collision shape of the selected asset environments is updated. Body selection
    is unsupported. These are native Newton parameters, without a PhysX offset conversion.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Bind this term to the active simulation and capture term-local state.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        from newton import ModelFlags  # noqa: PLC0415

        self.asset = asset
        self._newton_manager = env.sim.physics_manager
        self._notify_shape_properties = ModelFlags.SHAPE_PROPERTIES

        model = self._newton_manager.get_model()
        self._sim_bind_shape_margin = asset._root_view.get_attribute("shape_margin", model)[:, 0]  # type: ignore
        self._sim_bind_shape_gap = asset._root_view.get_attribute("shape_gap", model)[:, 0]  # type: ignore

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        margin_distribution_params: tuple[float, float] | None = None,
        gap_distribution_params: tuple[float, float] | None = None,
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ) -> None:
        """Apply the configured randomization.

        Args:
            env: Environment owning this term.
            env_ids: Environment selection; None selects all environments.
            asset_cfg: Asset selection; collider randomization operates on every body.
            margin_distribution_params: Shape margin distribution parameters [m].
            gap_distribution_params: Shape gap distribution parameters [m].
            distribution: Sampling distribution; gravity terms cache this at construction.
        """
        if asset_cfg.body_ids != slice(None):
            raise ValueError("Newton collider randomization requires all bodies of the asset.")
        if env_ids is None:
            env_ids = slice(None)
        changed = False
        for binding, params in (
            (self._sim_bind_shape_margin, margin_distribution_params),
            (self._sim_bind_shape_gap, gap_distribution_params),
        ):
            if params is None:
                continue
            values = wp.to_torch(binding)
            samples = values[env_ids].clone()
            _randomize_prop_by_op(samples, params, None, slice(None), "abs", distribution)
            values[env_ids] = samples
            changed = True
        if changed:
            self._newton_manager.add_model_change(self._notify_shape_properties)


class randomize_world_gravity(_GravityRandomization):
    """Randomize selected Newton worlds, leaving the global world unchanged.

    Add and scale operate on current gravity; repeated calls accumulate. Distribution
    is fixed at construction; distribution parameters [m/s^2] may change at runtime.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Bind this term to the active simulation and capture term-local state.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env, device=env.device)
        self._manager = env.sim.physics_manager

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        gravity_distribution_params: tuple[list[float], list[float]],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ) -> None:
        """Apply the configured randomization.

        Args:
            env: Environment owning this term.
            env_ids: Environment selection; None selects all environments.
            gravity_distribution_params: Distribution parameters [m/s^2] for add/abs, dimensionless for scale.
            operation: Apply absolute values, add to the baseline, or scale the baseline.
            distribution: Sampling distribution; gravity terms cache this at construction.
        """
        from newton import ModelFlags

        model = self._manager.get_model()
        if model is None or model.gravity is None:
            raise RuntimeError("Newton model is not initialized. Cannot randomize gravity.")
        # The trailing global-world row is not an environment.
        gravity = wp.to_torch(model.gravity)[: env.num_envs]
        if env_ids is None:
            env_ids = slice(None)
        selected = gravity[env_ids]
        if selected.shape[0] == 0:
            return
        gravity[env_ids] = self._sample_gravity(selected, gravity_distribution_params, operation)
        self._manager.add_model_change(ModelFlags.MODEL_PROPERTIES)


class _legacy_randomize_rigid_body_material(randomize_rigid_body_material):
    """Preserve the old PhysX-shaped signature and construction-time Newton ranges."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        params = dict(cfg.params)
        params["friction_range"] = params.pop("static_friction_range", (1.0, 1.0))
        super().__init__(cfg.replace(params=params), env)
        self._static_friction_range = params["friction_range"]
        self._restitution_range = params.get("restitution_range", (0.0, 0.0))

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ) -> None:
        super().__call__(env, env_ids, self._static_friction_range, self._restitution_range, asset_cfg)


class _legacy_randomize_rigid_body_collider_offsets(randomize_rigid_body_collider_parameters):
    """Preserve the deprecated PhysX-offset to Newton margin/gap conversion."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.default_margin = wp.to_torch(self._sim_bind_shape_margin).clone()
        self.default_gap = wp.to_torch(self._sim_bind_shape_gap).clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        rest_offset_distribution_params: tuple[float, float] | None = None,
        contact_offset_distribution_params: tuple[float, float] | None = None,
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        if env_ids is None:
            env_ids = slice(None)

        margin_view = wp.to_torch(self._sim_bind_shape_margin)

        if rest_offset_distribution_params is not None:
            margin = self.default_margin.clone()
            margin = _randomize_prop_by_op(
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
            contact_offset = _randomize_prop_by_op(
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
