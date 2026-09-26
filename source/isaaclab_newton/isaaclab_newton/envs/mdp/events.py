# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend implementations of MDP event terms for Newton."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import warp as wp
from newton import ModelFlags
from newton.solvers import SolverKamino

from isaaclab.envs.mdp.events import _GravityRandomization, _randomize_prop_by_op
from isaaclab.envs.mdp.visual_events import _compile_distribution
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

from ... import assets
from ...physics.newton_manager import NewtonManager

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


class randomize_rigid_body_material(ManagerTermBase):
    """Sample friction and restitution per shape.

    Newton uses one friction coefficient, so ``dynamic_friction_range``, ``num_buckets``,
    and ``make_consistent`` are ignored.

    Kamino shares materials across environments. It samples one value per original
    ``(mu, restitution)`` group and applies it to every environment, ignoring ``env_ids``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize the asset bindings and sampling state.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        self.asset = asset
        self.asset_cfg = asset_cfg
        self._newton_manager = env.sim.physics_manager
        # Capture material groups on the first call, before any randomized writes.
        self._kamino_group_inverse: torch.Tensor | None = None
        self._kamino_num_groups = 0

        self._static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        self._restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))

        model = self._newton_manager.get_model()
        self._friction_binding = asset._root_view.get_attribute("shape_material_mu", model)[:, 0]  # type: ignore
        self._restitution_binding = asset._root_view.get_attribute("shape_material_restitution", model)[:, 0]  # type: ignore

        if isinstance(asset, assets.Articulation) and asset_cfg.body_ids != slice(None):
            # Shape counts use backend body order.
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
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ) -> None:
        """Sample friction and restitution for the selected shapes.

        Args:
            env: Environment owning this term.
            env_ids: Environment selection; ignored by Kamino's shared material groups.
                None selects all environments.
            static_friction_range: Friction bounds captured at construction.
            dynamic_friction_range: Unused; Newton has a single friction coefficient.
            restitution_range: Restitution bounds captured at construction.
            num_buckets: Unused; Newton samples continuous values.
            asset_cfg: Asset and body selection resolved at construction.
            make_consistent: Unused; Newton has a single friction coefficient.
        """
        device = env.device
        if env_ids is None:
            env_ids = slice(None)
        env_rows = env_ids if isinstance(env_ids, slice) else env_ids[:, None]

        num_shapes = len(self._shape_indices)
        shape_idx = self._shape_indices.to(device)

        friction_range = torch.tensor(self._static_friction_range, device=device)
        restitution_range_t = torch.tensor(self._restitution_range, device=device)
        friction_view = wp.to_torch(self._friction_binding)
        restitution_view = wp.to_torch(self._restitution_binding)

        num_envs = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        if isinstance(self._newton_manager._solver, SolverKamino):
            # Kamino shares each material group across all environments.
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
            friction_samples = math_utils.sample_uniform(
                friction_range[0], friction_range[1], (num_envs, num_shapes), device=device
            )
            restitution_samples = math_utils.sample_uniform(
                restitution_range_t[0], restitution_range_t[1], (num_envs, num_shapes), device=device
            )
            friction_view[env_rows, shape_idx] = friction_samples
            restitution_view[env_rows, shape_idx] = restitution_samples

        self._newton_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)


class randomize_rigid_body_collider_offsets(ManagerTermBase):
    """Newton backend implementation for collider offset randomization.

    Maps PhysX concepts to Newton's geometry properties:

    - ``rest_offset`` -> ``shape_margin`` (Newton margin)
    - ``contact_offset`` -> ``shape_gap`` (Newton gap = contact_offset - margin)

    See the `Newton collision schema`_ for details.

    .. _Newton collision schema: https://newton-physics.github.io/newton/latest/concepts/collisions.html
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize the asset bindings and sampling state.

        Args:
            cfg: Event configuration.
            env: Environment owning this term.
        """
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        self.asset = asset
        self._newton_manager = env.sim.physics_manager

        model = self._newton_manager.get_model()
        self._sim_bind_shape_margin = asset._root_view.get_attribute("shape_margin", model)[:, 0]  # type: ignore
        self._sim_bind_shape_gap = asset._root_view.get_attribute("shape_gap", model)[:, 0]  # type: ignore

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
    ) -> None:
        """Sample offsets and translate them to Newton margins and gaps.

        Args:
            env: Environment owning this term.
            env_ids: Environment selection; None selects all environments.
            asset_cfg: Asset selection; collider randomization operates on every body.
            rest_offset_distribution_params: Rest offset distribution parameters [m].
            contact_offset_distribution_params: Contact offset distribution parameters [m].
            distribution: Sampling distribution for the offsets.
        """
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
            self._newton_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)


class randomize_physics_scene_gravity(_GravityRandomization):
    """Randomize selected Newton worlds, leaving the global world unchanged.

    Add and scale operate on current gravity; repeated calls accumulate. Distribution
    is fixed at construction; distribution parameters [m/s^2] may change at runtime.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        """Initialize gravity sampling for the active simulation.

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
        """Sample and set gravity.

        Args:
            env: Environment owning this term.
            env_ids: Environment selection; None selects all environments.
            gravity_distribution_params: Distribution parameters [m/s^2] for add/abs, dimensionless for scale.
            operation: Apply absolute values, add to current gravity, or scale current gravity.
            distribution: Sampling distribution; gravity terms cache this at construction.
        """
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


class randomize_visual_shape(ManagerTermBase):
    """Sample one color per selected link and write Newton shape storage on device."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        channels = cfg.params["channels"]
        if tuple(channels) != ("color",):
            raise NotImplementedError("Newton per-shape randomization currently supports only the 'color' channel.")
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        if isinstance(asset_cfg.body_ids, slice):
            ids = range(asset.num_bodies)[asset_cfg.body_ids]
        else:
            ids = asset_cfg.body_ids
        body_names = tuple(asset.body_names[index] for index in ids)
        self._writer = NewtonManager.create_visual_shape_color_writer(asset, body_names)
        self._sample = _compile_distribution(channels["color"], env.device)
        self._all_env_ids = torch.arange(env.num_envs, dtype=torch.int32, device=self._writer.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        channels: dict[str, tuple | dict],
    ) -> None:
        del asset_cfg, channels
        if env_ids is None:
            env_ids = slice(None)
        selected = self._all_env_ids[env_ids] if isinstance(env_ids, slice) else env_ids.to(dtype=torch.int32)
        model = NewtonManager.get_model()
        if self._writer.model is not model:
            self._writer.rebind(model)
        self._writer(self._sample((len(selected), self._writer.body_count)), selected)
