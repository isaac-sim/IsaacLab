# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend implementations of MDP event terms for Newton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import warp as wp
from newton import Model, ModelFlags
from newton.selection import ArticulationView
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

    A rigid object whose shape count differs between environments is sampled through one view per
    group of environments with equal shape counts.
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

        self._static_friction_range = cfg.params["static_friction_range"]
        self._restitution_range = cfg.params["restitution_range"]

        model = self._newton_manager.get_model()
        view: ArticulationView = asset._root_view  # type: ignore
        if view.shape_count is None:
            self._shape_groups = _split_view_by_shape_count(view, model)
        elif isinstance(asset, assets.Articulation) and asset_cfg.body_ids != slice(None):
            # The view's shape axis follows the model's shape order, which is not grouped by body,
            # so select each backend body's own shape indices.
            backend_body_ids = asset.map_body_ids_to_backend(asset_cfg.body_ids)
            shape_indices = [shape_id for body_id in backend_body_ids for shape_id in view.body_shapes[body_id]]
            self._shape_groups = [_ShapeGroup(view, None, torch.tensor(shape_indices, dtype=torch.long))]
        else:
            self._shape_groups = [_ShapeGroup(view, None, torch.arange(view.shape_count, dtype=torch.long))]

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
        if env_ids is None:
            env_ids = slice(None)
        all_envs = isinstance(env_ids, slice) and env_ids == slice(None)
        for group in self._shape_groups:
            group_env_ids = env_ids
            if group.env_ids is not None and not all_envs:
                selected = (
                    torch.arange(env.num_envs, device=env.device)[env_ids] if isinstance(env_ids, slice) else env_ids
                )
                group_env_ids = torch.isin(group.env_ids.to(env.device), selected).nonzero().flatten()
                if group_env_ids.numel() == 0:
                    continue
            self._randomize_group(group, group_env_ids, env.device)
        self._newton_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)

    def _randomize_group(self, group: _ShapeGroup, env_ids: torch.Tensor | slice, device: str) -> None:
        """Sample the selected rows of one shape group and write them to the model.

        Args:
            group: Shape group to sample.
            env_ids: Rows of the group's view to sample.
            device: Device of the samples.
        """
        env_rows = env_ids if isinstance(env_ids, slice) else env_ids[:, None]
        num_shapes = len(group.shape_indices)
        shape_idx = group.shape_indices.to(device)

        friction_range = torch.tensor(self._static_friction_range, device=device)
        restitution_range_t = torch.tensor(self._restitution_range, device=device)
        # Views of shapes that are not regularly spaced between worlds return gathered copies, which
        # are read here and scattered back below.
        model = self._newton_manager.get_model()
        friction = group.view.get_attribute("shape_material_mu", model)
        restitution = group.view.get_attribute("shape_material_restitution", model)
        friction_view = wp.to_torch(friction)[:, 0]
        restitution_view = wp.to_torch(restitution)[:, 0]

        num_envs = len(range(group.view.count)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        if isinstance(self._newton_manager._solver, SolverKamino):
            # Kamino shares each material group across all environments.
            # Capture material groups on the first call, before any randomized writes.
            if group.kamino_group_inverse is None:
                build_keys = torch.stack((friction_view[0, shape_idx], restitution_view[0, shape_idx]), dim=-1)
                _, group.kamino_group_inverse = torch.unique(build_keys, dim=0, return_inverse=True)
            inverse = group.kamino_group_inverse
            num_groups = int(inverse.max().item()) + 1 if inverse.numel() else 0
            friction_groups = math_utils.sample_uniform(
                friction_range[0], friction_range[1], (num_groups,), device=device
            )
            restitution_groups = math_utils.sample_uniform(
                restitution_range_t[0], restitution_range_t[1], (num_groups,), device=device
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

        group.view.set_attribute("shape_material_mu", model, friction)
        group.view.set_attribute("shape_material_restitution", model, restitution)


@dataclass
class _ShapeGroup:
    """Shapes that one view addresses with equal shape counts in each of its worlds."""

    view: ArticulationView
    """View of the group's articulations."""

    env_ids: torch.Tensor | None
    """Rows of the asset's view that the group's view covers, in order. None when the views are the same."""

    shape_indices: torch.Tensor
    """Shape indices along the group view's shape axis."""

    kamino_group_inverse: torch.Tensor | None = None
    """Kamino material group of each selected shape, captured on the first call."""


def _split_view_by_shape_count(view: ArticulationView, model: Model) -> list[_ShapeGroup]:
    """Split a view whose shape count differs between its worlds into views with equal shape counts.

    Args:
        view: View with one articulation per world and no common shape layout.
        model: Model the view selects from.

    Returns:
        One group per distinct shape count, in order of first occurrence.
    """
    articulation_ids = view.articulation_ids.numpy()[:, 0]
    joint_child = model.joint_child.numpy()
    articulation_start = model.articulation_start.numpy()
    articulation_end = model.articulation_end.numpy()
    shape_counts = [
        sum(
            len(model.body_shapes.get(int(body), ()))
            for body in joint_child[articulation_start[a] : articulation_end[a]]
        )
        for a in articulation_ids
    ]
    groups = []
    for shape_count in dict.fromkeys(shape_counts):
        rows = [row for row, count in enumerate(shape_counts) if count == shape_count]
        group_view = ArticulationView(model, articulation_ids[rows].tolist(), verbose=False)
        groups.append(
            _ShapeGroup(group_view, torch.tensor(rows, dtype=torch.long), torch.arange(shape_count, dtype=torch.long))
        )
    return groups


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
        self.default_margin = wp.to_torch(asset._root_view.get_attribute("shape_margin", model))[:, 0].clone()  # type: ignore
        self.default_gap = wp.to_torch(asset._root_view.get_attribute("shape_gap", model))[:, 0].clone()  # type: ignore

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

        # Views of shapes that are not regularly spaced between worlds return gathered copies, which
        # are read here and scattered back below.
        view = self.asset._root_view
        model = self._newton_manager.get_model()

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
            margin_binding = view.get_attribute("shape_margin", model)  # type: ignore
            wp.to_torch(margin_binding)[:, 0][env_ids] = margin[env_ids]
            view.set_attribute("shape_margin", model, margin_binding)  # type: ignore
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
            gap_binding = view.get_attribute("shape_gap", model)  # type: ignore
            wp.to_torch(gap_binding)[:, 0][env_ids] = gap[env_ids]
            view.set_attribute("shape_gap", model, gap_binding)  # type: ignore
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
