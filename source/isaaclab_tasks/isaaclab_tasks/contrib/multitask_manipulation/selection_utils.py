# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selection-aware scene entities for heterogeneous physics views."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.cloner import path as cloner_path
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.warp import ProxyArray

if TYPE_CHECKING:
    from isaaclab.assets import BaseArticulation, BaseRigidObject, BaseRigidObjectCollection
    from isaaclab.scene import InteractiveScene


@configclass
class SceneEntitySelectionCfg(SceneEntityCfg):
    """Scene entity resolved across a heterogeneous subset of environments.

    In addition to resolving joints, bodies, tendons, and collection objects like
    :class:`~isaaclab.managers.SceneEntityCfg`, this configuration maps global environment IDs to
    the instance rows of the entity's physics view. Environments that do not contain the entity map
    to ``-1``.
    """

    env_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long), init=False, repr=False)
    """Global environment ID of each physics-view instance row, shape ``(num_instances,)``."""

    instance_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long), init=False, repr=False)
    """Physics-view instance row of each environment, or ``-1`` if absent, shape ``(num_envs,)``."""

    row_mask: ProxyArray | None = field(default=None, init=False, repr=False)
    """Reused buffer of :meth:`view_row_mask`, shape ``(num_instances,)``."""

    def resolve(self, scene: InteractiveScene):
        """Resolve entity members and its environment-to-instance mapping."""
        super().resolve(scene)
        entity: BaseArticulation | BaseRigidObject | BaseRigidObjectCollection = scene[self.name]
        self.env_ids = _view_env_ids(entity, scene.cloner_cfg.clone_template)
        self.instance_ids = torch.full((scene.num_envs,), -1, dtype=torch.long, device=entity.device)
        self.instance_ids[self.env_ids] = torch.arange(entity.num_instances, dtype=torch.long, device=entity.device)

    def select(self, env_ids: torch.Tensor, *, strict: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Map global environment IDs to physics-view instance rows.

        Args:
            env_ids: Global environment IDs to select. Moved to the resolved entity's device.
            strict: Whether to reject environments that do not contain the entity. Defaults to False.

        Returns:
            A tuple containing selected instance rows and their aligned global environment IDs on the
            resolved entity's device.

        Raises:
            ValueError: If ``strict`` is enabled and an environment does not contain the entity.
        """
        env_ids = env_ids.to(self.instance_ids.device)
        instance_ids = self.instance_ids[env_ids]
        selected = instance_ids >= 0
        if strict and not bool(selected.all()):
            raise ValueError(f"Environments {env_ids[~selected].tolist()} contain no '{self.name}'.")
        return instance_ids[selected], env_ids[selected]

    def view_row_mask(self, env_mask: ProxyArray) -> wp.array:
        """Select the physics-view instance rows of the environments in ``env_mask``.

        The rows are gathered on the device into a buffer reused across calls, so the mask stays valid until the
        next call.

        Args:
            env_mask: Boolean mask of the global environments. Shape is (num_envs,).

        Returns:
            Boolean row mask, dtype ``wp.bool``. Shape is (num_instances,).
        """
        if self.row_mask is None:
            device = str(self.env_ids.device)
            self.row_mask = ProxyArray(wp.zeros(self.env_ids.numel(), dtype=wp.bool, device=device))
        torch.index_select(env_mask.torch, 0, self.env_ids, out=self.row_mask.torch)
        return self.row_mask.warp

    def scatter_to_envs(self, values: torch.Tensor, *, fill_value: float | int | bool = 0) -> torch.Tensor:
        """Scatter physics-view values into global environment order.

        Args:
            values: Values whose first dimension follows this entity's physics-view instance order.
            fill_value: Value assigned to environments that do not contain the entity. Defaults to 0.

        Returns:
            Values in global environment order on the same device as ``values``.

        Raises:
            ValueError: If the values do not have an instance dimension or its length does not match
                the entity's physics view.
        """
        if values.ndim == 0:
            raise ValueError("Expected values to have an instance dimension.")
        if values.shape[0] != self.env_ids.numel():
            raise ValueError(f"Expected {self.env_ids.numel()} values for '{self.name}', got {values.shape[0]}.")

        result = values.new_full((self.instance_ids.numel(), *values.shape[1:]), fill_value)
        result[self.env_ids.to(values.device)] = values
        return result


def _view_env_ids(
    entity: BaseArticulation | BaseRigidObject | BaseRigidObjectCollection, env_template: str
) -> torch.Tensor:
    """Read the global environment ID of every physics-view instance row."""
    env_ids = []
    for prim_path in entity.root_view.prim_paths[: entity.num_instances]:
        matched = cloner_path.match(prim_path, env_template)
        if matched is None:
            raise ValueError(f"Prim path '{prim_path}' is not under the environment template '{env_template}'.")
        env_ids.append(int(matched.instance))
    return torch.tensor(env_ids, dtype=torch.long, device=entity.device)
