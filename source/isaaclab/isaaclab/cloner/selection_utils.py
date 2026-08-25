# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selection-aware scene entities for heterogeneous physics views."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .path import match

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
            env_ids: Global environment IDs to select.
            strict: Whether to reject environments that do not contain the entity. Defaults to False.

        Returns:
            A tuple containing selected instance rows and their aligned global environment IDs.

        Raises:
            ValueError: If ``strict`` is enabled and an environment does not contain the entity.
        """
        instance_ids = self.instance_ids[env_ids]
        selected = instance_ids >= 0
        if strict and not bool(selected.all()):
            raise ValueError(f"Environments {env_ids[~selected].tolist()} contain no '{self.name}'.")
        return instance_ids[selected], env_ids[selected]

    def scatter_to_envs(self, values: torch.Tensor, *, fill_value: float | int | bool = 0) -> torch.Tensor:
        """Scatter physics-view values into global environment order.

        Args:
            values: Values whose first dimension follows this entity's physics-view instance order.
            fill_value: Value assigned to environments that do not contain the entity. Defaults to 0.

        Returns:
            Values in global environment order.

        Raises:
            ValueError: If the values do not have an instance dimension or its length does not match
                the entity's physics view.
        """
        if values.ndim == 0:
            raise ValueError("Expected values to have an instance dimension.")
        if values.shape[0] != self.env_ids.numel():
            raise ValueError(f"Expected {self.env_ids.numel()} values for '{self.name}', got {values.shape[0]}.")

        result = values.new_full((self.instance_ids.numel(), *values.shape[1:]), fill_value)
        result[self.env_ids] = values
        return result


def _view_env_ids(
    entity: BaseArticulation | BaseRigidObject | BaseRigidObjectCollection, env_template: str
) -> torch.Tensor:
    """Read the global environment ID of every physics-view instance row."""
    env_ids = []
    for prim_path in entity.root_view.prim_paths[: entity.num_instances]:
        matched = match(prim_path, env_template)
        if matched is None:
            raise ValueError(f"Prim path '{prim_path}' is not under the environment template '{env_template}'.")
        env_ids.append(int(matched.instance))
    return torch.tensor(env_ids, dtype=torch.long, device=entity.device)
