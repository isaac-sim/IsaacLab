# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Success terms of the VLA benchmark tasks, following the RoboLab predicates.

Each term evaluates its predicate for all environments at once (see :mod:`.utils`); :class:`task_success` evaluates
the success terms of all tasks together, one evaluation per predicate, each environment with the parameters of the
task it runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .utils import InContainer, InOrder, OnTopOf, OutsideOf, Predicate, Scene, Stacked, entity_names, with_defaults

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg


class _PredicateTerm(ManagerTermBase):
    """Termination term of a predicate with the parameters of the term, which its ``__call__`` signature declares."""

    predicate: type[Predicate]

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._scene = Scene(env, entity_names([cfg.params]))
        self._predicate = self.predicate(self._scene, [with_defaults(type(self), cfg.params)])
        self._rows = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._predicate.reset(env_ids)

    def _evaluate(self) -> torch.Tensor:
        self._scene.update()
        return self._predicate(self._rows)


class object_in_container(_PredicateTerm):
    """Objects in an open-top container, as RoboLab's ``object_in_container``."""

    predicate = InContainer

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        objects: list[SceneEntityCfg],
        container: SceneEntityCfg,
        logical: str = "all",
        count: int = 1,
        require_contact: bool = False,
        require_gripper_detached: bool = False,
        require_upright: bool = False,
    ) -> torch.Tensor:
        """Whether the objects are in the container.

        Args:
            env: The environment.
            objects: The objects.
            container: The open-top container.
            logical: Whether ``"all"`` objects, ``"any"`` object, or exactly ``count`` of them (``"choose"``) count.
            count: The number of objects for ``"choose"``.
            require_contact: Whether each object must touch the container.
            require_gripper_detached: Whether each object must be released by the gripper.
            require_upright: Whether each object must stand upright (see :data:`~.utils.UPRIGHT_TOLERANCE`).
        """
        return self._evaluate()


class object_outside_of(_PredicateTerm):
    """Objects out of an open-top container, as RoboLab's ``object_outside_of``."""

    predicate = OutsideOf

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        objects: list[SceneEntityCfg],
        container: SceneEntityCfg,
        logical: str = "all",
        count: int = 1,
        require_gripper_detached: bool = False,
        require_upright: bool = False,
    ) -> torch.Tensor:
        """Whether the objects are out of the container.

        Args:
            env: The environment.
            objects: The objects.
            container: The open-top container.
            logical: As in :class:`object_in_container`.
            count: As in :class:`object_in_container`.
            require_gripper_detached: Whether each object must be released by the gripper.
            require_upright: Whether each object must stand upright (see :data:`~.utils.UPRIGHT_TOLERANCE`).
        """
        return self._evaluate()


class object_on_top(_PredicateTerm):
    """Objects supported on a reference object's top, as RoboLab's ``object_on_top``.

    The reference pushes the object up within 45 degrees of vertical, and the object's box center lies over the
    reference's box footprint with 1 cm slack.
    """

    predicate = OnTopOf

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        objects: list[SceneEntityCfg],
        reference: SceneEntityCfg,
        logical: str = "all",
        count: int = 1,
        require_gripper_detached: bool = False,
        require_upright: bool = False,
    ) -> torch.Tensor:
        """Whether the objects are on top of the reference.

        Args:
            env: The environment.
            objects: The objects.
            reference: The object under them.
            logical: As in :class:`object_in_container`.
            count: As in :class:`object_in_container`.
            require_gripper_detached: Whether each object must be released by the gripper.
            require_upright: Whether each object must stand upright (see :data:`~.utils.UPRIGHT_TOLERANCE`).
        """
        return self._evaluate()


class stacked(_PredicateTerm):
    """Objects stacked in any order, as RoboLab's ``stacked`` without an order.

    Sorted by height, each object touches the next lower one and lies above it: its box center over the lower box
    footprint (1 cm slack) and its box above the lower box top, or exactly over the footprint and above the bottom.
    """

    predicate = Stacked

    def __call__(self, env: ManagerBasedRLEnv, objects: list[SceneEntityCfg]) -> torch.Tensor:
        """Whether the objects are stacked.

        Args:
            env: The environment.
            objects: The objects.
        """
        return self._evaluate()


class objects_placed_in_container_in_order(_PredicateTerm):
    """Objects placed in a container in the listed order, as RoboLab's predicate of the same name.

    Remembers the step at which each object was first placed in the episode; holds when all objects are placed and
    their first placements do not decrease along the list.
    """

    predicate = InOrder

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        objects: list[SceneEntityCfg],
        container: SceneEntityCfg,
        require_contact: bool = True,
        require_gripper_detached: bool = True,
        require_upright: bool = False,
    ) -> torch.Tensor:
        """Whether the objects were placed in the container in their order.

        Args:
            env: The environment.
            objects: The objects, in the order of their placement.
            container: The open-top container.
            require_contact: Whether each object must touch the container.
            require_gripper_detached: Whether each object must be released by the gripper.
            require_upright: Whether each object must stand upright (see :data:`~.utils.UPRIGHT_TOLERANCE`).
        """
        return self._evaluate()


class task_success(ManagerTermBase):
    """Success of the task that each environment runs, by the ``success`` term of every task.

    The tasks' terms are grouped by predicate, and each predicate is evaluated once for all environments with the
    parameters of their tasks (see :meth:`~isaaclab.managers.HeterogeneousTaskManager.task_terms`).
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        terms = env.task_manager.task_terms("success")
        self._scene = Scene(env, entity_names([term.params for term in terms.values()]))
        groups: dict[type[Predicate], dict[int, dict]] = {}
        for task, term in terms.items():
            groups.setdefault(term.func.predicate, {})[task] = with_defaults(term.func, term.params)
        self._predicates = [predicate(self._scene, list(rows.values())) for predicate, rows in groups.items()]
        # Row of each task in each predicate, -1 for the tasks of other predicates.
        self._rows = torch.full((len(groups), len(env.task_manager.task_names)), -1, dtype=torch.long)
        for group, tasks in enumerate(groups.values()):
            self._rows[group, list(tasks)] = torch.arange(len(tasks))
        self._rows = self._rows.to(env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        for predicate in self._predicates:
            predicate.reset(env_ids)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        self._scene.update()
        success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for predicate, rows in zip(self._predicates, self._rows):
            rows = rows[env.task_manager.task_ids]
            success |= predicate(rows) & (rows >= 0)
        return success
