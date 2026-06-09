# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for task modules used in multitask environments.

A :class:`TaskModuleCfg` encapsulates the task-specific MDP components:
scene objects, goal commands, task-specific observations, rewards,
termination conditions, and reset events.  All builder methods receive both
the ``group`` string and a :class:`~...robots._base.RobotModuleCfg` so the
generated configs can reference the correct robot asset names and joint
patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.managers import EventTermCfg, ManagerTermBaseCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg

    from ..robots._base import RobotModuleCfg


@dataclass
class TaskModuleCfg(ABC):
    """Abstract dataclass defining a task's MDP contribution.

    Subclasses implement all abstract methods.  Instance attributes
    control task hyperparameters (object spawn ranges, reward weights, etc.)
    without requiring subclassing.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical task identifier (e.g. ``"reach"``, ``"lift"``, ``"cabinet"``).

        Used to derive default command names (``f"{group}_{name}"``).
        Must be a valid Python identifier.
        """
        ...

    @abstractmethod
    def scene_assets(self, group: str, robot: "RobotModuleCfg") -> dict[str, object]:
        """Return task-specific scene asset configs.

        Typically includes manipulated objects (cube, cabinet) and their
        associated sensors (frame transformers).  Return an empty dict
        for tasks that add no scene assets (e.g. pure reach).

        Asset names should be globally unique; the convention
        ``f"{robot.name}_{asset_suffix}"`` avoids collisions when the
        same task is registered with different robots.

        Args:
            group: Clone-group name.
            robot: Robot module for this registration.

        Returns:
            Mapping from scene attribute name to asset config.
        """
        ...

    @abstractmethod
    def command_terms(self, group: str, robot: "RobotModuleCfg") -> dict[str, object]:
        """Return goal-command configs for this task.

        The command name used as a key is referenced by obs / reward terms.
        The canonical naming convention is ``f"{group}_{self.name}"``
        (e.g. ``"franka_reach"``, ``"openarm_lift"``).

        Args:
            group: Clone-group name.
            robot: Robot module.

        Returns:
            Mapping from command name to command cfg.
        """
        ...

    @abstractmethod
    def task_obs_terms(self, group: str, robot: "RobotModuleCfg") -> dict[str, "ObservationTermCfg"]:
        """Return group-local observation terms specific to this task.

        These terms use the ``@scatterable`` decorator internally and return
        zeros automatically for environments that belong to other groups.
        Keys are *local* names; the registry namespaces them as
        ``f"{group}_{name}"`` in the final policy.

        Args:
            group: Clone-group name.
            robot: Robot module.

        Returns:
            Mapping from local obs name to ObsTerm.
        """
        ...

    @abstractmethod
    def scatter_obs_terms(
        self, group: str, robot: "RobotModuleCfg"
    ) -> dict[str, tuple[int | None, "ManagerTermBaseCfg"]]:
        """Return task-side contributions to cross-group scatter observations.

        Same structure as :meth:`~...robots._base.RobotModuleCfg.scatter_obs_terms`:
        maps slot names to ``(output_dim, inner_TermCfg)``.  Common slots
        include ``"commands"`` (target pose) and ``"ee_pos_error"`` (tracking error).

        Return an empty dict for tasks that need no additional scatter terms.

        Args:
            group: Clone-group name.
            robot: Robot module.

        Returns:
            Mapping from slot name to (output_dim, inner TermCfg).
        """
        ...

    @abstractmethod
    def reward_terms(self, group: str, robot: "RobotModuleCfg") -> dict[str, "RewardTermCfg"]:
        """Return task-specific reward terms.

        Keys must be globally unique; the convention ``f"{group}_{name}"``
        avoids collisions across registrations.

        Args:
            group: Clone-group name.
            robot: Robot module.

        Returns:
            Mapping from reward term name to RewTerm.
        """
        ...

    @abstractmethod
    def termination_terms(self, group: str, robot: "RobotModuleCfg") -> dict[str, "TerminationTermCfg"]:
        """Return task-specific termination conditions.

        The global ``time_out`` term is always added by the registry.

        Args:
            group: Clone-group name.
            robot: Robot module.

        Returns:
            Mapping from termination name to DoneTerm.
        """
        ...

    @abstractmethod
    def reset_events(self, group: str, robot: "RobotModuleCfg") -> dict[str, "EventTermCfg"]:
        """Return task-specific reset event terms (e.g. object randomisation).

        Args:
            group: Clone-group name.
            robot: Robot module.

        Returns:
            Mapping from event name to EventTermCfg.
        """
        ...
