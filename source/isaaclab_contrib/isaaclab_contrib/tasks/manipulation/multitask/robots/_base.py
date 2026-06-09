# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for robot modules used in multitask environments.

A :class:`RobotModuleCfg` encapsulates everything a single robot contributes
to a multitask scene: its USD asset, EE-frame sensor, action columns,
robot-specific observations, and reset events.  All builder methods receive
a ``group`` string (the clone-group name, e.g. ``"franka_lift"``) so that
:class:`~isaaclab.managers.SceneEntityCfg` ``groups`` fields are set
correctly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.managers import ActionTermCfg, EventTermCfg, ManagerTermBaseCfg


@dataclass
class RobotModuleCfg(ABC):
    """Abstract dataclass defining a robot's contribution to a multitask environment.

    Subclasses implement all abstract methods.  Instance attributes
    (e.g. :attr:`ik_scale`) allow users to customise robot behaviour
    without subclassing again.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical robot identifier.

        Used to derive default scene asset names (``f"{name}_robot"``,
        ``f"{name}_ee_frame"``) and event/obs term name prefixes.
        Must be a valid Python identifier and unique within a registry.
        """
        ...

    @abstractmethod
    def scene_assets(self, group: str) -> dict[str, object]:
        """Return scene asset configs keyed by their scene attribute names.

        Keys follow the convention ``f"{self.name}_{asset_suffix}"``
        (e.g. ``"franka_robot"``, ``"franka_ee_frame"``).  These names
        must match the ``asset_name`` fields used in action / obs / event terms.

        Args:
            group: Clone-group name this robot belongs to.

        Returns:
            Mapping from scene attribute name to asset config.
        """
        ...

    @abstractmethod
    def action_specs(self, group: str | None = None) -> dict[str, tuple[int, "ActionTermCfg"]]:
        """Return action term configs grouped by shared action column.

        Each entry maps an *action column key* to ``(dim, ActionTermCfg)``.
        Robots that share the same column key are bundled into a single
        :class:`~...mdp.actions_cfg.ScatteredActionTermCfg` by the
        :class:`~...registry.MultiTaskRegistry`.

        Example::

            {
                "arm":     (6, DifferentialInverseKinematicsActionCfg(...)),
                "gripper": (1, BinaryJointPositionActionCfg(...)),
            }

        Args:
            group: Clone-group name (passed to ``asset_name`` / ``groups``).

        Returns:
            Mapping from column key to (dim, action cfg).
        """
        ...

    @property
    @abstractmethod
    def all_joint_names(self) -> list[str]:
        """Regex patterns covering every joint for the joint-velocity penalty.

        Typically includes both arm and finger joints, e.g.
        ``["panda_joint.*", "panda_finger.*"]``.
        """
        ...

    @abstractmethod
    def scatter_obs_terms(self, group: str) -> dict[str, tuple[int | None, "ManagerTermBaseCfg"]]:
        """Return robot-side contributions to cross-group scatter observations.

        Each entry maps an observation *slot name* to
        ``(output_dim, inner_TermCfg)``.  The registry collects matching
        slot names from all robots (and tasks) and wraps them in a single
        :class:`~...mdp.utils.scatter_term`.

        Use ``output_dim=None`` only if the slot already has the dim
        declared via ``@scatterable(output_dim=D)`` on the inner function.

        Example::

            {
                "ee_pose": (7, TermCfg(func=mdp.ee_pose, params={...})),
            }

        Args:
            group: Clone-group name.

        Returns:
            Mapping from slot name to (output_dim, inner TermCfg).
        """
        ...

    @abstractmethod
    def reset_events(self, group: str) -> dict[str, "EventTermCfg"]:
        """Return robot-specific reset event terms.

        Keys are globally unique event names (e.g. ``f"{self.name}_reset_joints"``).

        Args:
            group: Clone-group name.

        Returns:
            Mapping from event name to EventTermCfg.
        """
        ...
