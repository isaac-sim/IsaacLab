# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.scene import InteractiveScene
from omni.isaac.orbit.utils import configclass


@configclass
class SceneEntityCfg:
    """Configuration for a scene entity that is used by the manager's term.

    This class is used to specify the name of the scene entity that is queried from the
    :class:`InteractiveScene` and passed to the manager's term function.
    """

    name: str = MISSING
    """The name of the scene entity.

    This is the name defined in the scene configuration file. See the :class:`InteractiveSceneCfg`
    class for more details.
    """

    joint_names: str | list[str] | None = None
    """The names of the joints from the scene entity. Defaults to None.

    The names can be either joint names or a regular expression matching the joint names.

    These are converted to joint indices on initialization of the manager and passed to the term
    function as a list of joint indices under :attr:`joint_ids`.
    """

    joint_ids: list[int] | slice = slice(None)
    """The indices of the joints from the asset required by the term. Defaults to slice(None), which means
    all the joints in the asset (if present).

    If :attr:`joint_names` is specified, this is filled in automatically on initialization of the
    manager.
    """

    body_names: str | list[str] | None = None
    """The names of the bodies from the asset required by the term. Defaults to None.

    The names can be either body names or a regular expression matching the body names.

    These are converted to body indices on initialization of the manager and passed to the term
    function as a list of body indices under :attr:`body_ids`.
    """

    body_ids: list[int] | slice = slice(None)
    """The indices of the bodies from the asset required by the term. Defaults to slice(None), which means
    all the bodies in the asset.

    If :attr:`body_names` is specified, this is filled in automatically on initialization of the
    manager.
    """

    def resolve(self, scene: InteractiveScene):
        """Resolves the scene entity and converts the joint and body names to indices.

        This function examines the scene entity from the :class:`InteractiveScene` and resolves the indices
        and names of the joints and bodies. It is an expensive operation as it resolves regular expressions
        and should be called only once.

        Args:
            scene: The interactive scene instance.

        Raises:
            ValueError: If the scene entity is not found.
            ValueError: If both ``joint_names`` and ``joint_ids`` are specified and are not consistent.
            ValueError: If both ``body_names`` and ``body_ids`` are specified and are not consistent.
        """
        # check if the entity is valid
        if self.name not in scene.keys():
            raise ValueError(f"The scene entity '{self.name}' does not exist. Available entities: {scene.keys()}.")

        # convert joint names to indices based on regex
        if self.joint_names is not None or self.joint_ids != slice(None):
            entity: Articulation = scene[self.name]
            # -- if both are not their default values, check if they are valid
            if self.joint_names is not None and self.joint_ids != slice(None):
                if isinstance(self.joint_names, str):
                    self.joint_names = [self.joint_names]
                if isinstance(self.joint_ids, int):
                    self.joint_ids = [self.joint_ids]
                joint_ids, _ = entity.find_joints(self.joint_names)
                joint_names = [entity.joint_names[i] for i in self.joint_ids]
                if joint_ids != self.joint_ids or joint_names != self.joint_names:
                    raise ValueError(
                        "Both 'joint_names' and 'joint_ids' are specified, and are not consistent."
                        f"\n\tfrom joint names: {self.joint_names} [{joint_ids}]"
                        f"\n\tfrom joint ids: {joint_names} [{self.joint_ids}]"
                        "\nHint: Use either 'joint_names' or 'joint_ids' to avoid confusion."
                    )
            # -- from joint names to joint indices
            elif self.joint_names is not None:
                if isinstance(self.joint_names, str):
                    self.joint_names = [self.joint_names]
                self.joint_ids, _ = entity.find_joints(self.joint_names)
                # performance optimization (slice offers faster indexing than list of indices)
                if len(self.joint_ids) == entity.num_joints:
                    self.joint_ids = slice(None)
            # -- from joint indices to joint names
            elif self.joint_ids != slice(None):
                if isinstance(self.joint_ids, int):
                    self.joint_ids = [self.joint_ids]
                self.joint_names = [entity.joint_names[i] for i in self.joint_ids]

        # convert body names to indices based on regex
        if self.body_names is not None or self.body_ids != slice(None):
            entity: RigidObject = scene[self.name]
            # -- if both are not their default values, check if they are valid
            if self.body_names is not None and self.body_ids != slice(None):
                if isinstance(self.body_names, str):
                    self.body_names = [self.body_names]
                if isinstance(self.body_ids, int):
                    self.body_ids = [self.body_ids]
                body_ids, _ = entity.find_bodies(self.body_names)
                body_names = [entity.body_names[i] for i in self.body_ids]
                if body_ids != self.body_ids or body_names != self.body_names:
                    raise ValueError(
                        "Both 'body_names' and 'body_ids' are specified, and are not consistent."
                        f"\n\tfrom body names: {self.body_names} [{body_ids}]"
                        f"\n\tfrom body ids: {body_names} [{self.body_ids}]"
                        "\nHint: Use either 'body_names' or 'body_ids' to avoid confusion."
                    )
            # -- from body names to body indices
            elif self.body_names is not None:
                if isinstance(self.body_names, str):
                    self.body_names = [self.body_names]
                self.body_ids, _ = entity.find_bodies(self.body_names)
                # performance optimization (slice offers faster indexing than list of indices)
                if len(self.body_ids) == entity.num_bodies:
                    self.body_ids = slice(None)
            # -- from body indices to body names
            elif self.body_ids != slice(None):
                if isinstance(self.body_ids, int):
                    self.body_ids = [self.body_ids]
                self.body_names = [entity.body_names[i] for i in self.body_ids]
