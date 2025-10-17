# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from dataclasses import MISSING

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.scene import InteractiveScene
from isaaclab.utils import configclass


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

    fixed_tendon_names: str | list[str] | None = None
    """The names of the fixed tendons from the scene entity. Defaults to None.

    The names can be either joint names or a regular expression matching the joint names.

    These are converted to fixed tendon indices on initialization of the manager and passed to the term
    function as a list of fixed tendon indices under :attr:`fixed_tendon_ids`.
    """

    fixed_tendon_ids: list[int] | slice = slice(None)
    """The indices of the fixed tendons from the asset required by the term. Defaults to slice(None), which means
    all the fixed tendons in the asset (if present).

    If :attr:`fixed_tendon_names` is specified, this is filled in automatically on initialization of the
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

    object_collection_names: str | list[str] | None = None
    """The names of the objects in the rigid object collection required by the term. Defaults to None.

    The names can be either names or a regular expression matching the object names in the collection.

    These are converted to object indices on initialization of the manager and passed to the term
    function as a list of object indices under :attr:`object_collection_ids`.
    """

    object_collection_ids: list[int] | slice = slice(None)
    """The indices of the objects from the rigid object collection required by the term. Defaults to slice(None),
    which means all the objects in the collection.

    If :attr:`object_collection_names` is specified, this is filled in automatically on initialization of the manager.
    """

    preserve_order: bool = False
    """Whether to preserve indices ordering to match with that in the specified joint, body, or object collection names.
    Defaults to False.

    If False, the ordering of the indices are sorted in ascending order (i.e. the ordering in the entity's joints,
    bodies, or object in the object collection). Otherwise, the indices are preserved in the order of the specified
    joint, body, or object collection names.

    For more details, see the :meth:`isaaclab.utils.string.resolve_matching_names` function.

    .. note::
        This attribute is only used when :attr:`joint_names`, :attr:`body_names`, or :attr:`object_collection_names` are specified.

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
            ValueError: If both ``fixed_tendon_names`` and ``fixed_tendon_ids`` are specified and are not consistent.
            ValueError: If both ``body_names`` and ``body_ids`` are specified and are not consistent.
            ValueError: If both ``object_collection_names`` and ``object_collection_ids`` are specified and are not consistent.
        """
        # check if the entity is valid
        if self.name not in scene.keys():
            raise ValueError(f"The scene entity '{self.name}' does not exist. Available entities: {scene.keys()}.")

        # convert joint names to indices based on regex
        self._resolve_joint_names(scene)

        # convert fixed tendon names to indices based on regex
        self._resolve_fixed_tendon_names(scene)

        # convert body names to indices based on regex
        self._resolve_body_names(scene)

        # convert object collection names to indices based on regex
        self._resolve_object_collection_names(scene)

    def _resolve_joint_names(self, scene: InteractiveScene):
        # convert joint names to indices based on regex
        if self.joint_names is not None or self.joint_ids != slice(None):
            entity: Articulation = scene[self.name]
            # -- if both are not their default values, check if they are valid
            if self.joint_names is not None and self.joint_ids != slice(None):
                if isinstance(self.joint_names, str):
                    self.joint_names = [self.joint_names]
                if isinstance(self.joint_ids, int):
                    self.joint_ids = [self.joint_ids]
                joint_ids, _ = entity.find_joints(self.joint_names, preserve_order=self.preserve_order)
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
                self.joint_ids, _ = entity.find_joints(self.joint_names, preserve_order=self.preserve_order)
                # performance optimization (slice offers faster indexing than list of indices)
                # only all joint in the entity order are selected
                if len(self.joint_ids) == entity.num_joints and self.joint_names == entity.joint_names:
                    self.joint_ids = slice(None)
            # -- from joint indices to joint names
            elif self.joint_ids != slice(None):
                if isinstance(self.joint_ids, int):
                    self.joint_ids = [self.joint_ids]
                self.joint_names = [entity.joint_names[i] for i in self.joint_ids]

    def _resolve_fixed_tendon_names(self, scene: InteractiveScene):
        # convert tendon names to indices based on regex
        if self.fixed_tendon_names is not None or self.fixed_tendon_ids != slice(None):
            entity: Articulation = scene[self.name]
            # -- if both are not their default values, check if they are valid
            if self.fixed_tendon_names is not None and self.fixed_tendon_ids != slice(None):
                if isinstance(self.fixed_tendon_names, str):
                    self.fixed_tendon_names = [self.fixed_tendon_names]
                if isinstance(self.fixed_tendon_ids, int):
                    self.fixed_tendon_ids = [self.fixed_tendon_ids]
                fixed_tendon_ids, _ = entity.find_fixed_tendons(
                    self.fixed_tendon_names, preserve_order=self.preserve_order
                )
                fixed_tendon_names = [entity.fixed_tendon_names[i] for i in self.fixed_tendon_ids]
                if fixed_tendon_ids != self.fixed_tendon_ids or fixed_tendon_names != self.fixed_tendon_names:
                    raise ValueError(
                        "Both 'fixed_tendon_names' and 'fixed_tendon_ids' are specified, and are not consistent."
                        f"\n\tfrom joint names: {self.fixed_tendon_names} [{fixed_tendon_ids}]"
                        f"\n\tfrom joint ids: {fixed_tendon_names} [{self.fixed_tendon_ids}]"
                        "\nHint: Use either 'fixed_tendon_names' or 'fixed_tendon_ids' to avoid confusion."
                    )
            # -- from fixed tendon names to fixed tendon indices
            elif self.fixed_tendon_names is not None:
                if isinstance(self.fixed_tendon_names, str):
                    self.fixed_tendon_names = [self.fixed_tendon_names]
                self.fixed_tendon_ids, _ = entity.find_fixed_tendons(
                    self.fixed_tendon_names, preserve_order=self.preserve_order
                )
                # performance optimization (slice offers faster indexing than list of indices)
                # only all fixed tendon in the entity order are selected
                if (
                    len(self.fixed_tendon_ids) == entity.num_fixed_tendons
                    and self.fixed_tendon_names == entity.fixed_tendon_names
                ):
                    self.fixed_tendon_ids = slice(None)
            # -- from fixed tendon indices to fixed tendon names
            elif self.fixed_tendon_ids != slice(None):
                if isinstance(self.fixed_tendon_ids, int):
                    self.fixed_tendon_ids = [self.fixed_tendon_ids]
                self.fixed_tendon_names = [entity.fixed_tendon_names[i] for i in self.fixed_tendon_ids]

    def _resolve_body_names(self, scene: InteractiveScene):
        # convert body names to indices based on regex
        if self.body_names is not None or self.body_ids != slice(None):
            entity: RigidObject = scene[self.name]
            # -- if both are not their default values, check if they are valid
            if self.body_names is not None and self.body_ids != slice(None):
                if isinstance(self.body_names, str):
                    self.body_names = [self.body_names]
                if isinstance(self.body_ids, int):
                    self.body_ids = [self.body_ids]
                body_ids, _ = entity.find_bodies(self.body_names, preserve_order=self.preserve_order)
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
                self.body_ids, _ = entity.find_bodies(self.body_names, preserve_order=self.preserve_order)
                # performance optimization (slice offers faster indexing than list of indices)
                # only all bodies in the entity order are selected
                if len(self.body_ids) == entity.num_bodies and self.body_names == entity.body_names:
                    self.body_ids = slice(None)
            # -- from body indices to body names
            elif self.body_ids != slice(None):
                if isinstance(self.body_ids, int):
                    self.body_ids = [self.body_ids]
                self.body_names = [entity.body_names[i] for i in self.body_ids]

    def _resolve_object_collection_names(self, scene: InteractiveScene):
        # convert object names to indices based on regex
        if self.object_collection_names is not None or self.object_collection_ids != slice(None):
            entity: RigidObjectCollection = scene[self.name]
            # -- if both are not their default values, check if they are valid
            if self.object_collection_names is not None and self.object_collection_ids != slice(None):
                if isinstance(self.object_collection_names, str):
                    self.object_collection_names = [self.object_collection_names]
                if isinstance(self.object_collection_ids, int):
                    self.object_collection_ids = [self.object_collection_ids]
                object_ids, _ = entity.find_objects(self.object_collection_names, preserve_order=self.preserve_order)
                object_names = [entity.object_names[i] for i in self.object_collection_ids]
                if object_ids != self.object_collection_ids or object_names != self.object_collection_names:
                    raise ValueError(
                        "Both 'object_collection_names' and 'object_collection_ids' are specified, and are not"
                        " consistent.\n\tfrom object collection names:"
                        f" {self.object_collection_names} [{object_ids}]\n\tfrom object collection ids:"
                        f" {object_names} [{self.object_collection_ids}]\nHint: Use either 'object_collection_names' or"
                        " 'object_collection_ids' to avoid confusion."
                    )
            # -- from object names to object indices
            elif self.object_collection_names is not None:
                if isinstance(self.object_collection_names, str):
                    self.object_collection_names = [self.object_collection_names]
                self.object_collection_ids, _ = entity.find_objects(
                    self.object_collection_names, preserve_order=self.preserve_order
                )
            # -- from object indices to object names
            elif self.object_collection_ids != slice(None):
                if isinstance(self.object_collection_ids, int):
                    self.object_collection_ids = [self.object_collection_ids]
                self.object_collection_names = [entity.object_names[i] for i in self.object_collection_ids]
