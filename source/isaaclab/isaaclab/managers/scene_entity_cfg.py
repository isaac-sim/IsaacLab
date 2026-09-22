# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene


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
        This attribute is only used when :attr:`joint_names`, :attr:`body_names`, or :attr:`object_collection_names`
        are specified.

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
            ValueError: If both ``object_collection_names`` and ``object_collection_ids`` are specified and
                are not consistent.
        """
        if self.name not in scene.keys():
            raise ValueError(f"The scene entity '{self.name}' does not exist. Available entities: {scene.keys()}.")
        entity = scene[self.name]

        if self.joint_names is not None or self.joint_ids != slice(None):
            self.joint_names, self.joint_ids = self._resolve_names_and_ids(
                "joint", self.joint_names, self.joint_ids, entity.find_joints, entity.joint_names, entity.num_joints
            )
        if self.fixed_tendon_names is not None or self.fixed_tendon_ids != slice(None):
            self.fixed_tendon_names, self.fixed_tendon_ids = self._resolve_names_and_ids(
                "fixed_tendon",
                self.fixed_tendon_names,
                self.fixed_tendon_ids,
                entity.find_fixed_tendons,
                entity.fixed_tendon_names,
                entity.num_fixed_tendons,
            )
        if self.body_names is not None or self.body_ids != slice(None):
            # contact sensors expose their bodies through find_sensors/num_sensors
            is_sensor = hasattr(entity, "find_sensors")
            self.body_names, self.body_ids = self._resolve_names_and_ids(
                "body",
                self.body_names,
                self.body_ids,
                entity.find_sensors if is_sensor else entity.find_bodies,
                entity.body_names,
                entity.num_sensors if is_sensor else entity.num_bodies,
            )
        if self.object_collection_names is not None or self.object_collection_ids != slice(None):
            self.object_collection_names, self.object_collection_ids = self._resolve_names_and_ids(
                "object_collection",
                self.object_collection_names,
                self.object_collection_ids,
                entity.find_objects,
                entity.object_names,
                None,
            )

    def _resolve_names_and_ids(
        self,
        label: str,
        names: str | list[str] | None,
        ids: int | list[int] | slice,
        find_fn: Callable[..., tuple[list[int], list[str]]],
        entity_names: list[str],
        num_entities: int | None,
    ) -> tuple[list[str], list[int] | slice]:
        """Resolve the names and indices of one entity attribute (joints, bodies, ...) against each other.

        Args:
            label: The attribute label used in error messages (e.g. ``"joint"``).
            names: The configured names or regular expressions, if any.
            ids: The configured indices, or ``slice(None)`` when unspecified.
            find_fn: The entity method that resolves names to ``(indices, names)``.
            entity_names: All names of the attribute in the entity's order.
            num_entities: Number of entries in the entity. When given and all entries are selected in the
                entity's order, the indices collapse to ``slice(None)`` since slices index faster than lists.

        Returns:
            The resolved names and indices.

        Raises:
            ValueError: If both names and indices are specified and are not consistent.
        """
        if isinstance(names, str):
            names = [names]
        if isinstance(ids, int):
            ids = [ids]
        if names is not None and ids != slice(None):
            # both are specified: make sure they agree
            found_ids, _ = find_fn(names, preserve_order=self.preserve_order)
            found_names = [entity_names[i] for i in ids]
            if found_ids != ids or found_names != names:
                prose = label.replace("_", " ")
                raise ValueError(
                    f"Both '{label}_names' and '{label}_ids' are specified, and are not consistent."
                    f"\n\tfrom {prose} names: {names} [{found_ids}]"
                    f"\n\tfrom {prose} ids: {found_names} [{ids}]"
                    f"\nHint: Use either '{label}_names' or '{label}_ids' to avoid confusion."
                )
        elif names is not None:
            ids, _ = find_fn(names, preserve_order=self.preserve_order)
            if num_entities is not None and len(ids) == num_entities and names == entity_names:
                ids = slice(None)
        else:
            names = [entity_names[i] for i in ids]
        return names, ids
