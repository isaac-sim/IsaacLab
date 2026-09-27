# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING, field
from typing import TYPE_CHECKING

import torch

from ..utils import configclass

if TYPE_CHECKING:
    from ..assets import Articulation, RigidObject, RigidObjectCollection
    from ..scene import InteractiveScene


def _host_indices(indices: list[int] | torch.Tensor | slice) -> list[int] | slice:
    """Read device selections only during resolution or configuration serialization."""
    if isinstance(indices, torch.Tensor):
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Scene selectors must be one-dimensional integer tensors.")
        return indices.tolist()
    return indices


@configclass
class SceneEntityCfg:
    """Configuration for a scene entity that is used by the manager's term.

    This class is used to specify the name of the scene entity that is queried from the
    :class:`InteractiveScene` and passed to the manager's term function.

    After :meth:`resolve`, non-slice selectors are one-dimensional ``torch.long`` tensors
    on the scene device. Slices stay slices. No host copy is retained. Use ``ids.tolist()``
    only when Python values are required; on CUDA this copies and synchronizes. Configuration
    serialization performs this conversion outside the step loop. To select one body without
    reading an index back, use ``data[:, cfg.body_ids[:1]].squeeze(1)``.
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
    function as a device tensor of joint indices under :attr:`joint_ids`.
    """

    joint_ids: list[int] | torch.Tensor | slice = field(default=slice(None), metadata={"serializer": _host_indices})
    """The indices of the joints from the asset required by the term. Defaults to slice(None), which means
    all the joints in the asset (if present).

    If :attr:`joint_names` is specified, this is filled in automatically on initialization of the
    manager.
    """

    fixed_tendon_names: str | list[str] | None = None
    """The names of the fixed tendons from the scene entity. Defaults to None.

    The names can be either joint names or a regular expression matching the joint names.

    These are converted to fixed tendon indices on initialization of the manager and passed to the term
    function as a device tensor of fixed tendon indices under :attr:`fixed_tendon_ids`.
    """

    fixed_tendon_ids: list[int] | torch.Tensor | slice = field(
        default=slice(None), metadata={"serializer": _host_indices}
    )
    """The indices of the fixed tendons from the asset required by the term. Defaults to slice(None), which means
    all the fixed tendons in the asset (if present).

    If :attr:`fixed_tendon_names` is specified, this is filled in automatically on initialization of the
    manager.
    """

    body_names: str | list[str] | None = None
    """The names of the bodies from the asset required by the term. Defaults to None.

    The names can be either body names or a regular expression matching the body names.

    These are converted to body indices on initialization of the manager and passed to the term
    function as a device tensor of body indices under :attr:`body_ids`.
    """

    body_ids: list[int] | torch.Tensor | slice = field(default=slice(None), metadata={"serializer": _host_indices})
    """The indices of the bodies from the asset required by the term. Defaults to slice(None), which means
    all the bodies in the asset.

    If :attr:`body_names` is specified, this is filled in automatically on initialization of the
    manager.
    """

    object_collection_names: str | list[str] | None = None
    """The names of the objects in the rigid object collection required by the term. Defaults to None.

    The names can be either names or a regular expression matching the object names in the collection.

    These are converted to object indices on initialization of the manager and passed to the term
    function as a device tensor of object indices under :attr:`object_collection_ids`.
    """

    object_collection_ids: list[int] | torch.Tensor | slice = field(
        default=slice(None), metadata={"serializer": _host_indices}
    )
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

    def resolve(self, scene: InteractiveScene) -> None:
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

        # Re-resolution validates names on the host before uploading the final selections.
        self.joint_ids = _host_indices(self.joint_ids)
        self.fixed_tendon_ids = _host_indices(self.fixed_tendon_ids)
        self.body_ids = _host_indices(self.body_ids)
        self.object_collection_ids = _host_indices(self.object_collection_ids)

        # convert joint names to indices based on regex
        self._resolve_joint_names(scene)

        # convert fixed tendon names to indices based on regex
        self._resolve_fixed_tendon_names(scene)

        # convert body names to indices based on regex
        self._resolve_body_names(scene)

        # convert object collection names to indices based on regex
        self._resolve_object_collection_names(scene)

        self.joint_ids = _device_indices(self.joint_ids, scene.device)
        self.fixed_tendon_ids = _device_indices(self.fixed_tendon_ids, scene.device)
        self.body_ids = _device_indices(self.body_ids, scene.device)
        self.object_collection_ids = _device_indices(self.object_collection_ids, scene.device)

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
                if self.joint_names == _selected_names(entity.joint_names, self.joint_ids):
                    return
                joint_ids, _ = entity.find_joints(self.joint_names, preserve_order=self.preserve_order)
                joint_names = _selected_names(entity.joint_names, self.joint_ids)
                if joint_ids != _selected_ids(self.joint_ids, len(entity.joint_names)):
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
                self.joint_names = _selected_names(entity.joint_names, self.joint_ids)

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
                if self.fixed_tendon_names == _selected_names(entity.fixed_tendon_names, self.fixed_tendon_ids):
                    return
                fixed_tendon_ids, _ = entity.find_fixed_tendons(
                    self.fixed_tendon_names, preserve_order=self.preserve_order
                )
                fixed_tendon_names = _selected_names(entity.fixed_tendon_names, self.fixed_tendon_ids)
                if fixed_tendon_ids != _selected_ids(self.fixed_tendon_ids, len(entity.fixed_tendon_names)):
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
                self.fixed_tendon_names = _selected_names(entity.fixed_tendon_names, self.fixed_tendon_ids)

    def _resolve_body_names(self, scene: InteractiveScene):
        # convert body names to indices based on regex
        if self.body_names is not None or self.body_ids != slice(None):
            entity: RigidObject = scene[self.name]
            # -- if both are not their default values, check if they are valid
            # use find_sensors/num_sensors for ContactSensor, find_bodies/num_bodies for others
            _find_fn = entity.find_sensors if hasattr(entity, "find_sensors") else entity.find_bodies
            _num_bodies = entity.num_sensors if hasattr(entity, "num_sensors") else entity.num_bodies
            if self.body_names is not None and self.body_ids != slice(None):
                if isinstance(self.body_names, str):
                    self.body_names = [self.body_names]
                if isinstance(self.body_ids, int):
                    self.body_ids = [self.body_ids]
                if self.body_names == _selected_names(entity.body_names, self.body_ids):
                    return
                body_ids, _ = _find_fn(self.body_names, preserve_order=self.preserve_order)
                body_names = _selected_names(entity.body_names, self.body_ids)
                if body_ids != _selected_ids(self.body_ids, len(entity.body_names)):
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
                self.body_ids, _ = _find_fn(self.body_names, preserve_order=self.preserve_order)
                # performance optimization (slice offers faster indexing than list of indices)
                # only all bodies in the entity order are selected
                if len(self.body_ids) == _num_bodies and self.body_names == entity.body_names:
                    self.body_ids = slice(None)
            # -- from body indices to body names
            elif self.body_ids != slice(None):
                if isinstance(self.body_ids, int):
                    self.body_ids = [self.body_ids]
                self.body_names = _selected_names(entity.body_names, self.body_ids)

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
                if self.object_collection_names == _selected_names(entity.object_names, self.object_collection_ids):
                    return
                object_ids, _ = entity.find_objects(self.object_collection_names, preserve_order=self.preserve_order)
                object_names = _selected_names(entity.object_names, self.object_collection_ids)
                if object_ids != _selected_ids(self.object_collection_ids, len(entity.object_names)):
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
                self.object_collection_names = _selected_names(entity.object_names, self.object_collection_ids)


def _selected_ids(indices: Sequence[int] | slice, count: int) -> list[int]:
    """Compare resolved selections on the host, including repeated resolution."""
    if isinstance(indices, slice):
        return list(range(count)[indices])
    return [index + count if index < 0 else index for index in indices]


def _selected_names(names: list[str], indices: Sequence[int] | slice) -> list[str]:
    return names[indices] if isinstance(indices, slice) else [names[index] for index in indices]


def _device_indices(indices: Sequence[int] | slice, device: str) -> torch.Tensor | slice:
    if isinstance(indices, slice):
        return indices
    return torch.tensor(indices, dtype=torch.long, device=device)
