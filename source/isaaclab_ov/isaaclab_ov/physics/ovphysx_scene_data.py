# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Publish OvPhysX stage data through Isaac Lab's scene-data interface."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from isaaclab.scene_data import SceneDataBackend, SceneDataFormat
from isaaclab.scene_data.deformable_discovery import deformable_geometry_batches
from isaaclab.utils.buffers import TimestampedBuffer

from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView

if TYPE_CHECKING:
    from isaaclab.scene_data.deformable_discovery import DeformableStageEntry

    from .ovphysx_backend import OvPhysxBackend

__all__ = ["OvPhysxSceneDataBackend"]


class OvPhysxSceneDataBackend(SceneDataBackend):
    """Scene-data backend for the OVPhysX physics manager.

    The runtime's rigid-body view fills one pose buffer directly. Deformable views
    fill their native slices without an intermediate packing pass.
    """

    def __init__(self, manager: object | None = None):
        """Create an unbound publication; ``manager`` is deprecated and ignored."""
        if manager is not None:
            warnings.warn(
                "Passing a manager to OvPhysxSceneDataBackend is deprecated and ignored. Use"
                " OvPhysxSceneDataBackend() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.backend: OvPhysxBackend | None = None
        """Native resource borrowed from the context registry after warmup."""
        self._transforms = TimestampedBuffer(SceneDataFormat.Transform())
        self.transforms_timestamp = 0
        self.geometry_timestamp = 0
        self._geometry = TimestampedBuffer()
        self._deformable_bindings: list[tuple[OvPhysxView, Any, wp.array]] = []
        self._pending_setup: tuple[OvPhysxBackend, str, Sequence[DeformableStageEntry] | None] | None = None

    def _defer_setup(
        self, backend: OvPhysxBackend, device: str, entries: Sequence[DeformableStageEntry] | None = None
    ) -> None:
        """Defer renderer-only bindings until scene data is requested."""
        self._clear()
        self._pending_setup = (backend, device, entries)

    def invalidate(self) -> None:
        """Mark published transforms and geometry stale after native state changes."""
        self.transforms_timestamp += 1
        self.geometry_timestamp += 1

    def _clear(self) -> None:
        """Discard publications and deferred reads before their native handles are released."""
        self.backend = None
        self._transforms.data.transforms = None
        self._deformable_bindings = []
        self._geometry = TimestampedBuffer()
        self.invalidate()
        self._pending_setup = None

    def _ensure_setup(self) -> None:
        if self._pending_setup is not None:
            backend, device, entries = self._pending_setup
            if not backend.is_ready:
                raise RuntimeError("Warm up or step the OVPhysX backend before requesting scene data.")
            # Native output metadata names the actual bodies, not articulation-root aliases.
            # Keep these renderer-only bindings out of headless simulation startup.
            from ovphysx.types import SimObjectType

            kinds = (SimObjectType.RIGID_BODY, SimObjectType.ARTICULATION_LINK)
            body_paths = []
            for kind in kinds:
                with backend.physx.read_sync(kind, ["mass"]) as bodies:
                    for group in bodies.groups:
                        body_paths.extend(bodies.dictionary.get_path_strings(group.prim_list))
            if body_paths:
                backend.rigid_body_view = backend.physx.create_tensor_binding(prim_paths=body_paths)
            self.setup(backend, device, entries)
            self._pending_setup = None

    @property
    def transform_count(self) -> int:
        """Number of poses in the native publication."""
        self._ensure_setup()
        poses = self._transforms.data.transforms
        return 0 if poses is None else len(poses)

    @property
    def transform_paths(self) -> list[str]:
        """Native body paths in the same order as the published poses."""
        self._ensure_setup()
        view = None if self.backend is None else self.backend.rigid_body_view
        return [] if view is None else view.prim_paths

    def setup(
        self, backend: OvPhysxBackend, device: str, entries: Sequence[DeformableStageEntry] | None = None
    ) -> None:
        """Bind the native body table and declared deformable geometry.

        Args:
            backend: Initialized native physics resource.
            device: Warp device string used to allocate the published buffers.
            entries: Declared deformables captured before native stage import. ``None`` leaves
                scene geometry uninitialized; an empty sequence declares no deformables.
        """
        self.backend = backend
        self._transforms.data.transforms = None
        self._deformable_bindings = []
        self._geometry = TimestampedBuffer()
        self.invalidate()

        if backend.rigid_body_view is not None and backend.rigid_body_view.count:
            self._transforms.data.transforms = wp.empty(
                backend.rigid_body_view.count, dtype=wp.transformf, device=device
            )

        if entries is not None:
            self._setup_deformable_bindings(backend.physx, entries, device)

    def _setup_deformable_bindings(self, physx, entries: Sequence[DeformableStageEntry], device: str) -> None:
        """Bind exact planned deformables directly into one flat publication buffer."""
        from isaaclab_ov import tensor_types as TT

        groups = {}
        for entry in entries:
            groups.setdefault((entry.deformable_type, entry.vertex_count), []).append(entry)

        views, native_entries = [], []
        for (deformable_type, count), entries in groups.items():
            tensor_type = (
                TT.DEFORMABLE_SIM_NODAL_POSITION if deformable_type == "volume" else TT.SURFACE_DEFORMABLE_SIM_POSITION
            )
            view = OvPhysxView(
                physx,
                prim_paths=[entry.root_path for entry in entries],
                device=device,
                tensor_types=[tensor_type],
                eager=True,
            )
            by_path = {path: entry for entry in entries for path in (entry.root_path, entry.sim_mesh_path)}
            ordered = [by_path[path] for path in view.prim_paths]
            if sorted(entry.root_path for entry in ordered) != sorted(entry.root_path for entry in entries):
                raise RuntimeError("OVPhysX deformable binding does not cover the declared clone-plan entries.")
            if tuple(view.binding_for(tensor_type).shape) != (len(entries), count, 3):
                raise RuntimeError("OVPhysX deformable node counts disagree with the clone plan.")
            native_entries.extend(ordered)
            views.append((view, tensor_type, count))

        if not views:
            self._geometry.data = []
            return
        counts = [entry.vertex_count for entry in native_entries]
        points = wp.empty(sum(counts), dtype=wp.vec3f, device=device)
        offset = 0
        for view, tensor_type, count in views:
            buffer = wp.array(
                ptr=points.ptr + offset * wp.types.type_size_in_bytes(wp.vec3f),
                shape=(view.count, count),
                dtype=wp.vec3f,
                device=device,
                copy=False,
            )
            self._deformable_bindings.append((view, tensor_type, buffer))
            offset += view.count * count
        offsets = np.cumsum(np.r_[0, counts[:-1]])
        self._geometry.data = deformable_geometry_batches(native_entries, offsets, device=device)
        for publication, _ in self._geometry.data:
            publication.points = points

    def get_geometry_batches(self, output_format: Any = SceneDataFormat.Points) -> list:
        """Publish native positions with exact visual paths and interpolation metadata.

        Raises:
            RuntimeError: If scene geometry was not initialized from a clone plan.
        """
        self._ensure_setup()
        if self._geometry.data is None:
            raise RuntimeError("Declare and replicate a ClonePlan before requesting scene geometry.")
        if self._geometry.timestamp != self.geometry_timestamp:
            for view, tensor_type, buffer in self._deformable_bindings:
                view.read_into(tensor_type, buffer)
            self._geometry.timestamp = self.geometry_timestamp
        return self._geometry.data

    @property
    def native_geometry_formats(self) -> tuple[Any, ...]:
        """Return the native geometry formats compiled from the declared prototypes.

        Raises:
            RuntimeError: If scene geometry was not initialized from a clone plan.
        """
        self._ensure_setup()
        if self._geometry.data is None:
            raise RuntimeError("Declare and replicate a ClonePlan before requesting scene geometry.")
        return tuple(dict.fromkeys(publication._cls for publication, _ in self._geometry.data))

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        """Publish native rigid-body poses [m, xyzw]."""
        self._ensure_setup()
        if self._transforms.timestamp != self.transforms_timestamp:
            if self.backend is not None:
                self.backend.update_kinematics()
            if self._transforms.data.transforms is not None:
                self.backend.rigid_body_view.read(self._transforms.data.transforms)
            self._transforms.timestamp = self.transforms_timestamp
        return self._transforms.data
