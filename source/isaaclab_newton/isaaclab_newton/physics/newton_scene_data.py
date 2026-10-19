# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish Newton state through Isaac Lab's scene-data interface."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import warp as wp
from newton import Model, State

from isaaclab.scene_data import SceneDataBackend, SceneDataFormat

from . import newton_backend as nb
from .newton_backend import NewtonBackend


class NewtonSceneDataBackend(SceneDataBackend):
    """Scene data backend that reads rigid body transforms from Newton's simulation state.

    The backend reads ``body_q`` (an array of :class:`wp.transformf`) from Newton's current state and exposes it as
    :class:`SceneDataFormat.Transform`. Body paths come from the model's ``body_label`` attribute.
    """

    def __init__(self, backend: Callable[[], NewtonBackend | None]):
        """Initialize the scene data backend.

        Args:
            backend: Returns the Newton backend whose state to publish; it changes on every hard reset.
        """
        self._backend = backend
        self._transforms = SceneDataFormat.Transform()
        self.transforms_timestamp = 0
        self.geometry_timestamp = 0
        self._geometry_batches = []

    def initialize_geometry(self, geometry_batches: list, cable_bindings: dict[str, list[int]]) -> None:
        """Bind imported geometry paths to native particle ranges and capsule endpoints.

        Args:
            geometry_batches: Deformable and particle geometry recorded by replication.
            cable_bindings: Native capsule shape indices of each open cable.
        """
        self._geometry_batches = list(geometry_batches)
        endpoints, ranges = [], {}
        offset = 0
        for path, shapes in cable_bindings.items():
            ids = np.asarray(shapes, dtype=np.int32)
            left = np.concatenate((ids[:1], ids))
            right = np.concatenate((ids, ids[-1:]))
            left_sign = np.ones(len(left), dtype=np.int32)
            right_sign = -np.ones(len(right), dtype=np.int32)
            left_sign[0], right_sign[-1] = -1, 1
            endpoints.append(np.column_stack((left, left_sign, right, right_sign)))
            ranges[path] = (offset, len(left))
            offset += len(left)
        if endpoints:
            model = self.model
            source = SceneDataFormat.CapsuleEndpoints()
            source.transforms = self._backend().state_0.body_q
            source.shape_body = model.shape_body
            source.shape_transform = model.shape_transform
            source.shape_scale = model.shape_scale
            source.endpoints = wp.array(np.concatenate(endpoints), dtype=wp.vec4i, device=model.device)
            self._geometry_batches.append((source, ranges))

    @property
    def native_geometry_formats(self) -> tuple[type, ...]:
        return (SceneDataFormat.Points, SceneDataFormat.WeightedPoints, SceneDataFormat.CapsuleEndpoints)

    def get_geometry_batches(self, output_format=SceneDataFormat.Points):
        """Publish native arrays; SDP derives cable endpoints and applies destination layouts."""
        state = self.state
        for source, _ in self._geometry_batches:
            attribute = "transforms" if source._cls is SceneDataFormat.CapsuleEndpoints else "points"
            data = state.particle_q if attribute == "points" else state.body_q
            if getattr(source, attribute) is not data:
                setattr(source, attribute, data)
                self.geometry_timestamp += 1
        return self._geometry_batches

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        """Publish the authoritative native pointer, including solver state-buffer swaps."""
        transforms = self.state.body_q
        if self._transforms.transforms is not transforms:
            self._transforms.transforms = transforms
            self.transforms_timestamp += 1
        return self._transforms

    @property
    def transform_count(self) -> int:
        """Return the number of rigid body transforms in the Newton sim."""
        return self.model.body_count

    @property
    def transform_paths(self) -> list[str]:
        """Return the prim paths for each rigid body transform."""
        if self.model.body_label is not None:
            return list(self.model.body_label)
        return []

    @property
    def model(self) -> Model | None:
        backend = self._backend()
        return None if backend is None else backend.model

    @property
    def state(self) -> State | None:
        """Return native physics state, consistent with authored state, without entering the rendering path."""
        backend = self._backend()
        if backend is None:
            return None
        if backend.transforms_may_change_on_graph_replay:
            # Raw external graph replays bypass Python invalidation, so these reads must stay conservative.
            self.transforms_timestamp += 1
            self.geometry_timestamp += 1
        if backend.solver is not None:
            nb.notify_model_changes(backend)
            nb.forward(backend)
        return backend.state_0
