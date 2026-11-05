# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish Newton state through Isaac Lab's scene-data interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from newton import Model, State

from isaaclab.scene_data import SceneDataBackend, SceneDataFormat

if TYPE_CHECKING:
    from .newton_manager import NewtonManager


@wp.kernel(enable_backward=False)
def _publish_body_variant_transforms(
    body_q: wp.array(dtype=wp.transformf),
    body_variant: wp.array(dtype=wp.int32),
    variant_body: wp.array(dtype=wp.int32),
    variant_index: wp.array(dtype=wp.int32),
    transforms: wp.array(dtype=wp.transformf),
):
    """Copy the body poses, then pose each body variant like its body when selected and out of view otherwise."""
    i = wp.tid()
    if i < body_q.shape[0]:
        transforms[i] = body_q[i]
        return
    variant = i - body_q.shape[0]
    body = variant_body[variant]
    if body_variant[body] == variant_index[variant]:
        transforms[i] = body_q[body]
    else:
        transforms[i] = wp.transformf(wp.vec3f(0.0, 0.0, -1.0e4), wp.quatf(0.0, 0.0, 0.0, 1.0))


class NewtonSceneDataBackend(SceneDataBackend):
    """Scene data backend that reads rigid body transforms from Newton's simulation state.

    The backend reads ``body_q`` (an array of :class:`wp.transformf`) from Newton's current state and exposes it as
    :class:`SceneDataFormat.Transform`. Body paths come from the model's ``body_label`` attribute.
    """

    def __init__(self, manager: NewtonManager):
        """Initialize the scene data backend.

        Args:
            manager: Newton manager whose backend state to publish; the backend changes on every hard reset.
        """
        self._manager = manager
        self._transforms = SceneDataFormat.Transform()
        self._variant_model = None
        self._variant_labels: dict[int, list[str]] = {}
        self._variant_count = 0
        self._variant_transforms = None
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
            source.transforms = self._manager.backend.state_0.body_q
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
        if self._body_variants():
            transforms = self._publish_body_variants(transforms)
        if self._transforms.transforms is not transforms:
            self._transforms.transforms = transforms
            self.transforms_timestamp += 1
        return self._transforms

    @property
    def transform_count(self) -> int:
        """Return the number of rigid body transforms in the Newton sim, including body variants."""
        self._body_variants()
        return self.model.body_count + self._variant_count

    @property
    def transform_paths(self) -> list[str]:
        """Return the prim paths for each rigid body transform, then for each body variant."""
        model = self.model
        if model.body_label is None:
            return []
        paths = list(model.body_label)
        for body, labels in self._body_variants().items():
            paths.extend(f"{paths[body]}/{label}" for label in labels)
        return paths

    def close_model(self) -> None:
        """Drop the closed backend's model."""
        self._variant_model = None
        self._variant_labels = {}
        self._variant_count = 0
        self._variant_transforms = None

    def _body_variants(self) -> dict[int, list[str]]:
        """Return the variant labels of each body with shape variants, cached per model."""
        model = self.model
        if self._variant_model is not model:
            selected = getattr(model, "body_variant", None)
            bodies = [] if selected is None else np.flatnonzero(selected.numpy() >= 0).tolist()
            self._variant_labels = {body: model.body_variant_labels(body) for body in bodies}
            self._variant_count = sum(len(labels) for labels in self._variant_labels.values())
            self._variant_transforms = None
            self._variant_model = model
        return self._variant_labels

    def _publish_body_variants(self, body_q: wp.array) -> wp.array:
        """Return the body poses followed by the pose of each body variant's prim."""
        model = self.model
        if self._variant_transforms is None:
            counts = [len(labels) for labels in self._variant_labels.values()]
            variant_body = np.repeat(list(self._variant_labels), counts).astype(np.int32)
            variant_index = np.concatenate([np.arange(count, dtype=np.int32) for count in counts])
            self._variant_body = wp.array(variant_body, dtype=wp.int32, device=model.device)
            self._variant_index = wp.array(variant_index, dtype=wp.int32, device=model.device)
            self._variant_transforms = wp.empty(
                model.body_count + self._variant_count, dtype=wp.transformf, device=model.device
            )
        wp.launch(
            _publish_body_variant_transforms,
            dim=self._variant_transforms.shape[0],
            inputs=[body_q, model.body_variant, self._variant_body, self._variant_index],
            outputs=[self._variant_transforms],
            device=model.device,
        )
        return self._variant_transforms

    @property
    def model(self) -> Model | None:
        backend = self._manager.backend
        return None if backend is None else backend.model

    @property
    def state(self) -> State | None:
        """Return native physics state, consistent with authored state, without entering the rendering path."""
        manager = self._manager
        backend = manager.backend
        if backend is None:
            return None
        if backend.transforms_may_change_on_graph_replay:
            # Raw external graph replays bypass Python invalidation, so these reads must stay conservative.
            self.transforms_timestamp += 1
            self.geometry_timestamp += 1
        if backend.solver is not None:
            # Recording a graph must leave pending FK/reset work for the next eager read.
            if backend.device.is_capturing:
                backend.notify_model_changes()
            else:
                manager.forward()
        return backend.state_0
