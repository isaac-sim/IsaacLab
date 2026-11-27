# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publishers that keep a simulation-owned OVStage current with scene data provider state.

A publisher owns the queries and change tracking for one kind of producer data. The stage
holds one publisher per kind, shared by every consumer of the stage, and releases it before
its paths. Consumers call :meth:`~isaaclab_ov.stage.OvstageBackend.publish` before they
commit; a publisher writes only when its producer, or its own presentation state, changed.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import ovstage
import warp as wp

from pxr import Gf, Usd, UsdGeom

from isaaclab.scene_data import SceneDataFormat

from isaaclab_ov.renderers.ovrtx_renderer_kernels import HIDDEN_ENVIRONMENT_DROP, hide_environment_bodies_kernel
from isaaclab_ov.stage import xform_tensor_from_warp

if TYPE_CHECKING:
    from isaaclab.scene_data import SceneDataProvider

    from isaaclab_ov.stage import OvstageBackend


def body_transform_scales(
    stage: Usd.Stage, clone_copies: Sequence[tuple[str, Sequence[str]]], object_paths: Sequence[str], device: str
) -> wp.array | None:
    """Return the authored world scales of declared bodies in publication order.

    Physics poses omit scale. Native clone destinations do not exist on the authored USD
    stage, so they take the scale of their declared source prim.

    Args:
        stage: Authored USD stage holding the clone sources.
        clone_copies: Source roots paired with their destination roots.
        object_paths: Body prim paths in publication order.
        device: Warp device of the returned array.

    Returns:
        Scales of shape ``[len(object_paths)]``, or ``None`` when every body has unit scale,
        which lets consumers share the provider's publication instead of multiplying by ones.
    """
    scales_by_path = {}
    xform_cache = UsdGeom.XformCache()
    for root, targets in clone_copies:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not prim.IsA(UsdGeom.Xformable):
                continue
            scale = tuple(map(float, Gf.Transform(xform_cache.GetLocalToWorldTransform(prim)).GetScale()))
            if not all(math.isclose(axis, 1.0, rel_tol=1e-6, abs_tol=1e-6) for axis in scale):
                path = str(prim.GetPath())
                scales_by_path[path] = scale
                scales_by_path.update((target + path[len(root) :], scale) for target in targets)
    if not any(path in scales_by_path for path in object_paths):
        return None
    scales = [scales_by_path.get(path, (1.0, 1.0, 1.0)) for path in object_paths]
    return wp.array(scales, dtype=wp.vec3f, device=device)


def body_environment_ids(object_paths: Sequence[str], env_paths: Sequence[str]) -> list[int]:
    """Return each body's environment index, or ``len(env_paths)`` for a body outside any environment.

    Args:
        object_paths: Body prim paths in publication order.
        env_paths: Environment root paths in environment order.
    """
    roots = {root: i for i, root in enumerate(env_paths)}
    outside = len(env_paths)
    env_ids = []
    for path in object_paths:
        env_id, end = outside, path.rfind("/")
        while end > 0:  # Walk up the prim path until it reaches an environment root.
            env_id = roots.get(path[:end], outside)
            if env_id != outside:
                break
            end = path.rfind("/", 0, end)
        env_ids.append(env_id)
    return env_ids


class BodyPosePublisher:
    """Write physics body world poses to ``omni:xform`` once per producer timestamp.

    Visibility is stage state: every consumer of the stage sees the same hidden environments,
    so a refresh from any consumer publishes the same poses and repeated refreshes are free.
    """

    def __init__(self, scene: OvstageBackend, provider: SceneDataProvider, usd_stage: Usd.Stage, device: str):
        """Bind the provider's body paths on the populated stage and preserve their authored scales.

        Args:
            scene: Populated stage that owns this publisher.
            provider: Source of body paths and versioned physics poses.
            usd_stage: Authored USD stage providing the bodies' scales.
            device: Warp device of the simulation.
        """
        self._scene = scene
        self._provider = provider
        self._device = wp.get_device(device)
        self._paths = list(provider.backend.transform_paths)
        self._scales = body_transform_scales(usd_stage, scene.clone_copies, self._paths, str(self._device))
        self._output = SceneDataFormat.TransposedMatrix44d()
        self._timestamp = -1
        self._hidden_envs: wp.array | None = None
        self._hidden_poses: wp.array | None = None
        self._body_envs: wp.array | None = None
        with contextlib.ExitStack() as resources:
            self._query = resources.enter_context(scene.query(self._paths))
            scene.stage.write_attribute(
                self._query,
                "omni:resetXformStack",
                ordinal=scene.ordinal,
                tensors=np.ones(len(self._paths), dtype=np.bool_),
                is_array=False,
            ).wait()
            self._resources = resources.pop_all()

    def set_hidden_environments(self, hidden: Sequence[bool]) -> None:
        """Move the bodies of hidden environments out of view from the next publication on.

        Writing USD ``visibility`` on a cloned environment corrupts its body transforms once it is
        shown again, so hidden bodies are moved far below the scene instead.

        Args:
            hidden: Per-environment flag, in the stage's environment order.
        """
        env_paths = self._scene.clone_env_paths
        if len(hidden) != len(env_paths):
            raise ValueError(f"Expected {len(env_paths)} visibility flags, got {len(hidden)}.")
        mask = np.zeros(len(env_paths) + 1, dtype=np.uint8)  # The last entry stands for bodies outside any env.
        mask[:-1] = np.asarray(hidden, dtype=bool)
        self._hidden_envs = wp.array(mask, dtype=wp.uint8, device=self._device) if mask.any() else None
        if self._hidden_envs is not None and self._body_envs is None:
            env_ids = body_environment_ids(self._paths, env_paths)
            self._body_envs = wp.array(env_ids, dtype=wp.int32, device=self._device)
        self._timestamp = -1  # Publish again even when the simulation has not stepped.

    def publish(self) -> None:
        """Write changed poses at the stage's open ordinal; the caller commits."""
        timestamp = self._provider.backend.transforms_timestamp
        if timestamp == self._timestamp:
            return
        if not self._provider.get_transforms(self._output, scales=self._scales):
            return
        stream = self._device.stream if self._device.is_cuda else None
        matrices = self._output.matrices
        if self._hidden_envs is not None:
            # The publication may be shared with other consumers, so hidden poses are written to a copy.
            if self._hidden_poses is None or len(self._hidden_poses) != len(matrices):
                self._hidden_poses = wp.empty(len(matrices), dtype=wp.mat44d, device=self._device)
            wp.launch(
                hide_environment_bodies_kernel,
                len(matrices),
                [matrices, self._body_envs, self._hidden_envs, HIDDEN_ENVIRONMENT_DROP, self._hidden_poses],
                device=self._device,
                stream=stream,
            )
            matrices = self._hidden_poses
        # The write waits for consumption; the producing CUDA stream orders access to the borrowed buffer.
        ordering = {"cuda_stream": stream.cuda_stream or 1} if stream is not None else {}
        self._scene.stage.write_attribute(
            self._query,
            "omni:xform",
            ordinal=self._scene.ordinal,
            tensors=xform_tensor_from_warp(matrices),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
            **ordering,
        ).wait()
        self._timestamp = timestamp

    def close(self) -> None:
        """Release the pose query before the stage releases its paths."""
        self._resources.close()
