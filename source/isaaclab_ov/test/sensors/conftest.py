# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared scaffolding for the OVPhysX sensor CUDA-graph unit tests.

The graph tests in this directory all build a sensor through ``__new__`` (no USD stage),
hand it a fake native view that counts reads, and then assert on capture/replay behaviour.
The sensor-specific wiring stays in each test module; the pieces that are byte-identical
across sensors live here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
import warp as wp

__all__ = [
    "CountingReadView",
    "assert_invalidation_drops_captured_graph",
    "assert_update_refused_inside_outer_capture",
    "make_identity_quat_poses",
    "requires_cuda",
]


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
"""Skip marker for tests that need a real CUDA device to capture and replay graphs."""


class CountingReadView:
    """Fake OVPhysX view that counts native reads and optionally fills the destination buffer.

    Both native call shapes are supported: :meth:`read_into` (tensor-type keyed views) and
    :meth:`read` (single-tensor bindings). When ``source`` is given, every read copies it into
    the caller-owned destination buffer, so in-place mutations of ``source`` are picked up by
    later reads. When it is ``None``, reads are only counted.

    Args:
        source: Torch tensor copied into the destination buffer on each read, or ``None`` to
            only count reads.
        dtype: Warp dtype the trailing source dimension is reinterpreted as before copying
            (e.g. ``wp.transformf``), or ``None`` to copy the tensor's native layout.
        expected_tensor_type: Tensor type :meth:`read_into` must be called with, or ``None``
            to skip the check.
    """

    def __init__(
        self,
        source: torch.Tensor | None = None,
        dtype: Any | None = None,
        expected_tensor_type: Any | None = None,
    ):
        self.read_count = 0
        self.source_torch = source
        self._dtype = dtype
        self._expected_tensor_type = expected_tensor_type

    def read_into(self, tensor_type: Any = None, dst: wp.array | None = None, *args, **kwargs) -> None:
        """Count a read keyed by ``tensor_type`` and fill ``dst`` from the source tensor."""
        if self._expected_tensor_type is not None:
            assert tensor_type == self._expected_tensor_type
        self.read_count += 1
        self._copy_into(dst)

    def read(self, dst: wp.array | None = None) -> None:
        """Count a read and fill ``dst`` from the source tensor."""
        self.read_count += 1
        self._copy_into(dst)

    def _copy_into(self, dst: wp.array | None) -> None:
        """Copy the current source contents into ``dst`` when both are available."""
        if dst is None or self.source_torch is None:
            return
        source = self.source_torch.contiguous()
        src = wp.from_torch(source) if self._dtype is None else wp.from_torch(source, dtype=self._dtype)
        wp.copy(dst, src)


def make_identity_quat_poses(translations: torch.Tensor) -> torch.Tensor:
    """Build ``(num_envs, 7)`` poses [m, -] with the given translations and identity quaternions.

    Args:
        translations: Per-environment translations [m], shape ``(num_envs, 3)``.

    Returns:
        Poses laid out as ``(x, y, z, qx, qy, qz, qw)`` with identity rotations.
    """
    num_envs = translations.shape[0]
    poses = torch.zeros((num_envs, 7), dtype=torch.float32, device=translations.device)
    poses[:, :3] = translations
    poses[:, 6] = 1.0  # identity quaternion (x, y, z, w) = (0, 0, 0, 1)
    return poses


def assert_update_refused_inside_outer_capture(sensor: Any, update: Callable[[], Any], reader: Any) -> None:
    """Assert the sensor update refuses to run inside an outer CUDA-graph capture.

    Opens a real :class:`warp.ScopedCapture` on the sensor device, calls ``update`` inside it,
    and requires the refusal to happen before any native read is issued.

    Args:
        sensor: Sensor under test, used for its ``_device``.
        update: Zero-argument callable invoking the sensor's buffer update.
        reader: Fake view or binding exposing a ``read_count`` attribute.
    """
    device = wp.get_device(sensor._device)
    scratch_src = wp.ones(1, dtype=wp.int32, device=device)
    scratch_dst = wp.zeros(1, dtype=wp.int32, device=device)

    with wp.ScopedCapture(device=device):
        with pytest.raises(RuntimeError, match="CUDA graph capture is active"):
            update()
        wp.copy(scratch_dst, scratch_src)  # keep the outer capture non-empty

    assert reader.read_count == 0


def assert_invalidation_drops_captured_graph(
    sensor: Any, update: Callable[[], Any], base_class: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert invalidation drops the captured update graph alongside the native handles.

    Runs ``update`` once to capture the graph, then fires the invalidation callback with the
    base-class implementation neutralized (it would touch native handles that do not exist on
    a scene-free sensor).

    Args:
        sensor: Sensor under test, exposing ``_update_graph``.
        update: Zero-argument callable invoking the sensor's buffer update.
        base_class: Base sensor class whose ``_invalidate_initialize_callback`` is stubbed out.
        monkeypatch: Pytest monkeypatch fixture used to stub the base callback.
    """
    update()
    wp.synchronize_device(sensor._device)
    assert sensor._update_graph.is_captured

    monkeypatch.setattr(base_class, "_invalidate_initialize_callback", lambda self, event: None)
    sensor._invalidate_initialize_callback(None)
    assert not sensor._update_graph.is_captured
