# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SceneDataProvider transform conversion and index mapping."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from isaaclab.scene_data.scene_data_backend import SceneDataBackend, SceneDataFormat
from isaaclab.scene_data.scene_data_provider import SceneDataProvider


class _TransformsBackend(SceneDataBackend):
    def __init__(self, transforms: np.ndarray, paths: list[str], device: str = "cpu"):
        self._transforms = SceneDataFormat.Transform()
        self._transforms.transforms = wp.array(transforms, dtype=wp.transformf, device=device)
        self._paths = paths

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        return self._transforms

    @property
    def transform_count(self) -> int:
        return int(self._transforms.transforms.shape[0])

    @property
    def transform_paths(self) -> list[str]:
        return self._paths


def _identity_transform(x: float) -> list[float]:
    return [x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.skipif(
    wp.get_cuda_device_count() == 0, reason="requires a CUDA device to reproduce the default-device mismatch"
)
def test_get_transforms_matches_backend_device_when_warp_default_is_cuda():
    """Regression test for a CPU sim + CUDA-present device mismatch.

    ``SceneDataProvider`` must allocate its internal conversion outputs and index mappings on the
    sim backend's own device, not Warp's process-global default (``cuda:0`` whenever a CUDA device
    is present, independent of ``--device``). Omitting ``device=`` previously let a CPU-configured
    sim (``--device cpu``) launch the conversion kernel against CUDA-resident output/mapping arrays
    mismatched with the CPU-resident backend data, crashing with an illegal memory access.
    """
    transforms = np.array(
        [_identity_transform(0.0), _identity_transform(1.0), _identity_transform(2.0)], dtype=np.float32
    )
    backend = _TransformsBackend(transforms, ["/World/a", "/World/b", "/World/c"], device="cpu")
    provider = SceneDataProvider(backend)

    # Force a non-identity mapping so get_transforms takes the conversion-kernel path (not the
    # type-matching passthrough), while Warp's global default resolves to a CUDA device -- exactly
    # the process state a CPU-configured sim has whenever a CUDA device is also present.
    with wp.ScopedDevice("cuda:0"):
        mapping = provider.create_mapping(["/World/c", "/World/a", "/World/b"])
        assert mapping is not None
        assert str(mapping.device) == "cpu"

        output = SceneDataFormat.Vec3_Quat()
        assert provider.get_transforms(output, mapping=mapping, allow_passthrough=False)

    assert str(output.positions.device) == "cpu"
    assert str(output.orientations.device) == "cpu"
    # Mapping sends backend index 0 ("/World/a") to output slot 1, 1 ("/World/b") to slot 2,
    # 2 ("/World/c") to slot 0.
    assert np.allclose(output.positions.numpy()[:, 0], [2.0, 0.0, 1.0])


def test_create_mapping_returns_none_for_identity_order():
    backend = _TransformsBackend(
        np.array([_identity_transform(0.0), _identity_transform(1.0)], dtype=np.float32),
        ["/World/a", "/World/b"],
    )
    provider = SceneDataProvider(backend)
    assert provider.create_mapping(["/World/a", "/World/b"]) is None


def test_create_mapping_remaps_out_of_order_paths():
    backend = _TransformsBackend(
        np.array([_identity_transform(0.0), _identity_transform(1.0)], dtype=np.float32),
        ["/World/a", "/World/b"],
    )
    provider = SceneDataProvider(backend)
    mapping = provider.create_mapping(["/World/b", "/World/a"])
    assert mapping is not None
    assert mapping.numpy().tolist() == [1, 0]
