# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dual-grid geometry and modality-specific pooling without a simulator."""

from dataclasses import dataclass, field

import pytest
import torch
from fovi.arch.knn import KNNPoolingLayer
from fovi.arch.pooling import GridPoolingLayer
from isaaclab_fovi.sensors.pooling import (
    build_highres_coords,
    build_lowres_coords,
    build_pooler,
    is_segmentation_output,
    pool_hwc_buffer,
)

from isaaclab.sim import PinholeCameraCfg


@dataclass
class SamplingConfig:
    """Sampling fields consumed by the shared pooling helpers."""

    # A 90 degree pinhole window (focal length 1, aperture 2) with a CMF constant of a tenth of it.
    cmf_a_deg: float = 9.0
    spawn: PinholeCameraCfg = field(default_factory=lambda: PinholeCameraCfg(focal_length=1.0, horizontal_aperture=2.0))
    camera_model: dict[str, str | tuple[int, int] | tuple[float, ...]] | None = None
    retinal_window_reference_side: str = "short"
    retinal_window_short_side_fraction: float = 1.0
    foveated_res: int = 8
    foveated_style: str = "isotropic"
    fov_type: str = "circular"
    radius_norm: float = 2.0
    isotropic_plotting_type: str = "v1like"
    pool_enabled: bool = True
    pool_res_mult: float = 2.0
    pool_k: int | None = 3
    pool_mode: str = "avg"
    pool_sample_cortex: bool = False
    pool_gauss_sigma: float = 10.0


@pytest.mark.parametrize("mode", ["avg", "gaussian", "max"])
def test_dual_grid_preserves_constant_channels_and_batch_order(mode: str) -> None:
    cfg = SamplingConfig(pool_mode=mode)
    lowres, matched = build_lowres_coords(cfg, "cpu")
    highres = build_highres_coords(cfg, matched, "cpu")
    assert highres is not None
    assert len(highres) > len(lowres)
    assert highres.cmf_a == lowres.cmf_a
    pooler, neighbors = build_pooler(cfg, lowres, highres, "cpu")
    assert pooler is not None
    assert neighbors == 3
    constants = torch.tensor([[0.25, 0.5, 0.75], [3.0, 6.0, 9.0]])
    rendered = constants[:, None, None, :].expand(2, len(highres), 1, 3).clone()
    actual = pool_hwc_buffer(pooler, rendered)
    expected = constants[:, None, None, :].expand(2, len(lowres), 1, 3)
    assert actual.dtype == rendered.dtype
    torch.testing.assert_close(actual, expected)


def test_disabled_pooling_uses_only_output_grid() -> None:
    cfg = SamplingConfig(pool_enabled=False)
    lowres, matched = build_lowres_coords(cfg, "cpu")
    assert build_highres_coords(cfg, matched, "cpu") is None
    assert build_pooler(cfg, lowres, None, "cpu") == (None, None)


def test_segmentation_pooler_override_preserves_discrete_label_values() -> None:
    cfg = SamplingConfig(pool_mode="avg")
    lowres, matched = build_lowres_coords(cfg, "cpu")
    highres = build_highres_coords(cfg, matched, "cpu")
    assert highres is not None
    pooler, _ = build_pooler(cfg, lowres, highres, "cpu", mode="max")
    assert pooler is not None
    labels = torch.tensor([2.0, 7.0, 19.0])
    rendered = labels[torch.arange(len(highres)) % len(labels)].reshape(1, -1, 1, 1)
    pooled = pool_hwc_buffer(pooler, rendered)
    assert pooled.shape == (1, len(lowres), 1, 1)
    assert torch.isin(pooled, labels).all()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("rgb", False),
        ("distance_to_camera", False),
        ("semantic_segmentation", True),
        ("instance_segmentation_fast", True),
        ("instance_id_segmentation_fast", True),
    ],
)
def test_segmentation_dispatch_uses_semantics(name: str, expected: bool) -> None:
    assert is_segmentation_output(name) is expected


@pytest.mark.parametrize("mode", ["avg", "gaussian", "max"])
@pytest.mark.parametrize("sample_cortex", [False, True, "geodesic"])
@pytest.mark.parametrize("style", ["warped_cartesian_as_grid", "warped_cartesian"])
def test_native_grid_pooling_excludes_invalid_photoreceptors(mode, sample_cortex, style) -> None:
    cfg = SamplingConfig(
        foveated_style=style,
        pool_mode=mode,
        pool_sample_cortex=sample_cortex,
        # 3x3 grid windows; KNN keeps the default neighbor count.
        pool_k=9 if style.endswith("_as_grid") else 3,
    )
    lowres, matched = build_lowres_coords(cfg, "cpu")
    highres = build_highres_coords(cfg, matched, "cpu")
    assert highres is not None
    pooler, _ = build_pooler(cfg, lowres, highres, "cpu")
    values = torch.ones(1, len(highres), 1, 1)
    values[:, ~highres.valid_mask] = 1e6
    result = pool_hwc_buffer(pooler, values)[0, :, 0, 0]
    assert (result[~lowres.valid_mask] == 0).all()
    assert (result[lowres.valid_mask] <= 1).all()
    assert (result[lowres.valid_mask] > 0).all()


@pytest.mark.parametrize(
    "style,expected",
    [
        ("warped_cartesian_as_grid", GridPoolingLayer),
        ("warped_cartesian", KNNPoolingLayer),
    ],
)
def test_native_grid_default_kernel_matches_deployment(style: str, expected: type) -> None:
    cfg = SamplingConfig(
        foveated_style=style,
        pool_k=None,
        pool_sample_cortex=True,
    )
    lowres, matched = build_lowres_coords(cfg, "cpu")
    highres = build_highres_coords(cfg, matched, "cpu")
    pooler, resolved_k = build_pooler(cfg, lowres, highres, "cpu")
    assert resolved_k == 4
    assert isinstance(pooler, expected)
