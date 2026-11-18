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
from isaaclab_fovi.sensors import pooling
from isaaclab_fovi.sensors.pooling import (
    build_highres_coords,
    build_lowres_coords,
    build_pooler,
    pool_hwc_buffer,
    pool_segmentation_buffer,
    segmentation_neighbor_indices,
)

from isaaclab.sim import PinholeCameraCfg
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.kitless


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


def test_dual_grid_preserves_constant_channels_and_batch_order() -> None:
    cfg = SamplingConfig()
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


def reference_pool(pooler: KNNPoolingLayer, values: torch.Tensor, mode: str, sigma: float) -> torch.Tensor:
    """Pool ``(B, N_hi, 1, C)`` values over a KNN pooler's neighbour table by the definition of each mode."""
    table, distances = pooler.knn_indices_pad_token, pooler.knn_distances  # (k, N_lo); N_hi marks a pad
    num_inputs = values.shape[1]
    valid = table != num_inputs
    neighbours = values[:, table.clamp(max=num_inputs - 1), 0]  # (B, k, N_lo, C)
    if mode == "max":
        pooled = neighbours.masked_fill(~valid[None, :, :, None], -torch.inf).amax(1)
    else:
        if mode == "avg":
            weights = valid.double()
        else:
            # Gaussian in the neighbour distance rescaled to [0, 1] per output, nearest weighted most.
            low, high = distances.min(0).values, distances.max(0).values
            scaled = (distances - low) / torch.where(high > low, high - low, 1.0)
            weights = torch.exp(-scaled.double().square() / (2 * sigma**2)) * valid
        weights = weights / weights.sum(0)
        pooled = (neighbours.double() * weights[None, :, :, None]).sum(1).to(values.dtype)
    return pooled.unsqueeze(2)


@pytest.mark.parametrize("mode", ["avg", "gaussian", "max"])
def test_pooling_mode_reduces_each_neighbourhood(mode: str) -> None:
    # A narrow Gaussian weights the nearest of the three neighbours well above the farthest, so it differs from avg.
    sigma = 0.5
    cfg = SamplingConfig(pool_mode=mode, pool_gauss_sigma=sigma)
    lowres, matched = build_lowres_coords(cfg, "cpu")
    highres = build_highres_coords(cfg, matched, "cpu")
    pooler, _ = build_pooler(cfg, lowres, highres, "cpu")
    assert isinstance(pooler, KNNPoolingLayer)
    values = torch.rand(2, len(highres), 1, 3, generator=torch.Generator().manual_seed(0))
    actual = pool_hwc_buffer(pooler, values)
    expected = {other: reference_pool(pooler, values, other, sigma) for other in ("avg", "gaussian", "max")}
    torch.testing.assert_close(actual, expected[mode])
    # The modes give distinct results on this input, so the match identifies the reduction that ran.
    for other, other_expected in expected.items():
        if other != mode:
            assert (actual - other_expected).abs().max() > 0.05


def test_disabled_pooling_uses_only_output_grid() -> None:
    cfg = SamplingConfig(pool_enabled=False)
    lowres, matched = build_lowres_coords(cfg, "cpu")
    assert build_highres_coords(cfg, matched, "cpu") is None
    assert build_pooler(cfg, lowres, None, "cpu") == (None, None)


DEVICES = test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA)

# Three output samples over five input samples; 5 marks an excluded neighbour. Columns are outputs, rows are
# neighbour slots nearest first.
HAND_TABLE = torch.tensor([[0, 3, 5], [1, 4, 5], [2, 5, 5]])


def hand_table(device: str) -> torch.Tensor:
    return HAND_TABLE.to(device)


@pytest.mark.parametrize("device", DEVICES)
def test_modal_pooling_keeps_majority_color_vector(device: str) -> None:
    # Output 0 sees [0,0,0,255] twice and [0,0,1,255] once; a per-channel max would give [0,0,1,255].
    # Output 1 sees one vote each for two colors (its third slot is excluded): the nearest neighbour wins.
    # Output 2 has no valid neighbour and is zero.
    colors = torch.tensor(
        [[0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 1, 255], [10, 20, 30, 255], [200, 100, 0, 255]], dtype=torch.uint8
    )
    rendered = colors[None, :, None, :].to(device)
    pooled = pool_segmentation_buffer(hand_table(device), rendered)
    expected = torch.tensor([[0, 0, 0, 255], [10, 20, 30, 255], [0, 0, 0, 0]], dtype=torch.uint8)
    assert pooled.dtype == torch.uint8
    assert torch.equal(pooled.cpu(), expected[None, :, None, :])
    # Swapping the tied neighbours' order flips the winner.
    swapped = HAND_TABLE[[1, 0, 2]].to(device)
    assert torch.equal(pool_segmentation_buffer(swapped, rendered)[0, 1, 0].cpu(), colors[4])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_modal_pooling_preserves_ids_above_float_precision(device: str, dtype: torch.dtype) -> None:
    # 2**24 + 1 and 2**24 + 3 round to other values in float32; the IDs must come back exactly.
    big = 2**30 if dtype == torch.int32 else 2**40
    ids = torch.tensor([2**24 + 1, 2**24 + 1, 2**24 + 3, big + 7, big + 7], dtype=dtype)
    batch = torch.stack([ids, ids.flip(0)])[:, :, None, None].to(device)
    pooled = pool_segmentation_buffer(hand_table(device), batch)
    expected = torch.tensor([[2**24 + 1, big + 7, 0], [big + 7, 2**24 + 1, 0]], dtype=dtype)
    assert pooled.dtype == dtype
    assert torch.equal(pooled[..., 0, 0].cpu(), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_modal_pooling_compares_wide_vectors_exactly(device: str) -> None:
    # Two int64 channels exceed the 8-byte packed key; vectors differing only in the second channel stay distinct.
    vectors = torch.tensor([[1, 2**40], [1, 2**40 + 1], [1, 2**40 + 1], [5, 6], [7, 8]], dtype=torch.int64)
    pooled = pool_segmentation_buffer(hand_table(device), vectors[None, :, None, :].to(device))
    expected = torch.tensor([[1, 2**40 + 1], [5, 6], [0, 0]])
    assert torch.equal(pooled[0, :, 0].cpu(), expected)


def reference_modal_pool(table: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Brute-force modal pooling of ``(B, N_hi, C)`` values over a ``(k, N_lo)`` neighbour table."""
    batch, num_inputs, channels = values.shape
    out = torch.zeros(batch, table.shape[1], channels, dtype=values.dtype)
    for b in range(batch):
        for n in range(table.shape[1]):
            neighbours = [tuple(values[b, i].tolist()) for i in table[:, n].tolist() if i != num_inputs]
            if neighbours:
                # max() keeps the first of equally common values, i.e. the earliest neighbour slot.
                out[b, n] = torch.tensor(max(neighbours, key=neighbours.count), dtype=values.dtype)
    return out


@pytest.mark.parametrize(
    ("style", "pool_k", "sample_cortex"),
    [
        ("warped_cartesian_as_grid", 4, False),
        ("warped_cartesian_as_grid", 9, False),
        ("warped_cartesian", 3, True),
        ("isotropic", 3, False),
    ],
)
def test_modal_pooling_uses_fovi_neighbourhoods(monkeypatch, style: str, pool_k: int, sample_cortex: bool) -> None:
    cfg = SamplingConfig(foveated_style=style, pool_k=pool_k, pool_sample_cortex=sample_cortex)
    lowres, matched = build_lowres_coords(cfg, "cpu")
    highres = build_highres_coords(cfg, matched, "cpu")
    fovi_max, k = build_pooler(cfg, lowres, highres, "cpu", mode="max")
    # The camera takes the table from its standard pooler, whatever that pooler's mode.
    standard, _ = build_pooler(cfg, lowres, highres, "cpu")
    table = segmentation_neighbor_indices(standard)
    assert table.shape == (k, len(lowres))
    assert torch.equal(segmentation_neighbor_indices(fovi_max), table)

    # The table holds exactly the samples fovi's max pooler reduces over: a max over the table matches it.
    values = torch.rand(1, len(highres), 1, 1, generator=torch.Generator().manual_seed(0)) + 1
    padded = torch.cat([values[0, :, 0, 0], torch.zeros(1)])
    torch.testing.assert_close(padded[table].max(dim=0).values, pool_hwc_buffer(fovi_max, values)[0, :, 0, 0])

    # Few labels so neighbourhoods hold repeats and ties; tiny chunks exercise the batched loop.
    monkeypatch.setattr(pooling, "_MODAL_CHUNK_ELEMENTS", 1)
    labels = torch.randint(0, 3, (3, len(highres), 1, 2), generator=torch.Generator().manual_seed(1))
    labels = labels.to(torch.uint8)
    pooled = pool_segmentation_buffer(table, labels)
    assert torch.equal(pooled[:, :, 0], reference_modal_pool(table, labels[:, :, 0]))


def test_modal_mode_builds_no_pooler_of_its_own() -> None:
    cfg = SamplingConfig()
    lowres, matched = build_lowres_coords(cfg, "cpu")
    highres = build_highres_coords(cfg, matched, "cpu")
    with pytest.raises(ValueError, match="segmentation_neighbor_indices"):
        build_pooler(cfg, lowres, highres, "cpu", mode="mode")


@pytest.mark.parametrize("device", DEVICES)
def test_modal_neighbourhoods_stay_on_the_sensor_device(device: str) -> None:
    cfg = SamplingConfig()
    lowres, matched = build_lowres_coords(cfg, device)
    highres = build_highres_coords(cfg, matched, device)
    table = segmentation_neighbor_indices(build_pooler(cfg, lowres, highres, device)[0])
    assert table.device == torch.device(device)
    labels = torch.randint(0, 3, (1, len(highres), 1, 1), device=device)
    assert pool_segmentation_buffer(table, labels).device == torch.device(device)


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
