# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The camera's pooling of its sensor buffers, segmentation and sensor mask, without a simulator.

fovi's poolers are tested in fovi; these tests cover what the camera builds on them.
"""

import pytest
import torch
from fovi.sensing.projection import CameraModel
from isaaclab_fovi.sensors import FoveatedCameraCfg, pooling
from isaaclab_fovi.sensors.pooling import (
    build_pooler,
    build_retinal_coords,
    pool_hwc_buffer,
    pool_segmentation_buffer,
    segmentation_neighbor_indices,
)

from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.kitless

DEVICE = next(iter(test_devices(DeviceScope.DEFAULT_CUDA)), "cpu")
"""The modal pooling runs the same tensor operations on every device; the default GPU is what simulations use."""

CAMERA_MODEL = CameraModel("pinhole", (1024, 1024), (512.0, 512.0, 511.5, 511.5))


def pooling_cfg(**overrides) -> FoveatedCameraCfg:
    """A pooled foveated camera configuration for the pooling helpers, with ``overrides`` applied.

    A CMF constant of a tenth of the 90 degree pinhole window, and a small isotropic retina pooled from three
    visual-field neighbours.
    """
    values = dict(
        prim_path="/World/Camera",
        cmf_a_deg=9.0,
        foveated_res=8,
        foveated_style="isotropic",
        pool_enabled=True,
        pool_k=3,
        pool_sample_cortex=False,
    )
    return FoveatedCameraCfg(**{**values, **overrides})


def test_camera_pools_its_buffers_by_its_config() -> None:
    cfg = pooling_cfg(pool_mode="max")
    lowres, highres = build_retinal_coords(cfg, CAMERA_MODEL, "cpu")
    # The photoreceptor grid shares the retina's geometry at a higher resolution, and the pooler takes the config.
    assert len(highres) > len(lowres) and highres.cmf_a == lowres.cmf_a and highres.fov == lowres.fov
    pooler, neighbors = build_pooler(cfg, lowres, highres, "cpu")
    assert neighbors == 3 and pooler.mode == "max"
    # Pooling keeps each environment's channels apart and in order.
    constants = torch.tensor([[0.25, 0.5, 0.75], [3.0, 6.0, 9.0]])
    rendered = constants[:, None, None, :].expand(2, len(highres), 1, 3).clone()
    actual = pool_hwc_buffer(pooler, rendered)
    assert actual.dtype == rendered.dtype
    torch.testing.assert_close(actual, constants[:, None, None, :].expand(2, len(lowres), 1, 3))
    # Without pooling there is no photoreceptor grid.
    cfg = pooling_cfg(pool_enabled=False)
    _, highres = build_retinal_coords(cfg, CAMERA_MODEL, "cpu")
    assert highres is None


# Three output samples over five input samples; 5 marks an excluded neighbour. Columns are outputs, rows are
# neighbour slots nearest first.
HAND_TABLE = torch.tensor([[0, 3, 5], [1, 4, 5], [2, 5, 5]])


def test_modal_pooling_keeps_majority_color_vector() -> None:
    # Output 0 sees [0,0,0,255] twice and [0,0,1,255] once; a per-channel max would give [0,0,1,255].
    # Output 1 sees one vote each for two colors (its third slot is excluded): the nearest neighbour wins.
    # Output 2 has no valid neighbour and is zero.
    colors = torch.tensor(
        [[0, 0, 0, 255], [0, 0, 0, 255], [0, 0, 1, 255], [10, 20, 30, 255], [200, 100, 0, 255]], dtype=torch.uint8
    )
    rendered = colors[None, :, None, :].to(DEVICE)
    pooled = pool_segmentation_buffer(HAND_TABLE.to(DEVICE), rendered)
    expected = torch.tensor([[0, 0, 0, 255], [10, 20, 30, 255], [0, 0, 0, 0]], dtype=torch.uint8)
    assert pooled.dtype == torch.uint8
    assert torch.equal(pooled.cpu(), expected[None, :, None, :])
    # Swapping the tied neighbours' order flips the winner.
    swapped = HAND_TABLE[[1, 0, 2]].to(DEVICE)
    assert torch.equal(pool_segmentation_buffer(swapped, rendered)[0, 1, 0].cpu(), colors[4])


def test_modal_pooling_preserves_ids_above_float_precision() -> None:
    # 2**24 + 1 and 2**24 + 3 round to other values in float32; the IDs must come back exactly.
    big = 2**40
    ids = torch.tensor([2**24 + 1, 2**24 + 1, 2**24 + 3, big + 7, big + 7], dtype=torch.int64)
    batch = torch.stack([ids, ids.flip(0)])[:, :, None, None].to(DEVICE)
    pooled = pool_segmentation_buffer(HAND_TABLE.to(DEVICE), batch)
    expected = torch.tensor([[2**24 + 1, big + 7, 0], [big + 7, 2**24 + 1, 0]], dtype=torch.int64)
    assert pooled.dtype == torch.int64
    assert torch.equal(pooled[..., 0, 0].cpu(), expected)
    # Two int64 channels exceed the 8-byte packed key; vectors differing only in the second channel stay distinct.
    vectors = torch.tensor([[1, 2**40], [1, 2**40 + 1], [1, 2**40 + 1], [5, 6], [7, 8]], dtype=torch.int64)
    pooled = pool_segmentation_buffer(HAND_TABLE.to(DEVICE), vectors[None, :, None, :].to(DEVICE))
    assert torch.equal(pooled[0, :, 0].cpu(), torch.tensor([[1, 2**40 + 1], [5, 6], [0, 0]]))


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
    ("style", "pool_k", "sample_cortex"), [("warped_cartesian_as_grid", 9, False), ("isotropic", 3, False)]
)
def test_modal_pooling_uses_the_poolers_neighbourhoods(monkeypatch, style: str, pool_k: int, sample_cortex) -> None:
    """Segmentation pools over the same neighbourhoods as the camera's other outputs: grid windows or KNN."""
    cfg = pooling_cfg(foveated_style=style, pool_k=pool_k, pool_sample_cortex=sample_cortex, pool_mode="max")
    lowres, highres = build_retinal_coords(cfg, CAMERA_MODEL, "cpu")
    pooler, k = build_pooler(cfg, lowres, highres, "cpu")
    table = segmentation_neighbor_indices(pooler)
    assert table.shape == (k, len(lowres))
    # The table holds exactly the samples the max pooler reduces over: a max over the table matches it.
    values = torch.rand(1, len(highres), 1, 1, generator=torch.Generator().manual_seed(0)) + 1
    padded = torch.cat([values[0, :, 0, 0], torch.zeros(1)])
    torch.testing.assert_close(padded[table].max(dim=0).values, pool_hwc_buffer(pooler, values)[0, :, 0, 0])

    # Few labels so neighbourhoods hold repeats and ties; tiny chunks exercise the batched loop.
    monkeypatch.setattr(pooling, "_MODAL_CHUNK_ELEMENTS", 1)
    labels = torch.randint(0, 3, (3, len(highres), 1, 2), generator=torch.Generator().manual_seed(1)).to(torch.uint8)
    pooled = pool_segmentation_buffer(table, labels)
    assert torch.equal(pooled[:, :, 0], reference_modal_pool(table, labels[:, :, 0]))


@pytest.mark.parametrize(("style", "pool_k", "mode"), [("warped_cartesian_as_grid", 9, "avg"), ("isotropic", 6, "max")])
def test_pooling_leaves_out_photoreceptors_the_camera_does_not_see(style: str, pool_k: int, mode: str) -> None:
    cfg = pooling_cfg(foveated_style=style, pool_mode=mode, pool_k=pool_k)
    lowres, highres = build_retinal_coords(cfg, CAMERA_MODEL, DEVICE)
    pooler, _ = build_pooler(cfg, lowres, highres, DEVICE)
    # The camera's image edge cuts the retina at x = 0.3 in the first environment; the second sees everything.
    visible = torch.stack([highres.cartesian[:, 0] < 0.3, torch.ones(len(highres), dtype=torch.bool, device=DEVICE)])
    # The camera zeroes unseen photoreceptors before pooling. A negative value lets those zeros win a plain max.
    value = -0.7 if mode == "max" else 0.7
    values = torch.where(visible, value, 0.0)[:, :, None, None]
    pooled = pool_hwc_buffer(pooler, values, visible)[:, :, 0, 0]
    unseen = pool_hwc_buffer(pooler, visible.float()[:, :, None, None], visible)[:, :, 0, 0] == 0
    # Some outputs straddle the edge: plain pooling lets the mask's zeros into them.
    straddling = (pool_hwc_buffer(pooler, values)[0, :, 0, 0] - value).abs() > 0.01
    assert (straddling & ~unseen[0] & lowres.valid_mask).any()
    # Each output pools only what the camera sees: the constant where it sees any photoreceptor, zero where it sees
    # none or lies outside the retina.
    expected = torch.where(unseen | ~lowres.valid_mask, 0.0, value)
    torch.testing.assert_close(pooled, expected)
    assert unseen[0].any() and not unseen[1, lowres.valid_mask].any()


def test_modal_pooling_leaves_out_labels_the_camera_does_not_see() -> None:
    # Output 0 sees labels [1, 1, 2]; hiding the two 1s leaves 2. Output 1 sees [4, 5]; hiding both leaves nothing.
    labels = torch.tensor([1, 1, 2, 4, 5], dtype=torch.int32)[None, :, None, None].expand(2, -1, -1, -1)
    visible = torch.tensor([[False, False, True, False, False], [True, True, True, True, True]])
    pooled = pool_segmentation_buffer(HAND_TABLE, labels, visible)[..., 0, 0]
    assert torch.equal(pooled, torch.tensor([[2, 0, 0], [1, 4, 0]], dtype=torch.int32))
