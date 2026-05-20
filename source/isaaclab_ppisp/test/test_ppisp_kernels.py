# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp
from isaaclab_ppisp import PpispCfg, apply_ppisp_to_rgba, normalize_ppisp_cfg

from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg

wp.init()


def _hdr(fill: float, shape: tuple[int, ...]) -> wp.array:
    return wp.full(shape=shape, value=wp.float32(fill), dtype=wp.float32)


def _rgba(shape: tuple[int, ...]) -> wp.array:
    return wp.zeros(shape=shape, dtype=wp.uint8)


def test_ppisp_warp_exposure_increases_ldr_output():
    hdr_color = _hdr(0.25, (1, 4, 4, 3))
    baseline = _rgba((1, 4, 4, 4))
    exposed = _rgba((1, 4, 4, 4))

    apply_ppisp_to_rgba(hdr_color, baseline, normalize_ppisp_cfg(PpispCfg(inputs={"exposureOffset": 0.0})))
    apply_ppisp_to_rgba(hdr_color, exposed, normalize_ppisp_cfg(PpispCfg(inputs={"exposureOffset": 1.0})))

    baseline_np = baseline.numpy()
    exposed_np = exposed.numpy()
    assert (baseline_np[..., 3] == 255).all()
    assert (exposed_np[..., 3] == 255).all()
    assert exposed_np[..., :3].astype(float).mean() > baseline_np[..., :3].astype(float).mean()


def test_ppisp_warp_responsivity_lifts_dim_input():
    """Responsivity is applied pre-PPISP and recovers near-zero linear inputs."""
    hdr_color = _hdr(1.0e-3, (1, 4, 4, 3))
    baseline = _rgba((1, 4, 4, 4))
    boosted = _rgba((1, 4, 4, 4))

    apply_ppisp_to_rgba(hdr_color, baseline, normalize_ppisp_cfg(PpispCfg(inputs={"responsivity": 1.0})))
    apply_ppisp_to_rgba(hdr_color, boosted, normalize_ppisp_cfg(PpispCfg(inputs={"responsivity": 1000.0})))

    baseline_np = baseline.numpy()
    boosted_np = boosted.numpy()
    assert (baseline_np[..., 3] == 255).all()
    assert (boosted_np[..., 3] == 255).all()
    assert boosted_np[..., :3].astype(float).mean() > baseline_np[..., :3].astype(float).mean()


def test_ppisp_warp_vignetting_uses_detiled_camera_coordinates():
    hdr_color = _hdr(0.5, (2, 5, 5, 3))
    rgba = _rgba((2, 5, 5, 4))

    apply_ppisp_to_rgba(
        hdr_color,
        rgba,
        normalize_ppisp_cfg(
            PpispCfg(
                inputs={
                    "vignettingAlpha1R": -1.0,
                    "vignettingAlpha1G": -1.0,
                    "vignettingAlpha1B": -1.0,
                }
            )
        ),
    )

    rgba_np = rgba.numpy()
    assert (rgba_np[:, 2, 2, :3] > rgba_np[:, 0, 0, :3]).all()
    assert (rgba_np[0] == rgba_np[1]).all()


def test_ppisp_warp_crf_extreme_centers_stay_finite():
    import numpy as np

    hdr_data = np.array(
        [
            [
                [[0.0, 0.25, 0.5], [0.75, 1.0, 0.5]],
                [[0.0, 0.25, 0.5], [0.75, 1.0, 0.5]],
            ]
        ],
        dtype=np.float32,
    )
    hdr_color = wp.from_numpy(hdr_data, dtype=wp.float32)
    rgba = _rgba((1, 2, 2, 4))

    apply_ppisp_to_rgba(
        hdr_color,
        rgba,
        normalize_ppisp_cfg(
            PpispCfg(
                inputs={
                    "crfCenterR": -100.0,
                    "crfCenterG": 100.0,
                    "crfCenterB": -100.0,
                }
            )
        ),
    )

    rgba_np = rgba.numpy()
    assert (rgba_np[..., 3] == 255).all()
    assert np.isfinite(rgba_np.astype(float)).all()


def test_tiled_camera_cfg_accepts_ppisp_cfg():
    ppisp_cfg = PpispCfg(inputs={"exposureOffset": 1.0})

    cfg = TiledCameraCfg(
        prim_path="/World/Camera",
        width=4,
        height=4,
        data_types=["rgb"],
        isp_cfg=ppisp_cfg,
    )

    assert cfg.isp_cfg == ppisp_cfg
