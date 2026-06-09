# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import warp as wp
from isaaclab_ppisp import (
    PpispCfg,
    apply_ppisp_to_rgba,
    apply_ppisp_to_rgba_with_controller_params,
    compute_ppisp_controller_params,
    normalize_ppisp_cfg,
)
from isaaclab_ppisp.kernels import (
    PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN,
    PPISP_CONTROLLER_FEATURE_LEN,
    PPISP_CONTROLLER_HIDDEN_DIM,
    PPISP_CONTROLLER_INPUT_DOWNSAMPLING,
    PPISP_CONTROLLER_MLP_INPUT_DIM,
    PPISP_CONTROLLER_OFF_COL_B,
    PPISP_CONTROLLER_OFF_COL_W,
    PPISP_CONTROLLER_OFF_CONV1_B,
    PPISP_CONTROLLER_OFF_CONV1_W,
    PPISP_CONTROLLER_OFF_CONV2_B,
    PPISP_CONTROLLER_OFF_CONV2_W,
    PPISP_CONTROLLER_OFF_CONV3_B,
    PPISP_CONTROLLER_OFF_CONV3_W,
    PPISP_CONTROLLER_OFF_EXP_B,
    PPISP_CONTROLLER_OFF_EXP_W,
    PPISP_CONTROLLER_OFF_TRUNK0_B,
    PPISP_CONTROLLER_OFF_TRUNK0_W,
    PPISP_CONTROLLER_OFF_TRUNK1_B,
    PPISP_CONTROLLER_OFF_TRUNK1_W,
    PPISP_CONTROLLER_OFF_TRUNK2_B,
    PPISP_CONTROLLER_OFF_TRUNK2_W,
    PPISP_CONTROLLER_PARAM_COUNT,
)

from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg

wp.init()


def _hdr(fill: float, shape: tuple[int, ...]) -> wp.array:
    return wp.full(shape=shape, value=wp.float32(fill), dtype=wp.float32)


def _rgba(shape: tuple[int, ...]) -> wp.array:
    return wp.zeros(shape=shape, dtype=wp.uint8)


def _controller_reference_params(hdr, weights, prior_exposure: float, responsivity: float = 1.0):
    import numpy as np

    num_cameras, image_height, image_width, _ = hdr.shape
    ds_height = max(1, image_height // PPISP_CONTROLLER_INPUT_DOWNSAMPLING)
    ds_width = max(1, image_width // PPISP_CONTROLLER_INPUT_DOWNSAMPLING)

    conv1_w = weights[PPISP_CONTROLLER_OFF_CONV1_W:PPISP_CONTROLLER_OFF_CONV1_B].reshape(16, 3)
    conv1_b = weights[PPISP_CONTROLLER_OFF_CONV1_B:PPISP_CONTROLLER_OFF_CONV2_W]
    conv2_w = weights[PPISP_CONTROLLER_OFF_CONV2_W:PPISP_CONTROLLER_OFF_CONV2_B].reshape(32, 16)
    conv2_b = weights[PPISP_CONTROLLER_OFF_CONV2_B:PPISP_CONTROLLER_OFF_CONV3_W]
    conv3_w = weights[PPISP_CONTROLLER_OFF_CONV3_W:PPISP_CONTROLLER_OFF_CONV3_B].reshape(64, 32)
    conv3_b = weights[PPISP_CONTROLLER_OFF_CONV3_B:PPISP_CONTROLLER_OFF_TRUNK0_W]

    pool1 = np.empty((num_cameras, ds_height, ds_width, 16), dtype=np.float32)
    for camera_id in range(num_cameras):
        for dy in range(ds_height):
            y0 = dy * PPISP_CONTROLLER_INPUT_DOWNSAMPLING
            y1 = min(y0 + PPISP_CONTROLLER_INPUT_DOWNSAMPLING, image_height)
            for dx in range(ds_width):
                x0 = dx * PPISP_CONTROLLER_INPUT_DOWNSAMPLING
                x1 = min(x0 + PPISP_CONTROLLER_INPUT_DOWNSAMPLING, image_width)
                conv1 = hdr[camera_id, y0:y1, x0:x1].reshape(-1, 3) * responsivity
                conv1 = conv1 @ conv1_w.T + conv1_b
                pool1[camera_id, dy, dx] = np.maximum(conv1.max(axis=0), 0.0)

    conv2 = np.maximum(np.einsum("nhwi,oi->nhwo", pool1, conv2_w) + conv2_b, 0.0)
    conv3 = np.einsum("nhwi,oi->nhwo", conv2, conv3_w) + conv3_b

    pooled = np.zeros((num_cameras, 64, 5, 5), dtype=np.float32)
    for gy in range(5):
        h_start = (gy * ds_height) // 5
        h_end = min(((gy + 1) * ds_height + 4) // 5, ds_height)
        for gx in range(5):
            w_start = (gx * ds_width) // 5
            w_end = min(((gx + 1) * ds_width + 4) // 5, ds_width)
            pooled[:, :, gy, gx] = conv3[:, h_start:h_end, w_start:w_end, :].mean(axis=(1, 2))

    features = pooled.reshape(num_cameras, PPISP_CONTROLLER_FEATURE_LEN)
    x = np.concatenate(
        [features, np.full((num_cameras, 1), prior_exposure, dtype=np.float32)],
        axis=1,
    )

    trunk0_w = weights[PPISP_CONTROLLER_OFF_TRUNK0_W:PPISP_CONTROLLER_OFF_TRUNK0_B].reshape(
        PPISP_CONTROLLER_HIDDEN_DIM, PPISP_CONTROLLER_MLP_INPUT_DIM
    )
    trunk0_b = weights[PPISP_CONTROLLER_OFF_TRUNK0_B:PPISP_CONTROLLER_OFF_TRUNK1_W]
    trunk1_w = weights[PPISP_CONTROLLER_OFF_TRUNK1_W:PPISP_CONTROLLER_OFF_TRUNK1_B].reshape(
        PPISP_CONTROLLER_HIDDEN_DIM, PPISP_CONTROLLER_HIDDEN_DIM
    )
    trunk1_b = weights[PPISP_CONTROLLER_OFF_TRUNK1_B:PPISP_CONTROLLER_OFF_TRUNK2_W]
    trunk2_w = weights[PPISP_CONTROLLER_OFF_TRUNK2_W:PPISP_CONTROLLER_OFF_TRUNK2_B].reshape(
        PPISP_CONTROLLER_HIDDEN_DIM, PPISP_CONTROLLER_HIDDEN_DIM
    )
    trunk2_b = weights[PPISP_CONTROLLER_OFF_TRUNK2_B:PPISP_CONTROLLER_OFF_EXP_W]
    exp_w = weights[PPISP_CONTROLLER_OFF_EXP_W:PPISP_CONTROLLER_OFF_EXP_B]
    exp_b = weights[PPISP_CONTROLLER_OFF_EXP_B]
    col_w = weights[PPISP_CONTROLLER_OFF_COL_W:PPISP_CONTROLLER_OFF_COL_B].reshape(8, 128)
    col_b = weights[PPISP_CONTROLLER_OFF_COL_B:PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN]

    hidden = np.maximum(x @ trunk0_w.T + trunk0_b, 0.0)
    hidden = np.maximum(hidden @ trunk1_w.T + trunk1_b, 0.0)
    hidden = np.maximum(hidden @ trunk2_w.T + trunk2_b, 0.0)
    exposure = hidden @ exp_w + exp_b
    color = hidden @ col_w.T + col_b
    return np.concatenate([exposure[:, None], color], axis=1).astype(np.float32)


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


def test_ppisp_warp_controller_exposure_increases_ldr_output():
    import numpy as np

    hdr_color = _hdr(0.25, (1, 4, 4, 3))
    baseline = _rgba((1, 4, 4, 4))
    exposed = _rgba((1, 4, 4, 4))
    baseline_params = wp.zeros((1, 9), dtype=wp.float32)
    exposed_params = wp.from_numpy(
        np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.float32,
    )
    cfg = normalize_ppisp_cfg(PpispCfg())

    apply_ppisp_to_rgba_with_controller_params(hdr_color, baseline, cfg, baseline_params)
    apply_ppisp_to_rgba_with_controller_params(hdr_color, exposed, cfg, exposed_params)

    baseline_np = baseline.numpy()
    exposed_np = exposed.numpy()
    assert (baseline_np[..., 3] == 255).all()
    assert (exposed_np[..., 3] == 255).all()
    assert exposed_np[..., :3].astype(float).mean() > baseline_np[..., :3].astype(float).mean()


def test_ppisp_warp_controller_params_match_static_color_latent_order():
    import numpy as np

    hdr_data = np.linspace(0.05, 0.95, 1 * 4 * 4 * 3, dtype=np.float32).reshape(1, 4, 4, 3)
    hdr_color = wp.from_numpy(hdr_data, dtype=wp.float32)
    static = _rgba((1, 4, 4, 4))
    controller = _rgba((1, 4, 4, 4))
    inputs = {
        "exposureOffset": 0.25,
        "colorLatentBlue": (0.20, -0.10),
        "colorLatentRed": (-0.15, 0.25),
        "colorLatentGreen": (0.05, -0.30),
        "colorLatentNeutral": (0.12, 0.08),
    }
    controller_params = wp.from_numpy(
        np.array(
            [
                [
                    inputs["exposureOffset"],
                    *inputs["colorLatentBlue"],
                    *inputs["colorLatentRed"],
                    *inputs["colorLatentGreen"],
                    *inputs["colorLatentNeutral"],
                ]
            ],
            dtype=np.float32,
        ),
        dtype=wp.float32,
    )

    apply_ppisp_to_rgba(hdr_color, static, normalize_ppisp_cfg(PpispCfg(inputs=inputs)))
    apply_ppisp_to_rgba_with_controller_params(
        hdr_color,
        controller,
        normalize_ppisp_cfg(PpispCfg()),
        controller_params,
    )

    np.testing.assert_array_equal(controller.numpy(), static.numpy())


@pytest.mark.parametrize("num_cameras", [1, 2])
@pytest.mark.parametrize("image_shape", [(7, 8), (80, 96)])
def test_ppisp_controller_network_matches_numpy_reference(num_cameras, image_shape):
    import numpy as np

    if not wp.is_cuda_available():
        pytest.skip("PPISP controller requires CUDA.")

    rng = np.random.default_rng(7)
    height, width = image_shape
    hdr_np = rng.uniform(0.0, 1.0, size=(num_cameras, height, width, 3)).astype(np.float32)
    weights_np = rng.normal(0.0, 0.003, size=PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN).astype(np.float32)
    for offset, size in (
        (PPISP_CONTROLLER_OFF_CONV1_B, 16),
        (PPISP_CONTROLLER_OFF_CONV2_B, 32),
        (PPISP_CONTROLLER_OFF_CONV3_B, 64),
        (PPISP_CONTROLLER_OFF_TRUNK0_B, PPISP_CONTROLLER_HIDDEN_DIM),
        (PPISP_CONTROLLER_OFF_TRUNK1_B, PPISP_CONTROLLER_HIDDEN_DIM),
        (PPISP_CONTROLLER_OFF_TRUNK2_B, PPISP_CONTROLLER_HIDDEN_DIM),
    ):
        weights_np[offset : offset + size] += 0.04
    prior_exposure = 0.35
    responsivity = 1.7

    with wp.ScopedDevice("cuda:0"):
        hdr_color = wp.from_numpy(hdr_np, dtype=wp.float32, device="cuda:0")
        weights = wp.from_numpy(weights_np, dtype=wp.float32, device="cuda:0")
        features = wp.empty((num_cameras, PPISP_CONTROLLER_FEATURE_LEN), dtype=wp.float32, device="cuda:0")
        controller_params = wp.empty((num_cameras, PPISP_CONTROLLER_PARAM_COUNT), dtype=wp.float32, device="cuda:0")

        compute_ppisp_controller_params(
            hdr_color,
            weights,
            features,
            controller_params,
            prior_exposure,
            responsivity,
        )
        wp.synchronize()

        np.testing.assert_allclose(
            controller_params.numpy(),
            _controller_reference_params(hdr_np, weights_np, prior_exposure, responsivity),
            rtol=2.0e-4,
            atol=2.0e-5,
        )


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
