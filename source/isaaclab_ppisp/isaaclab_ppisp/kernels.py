# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for PPISP post-processing."""

from __future__ import annotations

from string import Template

import warp as wp

from .cfg import PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN, PpispCfg

wp.init()

PPISP_CONTROLLER_INPUT_DOWNSAMPLING = 3
PPISP_CONTROLLER_CNN_FEATURE_DIM = 64
PPISP_CONTROLLER_CNN_FEATURE_CHUNK = 16
PPISP_CONTROLLER_POOL_GRID_H = 5
PPISP_CONTROLLER_POOL_GRID_W = 5
PPISP_CONTROLLER_POOL_CELL_COUNT = PPISP_CONTROLLER_POOL_GRID_H * PPISP_CONTROLLER_POOL_GRID_W
PPISP_CONTROLLER_FEATURE_LEN = PPISP_CONTROLLER_CNN_FEATURE_DIM * PPISP_CONTROLLER_POOL_CELL_COUNT
PPISP_CONTROLLER_MLP_INPUT_DIM = PPISP_CONTROLLER_FEATURE_LEN + 1
PPISP_CONTROLLER_HIDDEN_DIM = 128
PPISP_CONTROLLER_PARAM_COUNT = 9
PPISP_CONTROLLER_POOL_THREAD_GROUP_SIZE = 256
PPISP_CONTROLLER_MLP_THREAD_GROUP_SIZE = 128

# This layout mirrors NRE's exported PPISP controller weight tensor layout.
# Linear/conv matrices are flattened as row-major ``[out_channel, in_channel]``
# slices.
PPISP_CONTROLLER_OFF_CONV1_W = 0
PPISP_CONTROLLER_OFF_CONV1_B = PPISP_CONTROLLER_OFF_CONV1_W + 16 * 3
PPISP_CONTROLLER_OFF_CONV2_W = PPISP_CONTROLLER_OFF_CONV1_B + 16
PPISP_CONTROLLER_OFF_CONV2_B = PPISP_CONTROLLER_OFF_CONV2_W + 32 * 16
PPISP_CONTROLLER_OFF_CONV3_W = PPISP_CONTROLLER_OFF_CONV2_B + 32
PPISP_CONTROLLER_OFF_CONV3_B = PPISP_CONTROLLER_OFF_CONV3_W + 64 * 32
PPISP_CONTROLLER_OFF_TRUNK0_W = PPISP_CONTROLLER_OFF_CONV3_B + 64
PPISP_CONTROLLER_OFF_TRUNK0_B = (
    PPISP_CONTROLLER_OFF_TRUNK0_W + PPISP_CONTROLLER_HIDDEN_DIM * PPISP_CONTROLLER_MLP_INPUT_DIM
)
PPISP_CONTROLLER_OFF_TRUNK1_W = PPISP_CONTROLLER_OFF_TRUNK0_B + PPISP_CONTROLLER_HIDDEN_DIM
PPISP_CONTROLLER_OFF_TRUNK1_B = (
    PPISP_CONTROLLER_OFF_TRUNK1_W + PPISP_CONTROLLER_HIDDEN_DIM * PPISP_CONTROLLER_HIDDEN_DIM
)
PPISP_CONTROLLER_OFF_TRUNK2_W = PPISP_CONTROLLER_OFF_TRUNK1_B + PPISP_CONTROLLER_HIDDEN_DIM
PPISP_CONTROLLER_OFF_TRUNK2_B = (
    PPISP_CONTROLLER_OFF_TRUNK2_W + PPISP_CONTROLLER_HIDDEN_DIM * PPISP_CONTROLLER_HIDDEN_DIM
)
PPISP_CONTROLLER_OFF_EXP_W = PPISP_CONTROLLER_OFF_TRUNK2_B + PPISP_CONTROLLER_HIDDEN_DIM
PPISP_CONTROLLER_OFF_EXP_B = PPISP_CONTROLLER_OFF_EXP_W + PPISP_CONTROLLER_HIDDEN_DIM
PPISP_CONTROLLER_OFF_COL_W = PPISP_CONTROLLER_OFF_EXP_B + 1
PPISP_CONTROLLER_OFF_COL_B = PPISP_CONTROLLER_OFF_COL_W + 8 * PPISP_CONTROLLER_HIDDEN_DIM
PPISP_CONTROLLER_TOTAL_WEIGHTS = PPISP_CONTROLLER_OFF_COL_B + 8
if PPISP_CONTROLLER_TOTAL_WEIGHTS != PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN:
    raise RuntimeError(
        "PPISP controller weight offsets do not match the expected exported weight count: "
        f"{PPISP_CONTROLLER_TOTAL_WEIGHTS} != {PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN}."
    )


@wp.func
def _bounded_softplus(raw: wp.float32, min_value: wp.float32):
    """Map an unconstrained parameter to a positive value with a lower bound.

    The PPISP CRF stores toe/shoulder/gamma as raw optimization parameters. This
    helper applies ``min + log(1 + exp(raw))`` so the resulting shape parameters
    stay positive and numerically away from degenerate zero values.
    """
    return min_value + wp.log(1.0 + wp.exp(raw))


@wp.func
def _sigmoid(raw: wp.float32):
    return 1.0 / (1.0 + wp.exp(0.0 - raw))


@wp.func
def _apply_vignetting(
    value: wp.float32,
    uv: wp.vec2f,
    optical_center: wp.vec2f,
    alpha1: wp.float32,
    alpha2: wp.float32,
    alpha3: wp.float32,
):
    """Apply per-channel radial vignetting in local normalized image coordinates.

    The pixel coordinate ``uv`` is centered on the current de-tiled camera image
    and normalized by ``max(width, height)``. The falloff is the clamped radial
    polynomial ``1 + a1 r^2 + a2 r^4 + a3 r^6``, where
    ``r^2 = dot(uv - optical_center, uv - optical_center)``.
    """
    delta = uv - optical_center
    radius_squared = wp.dot(delta, delta)
    radius_power = radius_squared
    falloff = wp.float32(1.0) + alpha1 * radius_power
    radius_power = radius_power * radius_squared
    falloff = falloff + alpha2 * radius_power
    radius_power = radius_power * radius_squared
    falloff = falloff + alpha3 * radius_power
    return value * wp.clamp(falloff, 0.0, 1.0)


@wp.func
def _apply_crf(
    value: wp.float32,
    toe_raw: wp.float32,
    shoulder_raw: wp.float32,
    gamma_raw: wp.float32,
    center_raw: wp.float32,
):
    """Apply one channel of the PPISP camera response function.

    The CRF is a piecewise power curve split around ``center = sigmoid(raw)``.
    Toe, shoulder, and gamma are bounded-softplus parameters. Values below the
    center use the toe exponent; values above it use the shoulder exponent; the
    result is then raised to ``gamma``. The center is clamped away from 0 and 1
    because it appears in the denominator of both curve segments.
    """
    x = wp.clamp(value, 0.0, 1.0)
    toe = _bounded_softplus(toe_raw, 0.3)
    shoulder = _bounded_softplus(shoulder_raw, 0.3)
    gamma = _bounded_softplus(gamma_raw, 0.1)
    center = wp.clamp(_sigmoid(center_raw), 1.0e-6, 1.0 - 1.0e-6)

    lerp_value = (shoulder - toe) * center + toe
    a = (shoulder * center) / lerp_value
    b = 1.0 - a

    y = wp.float32(0.0)
    if x <= center:
        y = a * wp.pow(x / center, toe)
    else:
        y = 1.0 - b * wp.pow((1.0 - x) / (1.0 - center), shoulder)

    return wp.pow(wp.max(0.0, y), gamma)


@wp.func
def _compute_homography_mul(
    rgb: wp.vec3f,
    blue_latent: wp.vec2f,
    red_latent: wp.vec2f,
    green_latent: wp.vec2f,
    neutral_latent: wp.vec2f,
):
    """Apply the PPISP color correction homography.

    The four 2D latent controls perturb the blue, red, green, and neutral
    chromaticity anchors. A projective transform is solved from those anchors
    and applied in ``r,g,intensity`` space, preserving the input pixel intensity
    after the chromaticity remap.
    """
    blue_delta = wp.vec2f(
        0.0480542 * blue_latent[0] - 0.0043631 * blue_latent[1],
        -0.0043631 * blue_latent[0] + 0.0481283 * blue_latent[1],
    )
    red_delta = wp.vec2f(
        0.0580570 * red_latent[0] - 0.0179872 * red_latent[1],
        -0.0179872 * red_latent[0] + 0.0431061 * red_latent[1],
    )
    green_delta = wp.vec2f(
        0.0433336 * green_latent[0] - 0.0180537 * green_latent[1],
        -0.0180537 * green_latent[0] + 0.0580500 * green_latent[1],
    )
    neutral_delta = wp.vec2f(
        0.0128369 * neutral_latent[0] - 0.0034654 * neutral_latent[1],
        -0.0034654 * neutral_latent[0] + 0.0128158 * neutral_latent[1],
    )

    target_blue = wp.vec3f(blue_delta[0], blue_delta[1], 1.0)
    target_red = wp.vec3f(1.0 + red_delta[0], red_delta[1], 1.0)
    target_green = wp.vec3f(green_delta[0], 1.0 + green_delta[1], 1.0)
    target_gray = wp.vec3f(1.0 / 3.0 + neutral_delta[0], 1.0 / 3.0 + neutral_delta[1], 1.0)

    row0 = wp.vec3f(target_gray[1] - target_blue[1], target_gray[1] - target_red[1], target_gray[1] - target_green[1])
    row1 = wp.vec3f(target_blue[0] - target_gray[0], target_red[0] - target_gray[0], target_green[0] - target_gray[0])
    row2 = wp.vec3f(
        -target_gray[1] * target_blue[0] + target_gray[0] * target_blue[1],
        -target_gray[1] * target_red[0] + target_gray[0] * target_red[1],
        -target_gray[1] * target_green[0] + target_gray[0] * target_green[1],
    )

    lam = wp.cross(row0, row1)
    if wp.dot(lam, lam) < 1.0e-20:
        lam = wp.cross(row0, row2)
        if wp.dot(lam, lam) < 1.0e-20:
            lam = wp.cross(row1, row2)

    col0 = -target_blue * lam[0] + target_red * lam[1]
    col1 = -target_blue * lam[0] + target_green * lam[2]
    col2 = target_blue * lam[0]

    h22 = col0[2] + col1[2] + col2[2]
    if wp.abs(h22) > 1.0e-20:
        inv_h22 = 1.0 / h22
        col0 = col0 * inv_h22
        col1 = col1 * inv_h22
        col2 = col2 * inv_h22

    intensity = rgb[0] + rgb[1] + rgb[2]
    rgi = col0 * rgb[0] + col1 * rgb[1] + col2 * intensity
    rgi = rgi * (intensity / (rgi[2] + 1.0e-5))
    return wp.vec3f(rgi[0], rgi[1], rgi[2] - rgi[0] - rgi[1])


_PPISP_CONTROLLER_POOL_NATIVE_SNIPPET = Template(r"""
    __shared__ float gs_reduce[$pool_thread_group_size];

    const int input_downsampling = $input_downsampling;
    const int cnn_feature_dim = $cnn_feature_dim;
    const int cnn_feature_chunk = $cnn_feature_chunk;
    const int pool_grid_h = $pool_grid_h;
    const int pool_grid_w = $pool_grid_w;
    const int pool_cell_count = $pool_cell_count;
    const int pool_thread_group_size = $pool_thread_group_size;
    const int off_conv1_w = $off_conv1_w;
    const int off_conv1_b = $off_conv1_b;
    const int off_conv2_w = $off_conv2_w;
    const int off_conv2_b = $off_conv2_b;
    const int off_conv3_w = $off_conv3_w;
    const int off_conv3_b = $off_conv3_b;

    int ds_width = image_width / input_downsampling;
    int ds_height = image_height / input_downsampling;
    if (ds_width < 1) {
        ds_width = 1;
    }
    if (ds_height < 1) {
        ds_height = 1;
    }

    const int gy = cell / pool_grid_w;
    const int gx = cell - gy * pool_grid_w;

    const int h_start = (gy * ds_height) / pool_grid_h;
    int h_end = ((gy + 1) * ds_height + pool_grid_h - 1) / pool_grid_h;
    const int w_start = (gx * ds_width) / pool_grid_w;
    int w_end = ((gx + 1) * ds_width + pool_grid_w - 1) / pool_grid_w;
    if (h_end > ds_height) {
        h_end = ds_height;
    }
    if (w_end > ds_width) {
        w_end = ds_width;
    }

    const int cell_width = (w_end > w_start) ? (w_end - w_start) : 0;
    const int cell_height = (h_end > h_start) ? (h_end - h_start) : 0;
    const int count = cell_width * cell_height;

    for (int first_channel = 0; first_channel < cnn_feature_dim; first_channel += cnn_feature_chunk) {
        float partial[16];
#pragma unroll
        for (int c = 0; c < cnn_feature_chunk; ++c) {
            partial[c] = 0.0f;
        }

        for (int idx = lane; idx < count; idx += pool_thread_group_size) {
            const int dy = h_start + idx / cell_width;
            const int dx = w_start + idx - (idx / cell_width) * cell_width;
            const int x0 = dx * input_downsampling;
            const int y0 = dy * input_downsampling;
            int x1 = x0 + input_downsampling;
            int y1 = y0 + input_downsampling;
            if (x1 > image_width) {
                x1 = image_width;
            }
            if (y1 > image_height) {
                y1 = image_height;
            }

            float pooled[16];
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                pooled[c] = -3.4028234663852886e38f;
            }

            for (int yy = y0; yy < y1; ++yy) {
                for (int xx = x0; xx < x1; ++xx) {
                    const float r = (*wp::address(hdr_color, camera_id, yy, xx, 0)) * responsivity;
                    const float g = (*wp::address(hdr_color, camera_id, yy, xx, 1)) * responsivity;
                    const float b = (*wp::address(hdr_color, camera_id, yy, xx, 2)) * responsivity;

#pragma unroll
                    for (int o = 0; o < 16; ++o) {
                        float v = *wp::address(weights, off_conv1_b + o);
                        v += r * (*wp::address(weights, off_conv1_w + o * 3 + 0));
                        v += g * (*wp::address(weights, off_conv1_w + o * 3 + 1));
                        v += b * (*wp::address(weights, off_conv1_w + o * 3 + 2));
                        pooled[o] = fmaxf(pooled[o], v);
                    }
                }
            }

#pragma unroll
            for (int c = 0; c < 16; ++c) {
                pooled[c] = fmaxf(0.0f, pooled[c]);
            }

            float conv2[32];
#pragma unroll
            for (int o = 0; o < 32; ++o) {
                float v = *wp::address(weights, off_conv2_b + o);
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    v += pooled[i] * (*wp::address(weights, off_conv2_w + o * 16 + i));
                }
                conv2[o] = fmaxf(0.0f, v);
            }

#pragma unroll
            for (int c = 0; c < cnn_feature_chunk; ++c) {
                const int o = first_channel + c;
                float v = *wp::address(weights, off_conv3_b + o);
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    v += conv2[i] * (*wp::address(weights, off_conv3_w + o * 32 + i));
                }
                partial[c] += v;
            }
        }

#pragma unroll
        for (int c = 0; c < cnn_feature_chunk; ++c) {
            gs_reduce[lane] = partial[c];
            __syncthreads();

            for (int stride = pool_thread_group_size / 2; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    gs_reduce[lane] += gs_reduce[lane + stride];
                }
                __syncthreads();
            }

            if (lane == 0) {
                const int channel = first_channel + c;
                // Channel-major layout matches the controller trunk0 feature ordering.
                const int feature_id = channel * pool_cell_count + cell;
                const float inv_count = (count > 0) ? (1.0f / ((float)count)) : 0.0f;
                *wp::address(features, camera_id, feature_id) = gs_reduce[0] * inv_count;
            }
            __syncthreads();
        }
    }
""").substitute(
    {
        "input_downsampling": PPISP_CONTROLLER_INPUT_DOWNSAMPLING,
        "cnn_feature_dim": PPISP_CONTROLLER_CNN_FEATURE_DIM,
        "cnn_feature_chunk": PPISP_CONTROLLER_CNN_FEATURE_CHUNK,
        "pool_grid_h": PPISP_CONTROLLER_POOL_GRID_H,
        "pool_grid_w": PPISP_CONTROLLER_POOL_GRID_W,
        "pool_cell_count": PPISP_CONTROLLER_POOL_CELL_COUNT,
        "pool_thread_group_size": PPISP_CONTROLLER_POOL_THREAD_GROUP_SIZE,
        "off_conv1_w": PPISP_CONTROLLER_OFF_CONV1_W,
        "off_conv1_b": PPISP_CONTROLLER_OFF_CONV1_B,
        "off_conv2_w": PPISP_CONTROLLER_OFF_CONV2_W,
        "off_conv2_b": PPISP_CONTROLLER_OFF_CONV2_B,
        "off_conv3_w": PPISP_CONTROLLER_OFF_CONV3_W,
        "off_conv3_b": PPISP_CONTROLLER_OFF_CONV3_B,
    }
)


@wp.func_native(_PPISP_CONTROLLER_POOL_NATIVE_SNIPPET)
def _ppisp_controller_pool_features_native(
    hdr_color: wp.array4d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    features: wp.array2d(dtype=wp.float32),
    image_width: wp.int32,
    image_height: wp.int32,
    responsivity: wp.float32,
    camera_id: wp.int32,
    cell: wp.int32,
    lane: wp.int32,
): ...


@wp.kernel(enable_backward=False)
def _ppisp_controller_pool_features_native_kernel(
    hdr_color: wp.array4d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    features: wp.array2d(dtype=wp.float32),
    image_width: wp.int32,
    image_height: wp.int32,
    responsivity: wp.float32,
):
    """Run the controller CNN and adaptive pool using a native CUDA snippet."""
    camera_id, cell, lane = wp.tid()
    _ppisp_controller_pool_features_native(
        hdr_color,
        weights,
        features,
        image_width,
        image_height,
        responsivity,
        camera_id,
        cell,
        lane,
    )


_PPISP_CONTROLLER_MLP_NATIVE_SNIPPET = Template(r"""
    __shared__ float hidden_a[$hidden_dim];
    __shared__ float hidden_b[$hidden_dim];

    const int pool_feature_len = $feature_len;
    const int mlp_input_dim = $mlp_input_dim;
    const int mlp_hidden_dim = $hidden_dim;
    const int color_params_per_frame = 8;
    const int mlp_thread_group_size = $mlp_thread_group_size;
    const int off_trunk0_w = $off_trunk0_w;
    const int off_trunk0_b = $off_trunk0_b;
    const int off_trunk1_w = $off_trunk1_w;
    const int off_trunk1_b = $off_trunk1_b;
    const int off_trunk2_w = $off_trunk2_w;
    const int off_trunk2_b = $off_trunk2_b;
    const int off_exp_w = $off_exp_w;
    const int off_exp_b = $off_exp_b;
    const int off_col_w = $off_col_w;
    const int off_col_b = $off_col_b;

    for (int o = lane; o < mlp_hidden_dim; o += mlp_thread_group_size) {
        float v = *wp::address(weights, off_trunk0_b + o);
        for (int i = 0; i < pool_feature_len; ++i) {
            v += (*wp::address(features, camera_id, i)) * (*wp::address(weights, off_trunk0_w + o * mlp_input_dim + i));
        }
        v += prior_exposure * (*wp::address(weights, off_trunk0_w + o * mlp_input_dim + pool_feature_len));
        hidden_a[o] = fmaxf(0.0f, v);
    }
    __syncthreads();

    for (int o = lane; o < mlp_hidden_dim; o += mlp_thread_group_size) {
        float v = *wp::address(weights, off_trunk1_b + o);
        for (int i = 0; i < mlp_hidden_dim; ++i) {
            v += hidden_a[i] * (*wp::address(weights, off_trunk1_w + o * mlp_hidden_dim + i));
        }
        hidden_b[o] = fmaxf(0.0f, v);
    }
    __syncthreads();

    for (int o = lane; o < mlp_hidden_dim; o += mlp_thread_group_size) {
        float v = *wp::address(weights, off_trunk2_b + o);
        for (int i = 0; i < mlp_hidden_dim; ++i) {
            v += hidden_b[i] * (*wp::address(weights, off_trunk2_w + o * mlp_hidden_dim + i));
        }
        hidden_a[o] = fmaxf(0.0f, v);
    }
    __syncthreads();

    if (lane == 0) {
        float v = *wp::address(weights, off_exp_b);
        for (int i = 0; i < mlp_hidden_dim; ++i) {
            v += hidden_a[i] * (*wp::address(weights, off_exp_w + i));
        }
        *wp::address(controller_params, camera_id, 0) = v;
    }
    if (lane < color_params_per_frame) {
        const int o = lane;
        float v = *wp::address(weights, off_col_b + o);
        for (int i = 0; i < mlp_hidden_dim; ++i) {
            v += hidden_a[i] * (*wp::address(weights, off_col_w + o * mlp_hidden_dim + i));
        }
        *wp::address(controller_params, camera_id, 1 + o) = v;
    }
""").substitute(
    {
        "feature_len": PPISP_CONTROLLER_FEATURE_LEN,
        "mlp_input_dim": PPISP_CONTROLLER_MLP_INPUT_DIM,
        "hidden_dim": PPISP_CONTROLLER_HIDDEN_DIM,
        "mlp_thread_group_size": PPISP_CONTROLLER_MLP_THREAD_GROUP_SIZE,
        "off_trunk0_w": PPISP_CONTROLLER_OFF_TRUNK0_W,
        "off_trunk0_b": PPISP_CONTROLLER_OFF_TRUNK0_B,
        "off_trunk1_w": PPISP_CONTROLLER_OFF_TRUNK1_W,
        "off_trunk1_b": PPISP_CONTROLLER_OFF_TRUNK1_B,
        "off_trunk2_w": PPISP_CONTROLLER_OFF_TRUNK2_W,
        "off_trunk2_b": PPISP_CONTROLLER_OFF_TRUNK2_B,
        "off_exp_w": PPISP_CONTROLLER_OFF_EXP_W,
        "off_exp_b": PPISP_CONTROLLER_OFF_EXP_B,
        "off_col_w": PPISP_CONTROLLER_OFF_COL_W,
        "off_col_b": PPISP_CONTROLLER_OFF_COL_B,
    }
)


@wp.func_native(_PPISP_CONTROLLER_MLP_NATIVE_SNIPPET)
def _ppisp_controller_mlp_native(
    features: wp.array2d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    controller_params: wp.array2d(dtype=wp.float32),
    prior_exposure: wp.float32,
    camera_id: wp.int32,
    lane: wp.int32,
): ...


@wp.kernel(enable_backward=False)
def _ppisp_controller_mlp_native_kernel(
    features: wp.array2d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    controller_params: wp.array2d(dtype=wp.float32),
    prior_exposure: wp.float32,
):
    """Run the controller MLP using a native CUDA snippet."""
    camera_id, lane = wp.tid()
    _ppisp_controller_mlp_native(features, weights, controller_params, prior_exposure, camera_id, lane)


@wp.kernel(enable_backward=False)
def _apply_ppisp_kernel(
    hdr_color: wp.array4d(dtype=wp.float32),
    out_rgba: wp.array4d(dtype=wp.uint8),
    image_width: wp.int32,
    image_height: wp.int32,
    responsivity: wp.float32,
    exposure_offset: wp.float32,
    vignetting_center_r: wp.vec2f,
    vignetting_alpha1_r: wp.float32,
    vignetting_alpha2_r: wp.float32,
    vignetting_alpha3_r: wp.float32,
    vignetting_center_g: wp.vec2f,
    vignetting_alpha1_g: wp.float32,
    vignetting_alpha2_g: wp.float32,
    vignetting_alpha3_g: wp.float32,
    vignetting_center_b: wp.vec2f,
    vignetting_alpha1_b: wp.float32,
    vignetting_alpha2_b: wp.float32,
    vignetting_alpha3_b: wp.float32,
    color_latent_blue: wp.vec2f,
    color_latent_red: wp.vec2f,
    color_latent_green: wp.vec2f,
    color_latent_neutral: wp.vec2f,
    crf_toe_r: wp.float32,
    crf_shoulder_r: wp.float32,
    crf_gamma_r: wp.float32,
    crf_center_r: wp.float32,
    crf_toe_g: wp.float32,
    crf_shoulder_g: wp.float32,
    crf_gamma_g: wp.float32,
    crf_center_g: wp.float32,
    crf_toe_b: wp.float32,
    crf_shoulder_b: wp.float32,
    crf_gamma_b: wp.float32,
    crf_center_b: wp.float32,
):
    """Apply the camera PPISP pipeline to one de-tiled HDR color tensor.

    For each pixel, the model applies HDR responsivity scaling, exposure
    scaling, per-channel vignetting, color homography correction, CRF tone
    mapping, and uint8 RGBA packing. The first tensor dimension is the
    camera/environment index, so image-space effects use local ``height_id``
    and ``width_id`` coordinates per camera.
    """
    camera_id, height_id, width_id = wp.tid()
    rgb = wp.vec3f(
        hdr_color[camera_id, height_id, width_id, 0],
        hdr_color[camera_id, height_id, width_id, 1],
        hdr_color[camera_id, height_id, width_id, 2],
    )
    max_resolution = wp.float32(image_width)
    if image_height > image_width:
        max_resolution = wp.float32(image_height)
    uv = wp.vec2f(
        (wp.float32(width_id) + 0.5 - wp.float32(image_width) * 0.5) / max_resolution,
        (wp.float32(height_id) + 0.5 - wp.float32(image_height) * 0.5) / max_resolution,
    )

    # 0. HDR responsivity (achromatic, applied pre-PPISP) — orthogonal to the
    # asset-baked radiance scaling on Gaussian SH coefficients. Default 1.0 = no-op.
    out_rgb = rgb * responsivity
    # 1. Exposure
    out_rgb = out_rgb * wp.pow(2.0, exposure_offset)
    out_rgb[0] = _apply_vignetting(
        out_rgb[0], uv, vignetting_center_r, vignetting_alpha1_r, vignetting_alpha2_r, vignetting_alpha3_r
    )
    out_rgb[1] = _apply_vignetting(
        out_rgb[1], uv, vignetting_center_g, vignetting_alpha1_g, vignetting_alpha2_g, vignetting_alpha3_g
    )
    out_rgb[2] = _apply_vignetting(
        out_rgb[2], uv, vignetting_center_b, vignetting_alpha1_b, vignetting_alpha2_b, vignetting_alpha3_b
    )
    out_rgb = _compute_homography_mul(
        out_rgb, color_latent_blue, color_latent_red, color_latent_green, color_latent_neutral
    )
    out_rgb[0] = _apply_crf(out_rgb[0], crf_toe_r, crf_shoulder_r, crf_gamma_r, crf_center_r)
    out_rgb[1] = _apply_crf(out_rgb[1], crf_toe_g, crf_shoulder_g, crf_gamma_g, crf_center_g)
    out_rgb[2] = _apply_crf(out_rgb[2], crf_toe_b, crf_shoulder_b, crf_gamma_b, crf_center_b)

    out_rgba[camera_id, height_id, width_id, 0] = wp.uint8(wp.clamp(out_rgb[0], 0.0, 1.0) * 255.0)
    out_rgba[camera_id, height_id, width_id, 1] = wp.uint8(wp.clamp(out_rgb[1], 0.0, 1.0) * 255.0)
    out_rgba[camera_id, height_id, width_id, 2] = wp.uint8(wp.clamp(out_rgb[2], 0.0, 1.0) * 255.0)
    out_rgba[camera_id, height_id, width_id, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def _apply_ppisp_controller_kernel(
    hdr_color: wp.array4d(dtype=wp.float32),
    out_rgba: wp.array4d(dtype=wp.uint8),
    controller_params: wp.array2d(dtype=wp.float32),
    image_width: wp.int32,
    image_height: wp.int32,
    responsivity: wp.float32,
    vignetting_center_r: wp.vec2f,
    vignetting_alpha1_r: wp.float32,
    vignetting_alpha2_r: wp.float32,
    vignetting_alpha3_r: wp.float32,
    vignetting_center_g: wp.vec2f,
    vignetting_alpha1_g: wp.float32,
    vignetting_alpha2_g: wp.float32,
    vignetting_alpha3_g: wp.float32,
    vignetting_center_b: wp.vec2f,
    vignetting_alpha1_b: wp.float32,
    vignetting_alpha2_b: wp.float32,
    vignetting_alpha3_b: wp.float32,
    crf_toe_r: wp.float32,
    crf_shoulder_r: wp.float32,
    crf_gamma_r: wp.float32,
    crf_center_r: wp.float32,
    crf_toe_g: wp.float32,
    crf_shoulder_g: wp.float32,
    crf_gamma_g: wp.float32,
    crf_center_g: wp.float32,
    crf_toe_b: wp.float32,
    crf_shoulder_b: wp.float32,
    crf_gamma_b: wp.float32,
    crf_center_b: wp.float32,
):
    """Apply PPISP using per-camera exposure/color parameters predicted by the controller."""
    camera_id, height_id, width_id = wp.tid()
    rgb = wp.vec3f(
        hdr_color[camera_id, height_id, width_id, 0],
        hdr_color[camera_id, height_id, width_id, 1],
        hdr_color[camera_id, height_id, width_id, 2],
    )
    max_resolution = wp.float32(image_width)
    if image_height > image_width:
        max_resolution = wp.float32(image_height)
    uv = wp.vec2f(
        (wp.float32(width_id) + 0.5 - wp.float32(image_width) * 0.5) / max_resolution,
        (wp.float32(height_id) + 0.5 - wp.float32(image_height) * 0.5) / max_resolution,
    )

    # Exported controller params are:
    # [exposure, blue.x, blue.y, red.x, red.y, green.x, green.y, neutral.x, neutral.y].
    color_latent_blue = wp.vec2f(controller_params[camera_id, 1], controller_params[camera_id, 2])
    color_latent_red = wp.vec2f(controller_params[camera_id, 3], controller_params[camera_id, 4])
    color_latent_green = wp.vec2f(controller_params[camera_id, 5], controller_params[camera_id, 6])
    color_latent_neutral = wp.vec2f(controller_params[camera_id, 7], controller_params[camera_id, 8])

    out_rgb = rgb * responsivity
    out_rgb = out_rgb * wp.pow(2.0, controller_params[camera_id, 0])
    out_rgb[0] = _apply_vignetting(
        out_rgb[0], uv, vignetting_center_r, vignetting_alpha1_r, vignetting_alpha2_r, vignetting_alpha3_r
    )
    out_rgb[1] = _apply_vignetting(
        out_rgb[1], uv, vignetting_center_g, vignetting_alpha1_g, vignetting_alpha2_g, vignetting_alpha3_g
    )
    out_rgb[2] = _apply_vignetting(
        out_rgb[2], uv, vignetting_center_b, vignetting_alpha1_b, vignetting_alpha2_b, vignetting_alpha3_b
    )
    out_rgb = _compute_homography_mul(
        out_rgb, color_latent_blue, color_latent_red, color_latent_green, color_latent_neutral
    )
    out_rgb[0] = _apply_crf(out_rgb[0], crf_toe_r, crf_shoulder_r, crf_gamma_r, crf_center_r)
    out_rgb[1] = _apply_crf(out_rgb[1], crf_toe_g, crf_shoulder_g, crf_gamma_g, crf_center_g)
    out_rgb[2] = _apply_crf(out_rgb[2], crf_toe_b, crf_shoulder_b, crf_gamma_b, crf_center_b)

    out_rgba[camera_id, height_id, width_id, 0] = wp.uint8(wp.clamp(out_rgb[0], 0.0, 1.0) * 255.0)
    out_rgba[camera_id, height_id, width_id, 1] = wp.uint8(wp.clamp(out_rgb[1], 0.0, 1.0) * 255.0)
    out_rgba[camera_id, height_id, width_id, 2] = wp.uint8(wp.clamp(out_rgb[2], 0.0, 1.0) * 255.0)
    out_rgba[camera_id, height_id, width_id, 3] = wp.uint8(255)


def apply_ppisp_to_rgba(hdr_color: wp.array, out_rgba: wp.array, cfg: PpispCfg) -> None:
    """Apply PPISP to ``hdr_color`` and write LDR RGBA into ``out_rgba``.

    Args:
        hdr_color: HDR scene-linear input. ``wp.array4d(dtype=wp.float32)`` of
            shape ``(N, H, W, 3)``.
        out_rgba: LDR RGBA output. ``wp.array4d(dtype=wp.uint8)`` of shape
            ``(N, H, W, 4)``. Written in place.
        cfg: PPISP configuration.
    """
    if hdr_color.dtype is not wp.float32:
        raise ValueError(f"Camera PPISP HDR input must be wp.float32, got {hdr_color.dtype}.")
    if out_rgba.dtype is not wp.uint8:
        raise ValueError(f"Camera PPISP RGBA output must be wp.uint8, got {out_rgba.dtype}.")

    inputs = cfg.inputs
    wp.launch(
        _apply_ppisp_kernel,
        dim=out_rgba.shape[:3],
        inputs=[
            hdr_color,
            out_rgba,
            int(out_rgba.shape[2]),
            int(out_rgba.shape[1]),
            float(inputs.get("responsivity", 1.0)),
            float(inputs["exposureOffset"]),
            wp.vec2f(*inputs["vignettingCenterR"]),
            float(inputs["vignettingAlpha1R"]),
            float(inputs["vignettingAlpha2R"]),
            float(inputs["vignettingAlpha3R"]),
            wp.vec2f(*inputs["vignettingCenterG"]),
            float(inputs["vignettingAlpha1G"]),
            float(inputs["vignettingAlpha2G"]),
            float(inputs["vignettingAlpha3G"]),
            wp.vec2f(*inputs["vignettingCenterB"]),
            float(inputs["vignettingAlpha1B"]),
            float(inputs["vignettingAlpha2B"]),
            float(inputs["vignettingAlpha3B"]),
            wp.vec2f(*inputs["colorLatentBlue"]),
            wp.vec2f(*inputs["colorLatentRed"]),
            wp.vec2f(*inputs["colorLatentGreen"]),
            wp.vec2f(*inputs["colorLatentNeutral"]),
            float(inputs["crfToeR"]),
            float(inputs["crfShoulderR"]),
            float(inputs["crfGammaR"]),
            float(inputs["crfCenterR"]),
            float(inputs["crfToeG"]),
            float(inputs["crfShoulderG"]),
            float(inputs["crfGammaG"]),
            float(inputs["crfCenterG"]),
            float(inputs["crfToeB"]),
            float(inputs["crfShoulderB"]),
            float(inputs["crfGammaB"]),
            float(inputs["crfCenterB"]),
        ],
        device=str(out_rgba.device),
    )


def _validate_ppisp_controller_inputs(
    hdr_color: wp.array,
    controller_weights: wp.array,
    features: wp.array,
    controller_params: wp.array,
) -> None:
    if hdr_color.dtype is not wp.float32:
        raise ValueError(f"Camera PPISP controller HDR input must be wp.float32, got {hdr_color.dtype}.")
    if controller_weights.dtype is not wp.float32:
        raise ValueError(f"Camera PPISP controller weights must be wp.float32, got {controller_weights.dtype}.")
    if controller_weights.shape[0] != PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN:
        raise ValueError(
            "Camera PPISP controller weights must have shape "
            f"({PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN},), got {controller_weights.shape}."
        )
    if features.shape != (hdr_color.shape[0], PPISP_CONTROLLER_FEATURE_LEN):
        raise ValueError(
            "Camera PPISP controller features must have shape "
            f"({hdr_color.shape[0]}, {PPISP_CONTROLLER_FEATURE_LEN}), got {features.shape}."
        )
    if controller_params.shape != (hdr_color.shape[0], PPISP_CONTROLLER_PARAM_COUNT):
        raise ValueError(
            "Camera PPISP controller params must have shape "
            f"({hdr_color.shape[0]}, {PPISP_CONTROLLER_PARAM_COUNT}), got {controller_params.shape}."
        )


def compute_ppisp_controller_params(
    hdr_color: wp.array,
    controller_weights: wp.array,
    features: wp.array,
    controller_params: wp.array,
    prior_exposure: float,
    responsivity: float = 1.0,
) -> None:
    """Run the PPISP controller using native CUDA snippets inside Warp kernels.

    Args:
        hdr_color: HDR scene-linear input, shape ``(N, H, W, 3)``.
        controller_weights: Flattened embedded controller weights.
        features: Scratch buffer, shape ``(N, 1600)``.
        controller_params: Output buffer, shape ``(N, 9)``.
        prior_exposure: Scalar prior exposure supplied by the PPISP config.
        responsivity: Achromatic HDR multiplier applied before controller
            feature extraction.

    Raises:
        ValueError: When ``hdr_color`` is not on a CUDA device.
    """
    _validate_ppisp_controller_inputs(hdr_color, controller_weights, features, controller_params)
    device = wp.get_device(str(hdr_color.device))
    if not device.is_cuda:
        raise ValueError("Camera PPISP controller requires a CUDA device.")

    image_height = int(hdr_color.shape[1])
    image_width = int(hdr_color.shape[2])
    device_name = str(hdr_color.device)
    wp.launch_tiled(
        _ppisp_controller_pool_features_native_kernel,
        dim=(int(hdr_color.shape[0]), PPISP_CONTROLLER_POOL_CELL_COUNT),
        inputs=[hdr_color, controller_weights, features, image_width, image_height, float(responsivity)],
        device=device_name,
        block_dim=PPISP_CONTROLLER_POOL_THREAD_GROUP_SIZE,
    )
    wp.launch_tiled(
        _ppisp_controller_mlp_native_kernel,
        dim=(int(hdr_color.shape[0]),),
        inputs=[features, controller_weights, controller_params, float(prior_exposure)],
        device=device_name,
        block_dim=PPISP_CONTROLLER_MLP_THREAD_GROUP_SIZE,
    )


def apply_ppisp_to_rgba_with_controller_params(
    hdr_color: wp.array, out_rgba: wp.array, cfg: PpispCfg, controller_params: wp.array
) -> None:
    """Apply PPISP with per-camera controller parameters.

    Args:
        hdr_color: HDR scene-linear input, shape ``(N, H, W, 3)``.
        out_rgba: LDR RGBA output, shape ``(N, H, W, 4)``.
        cfg: PPISP configuration. Static inputs provide responsivity,
            vignetting and CRF.
        controller_params: ``wp.array2d(float32)`` of shape ``(N, 9)`` holding
            ``[exposureOffset, colorLatentBlue.xy, colorLatentRed.xy,
            colorLatentGreen.xy, colorLatentNeutral.xy]``.
    """
    if hdr_color.dtype is not wp.float32:
        raise ValueError(f"Camera PPISP HDR input must be wp.float32, got {hdr_color.dtype}.")
    if out_rgba.dtype is not wp.uint8:
        raise ValueError(f"Camera PPISP RGBA output must be wp.uint8, got {out_rgba.dtype}.")
    if controller_params.dtype is not wp.float32:
        raise ValueError(f"Camera PPISP controller params must be wp.float32, got {controller_params.dtype}.")
    if controller_params.shape[0] != out_rgba.shape[0] or controller_params.shape[1] != 9:
        raise ValueError(
            f"Camera PPISP controller params must have shape ({out_rgba.shape[0]}, 9), got {controller_params.shape}."
        )

    inputs = cfg.inputs
    wp.launch(
        _apply_ppisp_controller_kernel,
        dim=out_rgba.shape[:3],
        inputs=[
            hdr_color,
            out_rgba,
            controller_params,
            int(out_rgba.shape[2]),
            int(out_rgba.shape[1]),
            float(inputs.get("responsivity", 1.0)),
            wp.vec2f(*inputs["vignettingCenterR"]),
            float(inputs["vignettingAlpha1R"]),
            float(inputs["vignettingAlpha2R"]),
            float(inputs["vignettingAlpha3R"]),
            wp.vec2f(*inputs["vignettingCenterG"]),
            float(inputs["vignettingAlpha1G"]),
            float(inputs["vignettingAlpha2G"]),
            float(inputs["vignettingAlpha3G"]),
            wp.vec2f(*inputs["vignettingCenterB"]),
            float(inputs["vignettingAlpha1B"]),
            float(inputs["vignettingAlpha2B"]),
            float(inputs["vignettingAlpha3B"]),
            float(inputs["crfToeR"]),
            float(inputs["crfShoulderR"]),
            float(inputs["crfGammaR"]),
            float(inputs["crfCenterR"]),
            float(inputs["crfToeG"]),
            float(inputs["crfShoulderG"]),
            float(inputs["crfGammaG"]),
            float(inputs["crfCenterG"]),
            float(inputs["crfToeB"]),
            float(inputs["crfShoulderB"]),
            float(inputs["crfGammaB"]),
            float(inputs["crfCenterB"]),
        ],
        device=str(out_rgba.device),
    )
