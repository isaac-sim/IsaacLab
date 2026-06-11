# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for PPISP post-processing."""

from __future__ import annotations

import warp as wp

from .cfg import PpispCfg

wp.init()


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
