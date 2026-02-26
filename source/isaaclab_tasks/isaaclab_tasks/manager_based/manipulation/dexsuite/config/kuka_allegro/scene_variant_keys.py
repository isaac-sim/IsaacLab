# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene variant key parsing and key set (stdlib only, no isaaclab).

Use this module to test parse_scene_key() and variant keys without importing
isaaclab/Isaac Sim (e.g. plain python -c). The vision env config imports from here
to build actual scene configs.
"""

import re

# Convention: <width>x<height><renderer_tag>_<camera_tag> (TiledCameraCfg used regardless; rtx/warp = backend)
SCENE_KEY_PATTERN = re.compile(r"^(\d+)x(\d+)(rtx|warp)_(rgb|depth|albedo)$")
# Neutral keys: <width>x<height><camera_tag> (renderer set via env.scene.base_camera.renderer_type=...)
NEUTRAL_SCENE_KEY_PATTERN = re.compile(r"^(\d+)x(\d+)(rgb|depth|albedo)$")
RENDERER_TAG_TO_TYPE = {"rtx": "isaac_rtx", "warp": "newton_warp"}
CAMERA_TAG_TO_TYPE = {"rgb": "rgb", "depth": "distance_to_image_plane", "albedo": "diffuse_albedo"}

RESOLUTIONS = ((64, 64), (128, 128), (256, 256))
RENDERER_CAMERA_COMBO = (
    ("rtx", "depth"),
    ("rtx", "rgb"),
    ("rtx", "albedo"),
    ("warp", "depth"),
    ("warp", "rgb"),
    ("warp", "albedo"),
)
# For neutral keys: (camera_tag,) only; renderer_type is default "isaac_rtx", override via CLI
NEUTRAL_CAMERA_COMBO = ("rgb", "depth", "albedo")
DEFAULT_NEUTRAL_RENDERER_TYPE = "isaac_rtx"


def parse_scene_key(scene_key: str) -> dict | None:
    """Parse env.scene value into width, height, renderer_type, camera_type.

    Convention: <width>x<height><renderer_tag>_<camera_tag>
    E.g. 64x64rtx_rgb or 64x64warp_rgb -> width=64, height=64, renderer_type=isaac_rtx or newton_warp, camera_type=rgb.

    Returns:
        Dict with keys width, height, renderer_type, camera_type, or None if invalid.
    """
    m = SCENE_KEY_PATTERN.match(scene_key.strip())
    if not m:
        return None
    w, h, renderer_tag, camera_tag = m.groups()
    return {
        "width": int(w),
        "height": int(h),
        "renderer_type": RENDERER_TAG_TO_TYPE[renderer_tag],
        "camera_type": CAMERA_TAG_TO_TYPE[camera_tag],
    }


def parse_neutral_scene_key(scene_key: str) -> dict | None:
    """Parse neutral env.scene value (no renderer in key): <width>x<height><camera_tag>.

    E.g. 64x64rgb, 64x64depth -> width, height, camera_type; renderer_type defaults to isaac_rtx
    and can be overridden with env.scene.base_camera.renderer_type=isaac_rtx|newton_warp.

    Returns:
        Dict with keys width, height, renderer_type (default), camera_type, or None if invalid.
    """
    m = NEUTRAL_SCENE_KEY_PATTERN.match(scene_key.strip())
    if not m:
        return None
    w, h, camera_tag = m.groups()
    return {
        "width": int(w),
        "height": int(h),
        "renderer_type": DEFAULT_NEUTRAL_RENDERER_TYPE,
        "camera_type": CAMERA_TAG_TO_TYPE[camera_tag],
    }


def get_scene_variant_keys() -> set:
    """Return the set of all variant keys (same as single_camera_variants keys)."""
    out = set()
    for (w, h) in RESOLUTIONS:
        for renderer_tag, camera_tag in RENDERER_CAMERA_COMBO:
            out.add(f"{w}x{h}{renderer_tag}_{camera_tag}")
    return out


def get_neutral_scene_variant_keys() -> set:
    """Return the set of neutral variant keys (64x64rgb, 64x64depth, etc.)."""
    out = set()
    for (w, h) in RESOLUTIONS:
        for camera_tag in NEUTRAL_CAMERA_COMBO:
            out.add(f"{w}x{h}{camera_tag}")
    return out
