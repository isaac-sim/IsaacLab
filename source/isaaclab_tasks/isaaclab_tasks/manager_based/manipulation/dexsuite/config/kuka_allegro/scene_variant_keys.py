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
RENDERER_TAG_TO_TYPE = {"rtx": "rtx", "warp": "warp_renderer"}
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


def parse_scene_key(scene_key: str) -> dict | None:
    """Parse env.scene value into width, height, renderer_type, camera_type.

    Convention: <width>x<height><renderer_tag>_<camera_tag>
    E.g. 64x64rtx_rgb or 64x64warp_rgb -> width=64, height=64, renderer_type=rtx or warp_renderer, camera_type=rgb.

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


def get_scene_variant_keys() -> set:
    """Return the set of all variant keys (same as single_camera_variants keys)."""
    out = set()
    for (w, h) in RESOLUTIONS:
        for renderer_tag, camera_tag in RENDERER_CAMERA_COMBO:
            out.add(f"{w}x{h}{renderer_tag}_{camera_tag}")
    return out
