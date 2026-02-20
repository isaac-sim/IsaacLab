# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test parse_scene_key and scene variant keys (no isaaclab/Isaac Sim required).

Run from the kuka_allegro config directory (so only scene_variant_keys is loaded):

  cd .../dexsuite/config/kuka_allegro && python test_scene_variant_keys.py

  cd .../dexsuite/config/kuka_allegro && python -c "
  from scene_variant_keys import parse_scene_key, get_scene_variant_keys
  p = parse_scene_key('64x64rtx_rgb')
  assert p == {'width': 64, 'height': 64, 'renderer_type': 'rtx', 'camera_type': 'rgb'}, p
  p2 = parse_scene_key('128x128warp_depth')
  assert p2 == {'width': 128, 'height': 128, 'renderer_type': 'warp_renderer', 'camera_type': 'distance_to_image_plane'}, p2
  assert parse_scene_key('invalid') is None
  keys = get_scene_variant_keys()
  assert '64x64rtx_rgb' in keys and '64x64warp_rgb' in keys and '256x256warp_albedo' in keys
  print('parse_scene_key and get_scene_variant_keys OK')
  "
"""

import sys

# Allow running as script: add parent package path so "from . import scene_variant_keys" works
if __name__ == "__main__":
    from scene_variant_keys import (
        get_neutral_scene_variant_keys,
        get_scene_variant_keys,
        parse_neutral_scene_key,
        parse_scene_key,
    )

    # parsing (renderer in key)
    p = parse_scene_key("64x64rtx_rgb")
    assert p == {"width": 64, "height": 64, "renderer_type": "rtx", "camera_type": "rgb"}, p
    p2 = parse_scene_key("128x128warp_depth")
    assert p2 == {
        "width": 128,
        "height": 128,
        "renderer_type": "warp_renderer",
        "camera_type": "distance_to_image_plane",
    }, p2
    assert parse_scene_key("invalid") is None
    # parsing (neutral keys)
    n = parse_neutral_scene_key("64x64rgb")
    assert n["width"] == 64 and n["height"] == 64 and n["camera_type"] == "rgb" and n["renderer_type"] == "rtx", n
    assert parse_neutral_scene_key("64x64depth")["camera_type"] == "distance_to_image_plane"
    assert parse_neutral_scene_key("64x64rtx_rgb") is None  # neutral pattern does not match full key
    # variant keys (same set as single_camera_variants.keys())
    keys = get_scene_variant_keys()
    assert "64x64rtx_rgb" in keys
    assert "64x64warp_rgb" in keys
    assert "256x256warp_albedo" in keys
    neutral = get_neutral_scene_variant_keys()
    assert "64x64rgb" in neutral and "64x64depth" in neutral
    print("parse_scene_key, parse_neutral_scene_key, get_scene_variant_keys, get_neutral_scene_variant_keys OK")
    sys.exit(0)
