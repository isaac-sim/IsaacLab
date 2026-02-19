# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test parse_scene_key and scene variant keys (no isaaclab/Isaac Sim required).

Run from the kuka_allegro config directory (so only scene_variant_keys is loaded):

  cd .../dexsuite/config/kuka_allegro && python test_scene_variant_keys.py

  cd .../dexsuite/config/kuka_allegro && python -c "
  from scene_variant_keys import parse_scene_key, get_scene_variant_keys
  p = parse_scene_key('64x64tiled_rgb')
  assert p == {'width': 64, 'height': 64, 'renderer_type': 'rtx', 'camera_type': 'rgb'}, p
  p2 = parse_scene_key('128x128newton_depth')
  assert p2 == {'width': 128, 'height': 128, 'renderer_type': 'newton_warp', 'camera_type': 'distance_to_image_plane'}, p2
  assert parse_scene_key('invalid') is None
  keys = get_scene_variant_keys()
  assert '64x64tiled_rgb' in keys and '64x64newton_rgb' in keys and '256x256newton_albedo' in keys
  print('parse_scene_key and get_scene_variant_keys OK')
  "
"""

import sys

# Allow running as script: add parent package path so "from . import scene_variant_keys" works
if __name__ == "__main__":
    from scene_variant_keys import get_scene_variant_keys, parse_scene_key

    # parsing
    p = parse_scene_key("64x64tiled_rgb")
    assert p == {"width": 64, "height": 64, "renderer_type": "rtx", "camera_type": "rgb"}, p
    p2 = parse_scene_key("128x128newton_depth")
    assert p2 == {
        "width": 128,
        "height": 128,
        "renderer_type": "newton_warp",
        "camera_type": "distance_to_image_plane",
    }, p2
    assert parse_scene_key("invalid") is None
    # variant keys (same set as single_camera_variants.keys())
    keys = get_scene_variant_keys()
    assert "64x64tiled_rgb" in keys
    assert "64x64newton_rgb" in keys
    assert "256x256newton_albedo" in keys
    print("parse_scene_key and get_scene_variant_keys OK")
    sys.exit(0)
