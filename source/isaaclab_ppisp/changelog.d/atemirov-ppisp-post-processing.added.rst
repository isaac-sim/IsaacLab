* Added ``PpispProcessorCfg`` for ordered visual observation processing,
  including USD configuration discovery and controller buffer initialization. Declared
  ``rgb_radiance`` as its input so either the renderer or an earlier processor could supply
  scene-linear RGB before exposure and camera response.
* Added ``PpispDiscoveryMode`` in ``isaaclab_ppisp`` with the existing ``AUTO_CAMERA`` and
  ``AUTO_ANY`` discovery modes.
