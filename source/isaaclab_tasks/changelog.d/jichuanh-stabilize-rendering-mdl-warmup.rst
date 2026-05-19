Fixed
^^^^^

* Stabilized RTX MDL shader-warmup flakes in the rendering-correctness tests
  by stepping the env 10 frames before reading the camera tensor, instead of
  a single step. Affected ``test_shadow_hand_vision_presets.py``'s
  ``test_camera_renders_not_empty`` and the three helper-driven tests in
  ``rendering_test_utils.py``
  (``rendering_test_shadow_hand``/``_cartpole``/``_dexsuite_kuka``) — all of
  which intermittently failed with "Camera output is all zeros or all inf"
  for ``simple_shading_*_mdl`` and ``simple_shading_constant_diffuse``
  variants on cold-cache CI runners (the GPU returned a still-zero
  framebuffer because the MDL material hadn't finished compiling).
