Fixed
^^^^^

* Stabilized ``test_camera_renders_not_empty`` in
  ``test_shadow_hand_vision_presets.py`` by polling the camera output until
  all data-type tensors are non-zero, with a 60-step cap, instead of a
  single ``env.step()``. The test intermittently failed with "Camera output
  is all zeros or all inf" for ``simple_shading_*_mdl`` and
  ``simple_shading_constant_diffuse`` variants on cold-cache CI runners
  because the GPU returned a still-zero framebuffer before the MDL material
  finished compiling. The three goldenfile-comparing helpers in
  ``rendering_test_utils.py`` already use ``flaky(max_runs=3)`` and are left
  untouched.
