Fixed
^^^^^

* Stabilized RTX MDL shader-warmup flakes in the rendering-correctness tests
  by driving 10 extra ``sim.render()`` + ``scene.update()`` passes between env
  construction and the camera read in
  ``rendering_test_utils.py``'s ``rendering_test_shadow_hand``,
  ``rendering_test_cartpole``, and ``rendering_test_dexsuite_kuka`` helpers.
  The warmup mirrors the pattern already used by
  :attr:`~isaaclab.envs.DirectRLEnvCfg.num_rerenders_on_reset` —
  it does not advance physics, so the existing golden images remain valid.
  ``test_camera_renders_not_empty`` in
  ``test_shadow_hand_vision_presets.py`` (which has no golden compare) is
  stabilized by stepping the env 10 frames before the non-zero pixel check.
  Both flakes manifested as "Camera output is all zeros or all inf" for
  ``simple_shading_*_mdl`` and ``simple_shading_constant_diffuse`` variants on
  cold-cache CI runners — the GPU returned a still-zero framebuffer because
  the MDL material had not finished compiling.
