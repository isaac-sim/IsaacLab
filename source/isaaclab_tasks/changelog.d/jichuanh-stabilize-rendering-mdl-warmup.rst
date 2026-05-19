Fixed
^^^^^

* Stabilized RTX MDL shader-warmup flakes in the rendering-correctness tests
  by polling the camera output until every data-type tensor reports a
  non-zero max before the assertion / golden compare, instead of relying on
  a single render pass:

  * ``test_camera_renders_not_empty`` in
    ``test_shadow_hand_vision_presets.py`` polls via ``env.step()`` with a
    60-step cap.
  * The three goldenfile helpers in ``rendering_test_utils.py``
    (``rendering_test_shadow_hand`` / ``_cartpole`` / ``_dexsuite_kuka``)
    poll via ``sim.render()`` + ``scene.update()`` with a 30-pass cap, so
    physics state is not advanced and the existing goldens stay valid.

  All variants intermittently failed with "Camera output is all zeros or
  all inf" for ``simple_shading_*_mdl`` and ``simple_shading_constant_diffuse``
  on cold-cache CI runners because the GPU returned a still-zero
  framebuffer before the MDL material finished compiling.
