* Separated shared image selection from Newton window dimensions through ``WindowCfg``.
  RTX capture copied native output directly on the GPU;
  only CPU consumers downloaded pixels. Sensor-view window resizing retained source and output storage.
  ``WindowCfg.size`` set initial viewer dimensions; RTX perspective rendering kept that resolution,
  while GL perspective rendering followed its window framebuffer.
  Unified GL and RTX capture lifecycle checks; call ``render_rgb_array()`` after ``sim.reset()``.
* Kept shared views on ``VisualizerCfg.view``, separate from window presentation settings.
  Detached perspective render callbacks when their owning viewer closed, allowing another viewer
  to bind the retained view without accessing the destroyed renderer.
* Shared Kit window docking between viewport and camera-image presentation.
* Keyed sensor-view capture to the physics step, fixing frozen headless
  ``--video viz:newton_gl:streaming_view`` clips while reusing repeated reads within one step.
  Explicit headless viewer updates also refreshed perspective captures after marker edits without a physics step.
