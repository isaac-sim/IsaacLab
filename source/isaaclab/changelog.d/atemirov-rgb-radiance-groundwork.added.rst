* Added the ``rgb_radiance`` camera output: scene-linear RGB before camera exposure and response,
  in renderer-relative intensity units. When ``rgb_hdr`` and ``rgb_radiance`` are both requested,
  they share one buffer.
* Added :meth:`~isaaclab.renderers.BaseRenderer.apply_camera_settings` so renderers apply the
  process-wide settings their cameras need before ``sim.reset()``.
