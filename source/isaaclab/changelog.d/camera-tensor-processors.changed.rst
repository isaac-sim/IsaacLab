* Changed ``BaseRenderer.read_output``, ``RenderContext.render_into_camera``, and ``render_into_cameras`` to
  return whether a capture was published. Renderers that hold back a capture return ``False``; a ``None``
  return counts as published.
