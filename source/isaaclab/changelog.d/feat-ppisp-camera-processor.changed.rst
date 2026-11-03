* Changed ``BaseRenderer.read_output``, ``RenderContext.render_into_camera``, and ``render_into_cameras`` to
  return whether a capture was published. Custom renderers should return ``True`` after publishing.
