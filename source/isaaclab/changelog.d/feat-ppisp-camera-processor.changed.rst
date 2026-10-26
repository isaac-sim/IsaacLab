* Changed ``BaseRenderer.read_output`` to return whether it published a capture, and
  ``RenderContext.render_into_camera`` and ``render_into_cameras`` to return those results, so cameras run
  tensor processors once per published capture. Custom renderers should return ``True`` after publishing.
