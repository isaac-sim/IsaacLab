* Changed ``BaseRenderer.read_output``, ``RenderContext.render_into_camera``, and ``render_into_cameras`` to
  return the published capture as a :class:`~isaaclab.renderers.CaptureInfo`, or ``None`` when no new capture
  was published. ``BaseRenderer.prepare_capture`` receives the capture's ``env_mask``, which ``read_output``
  returns with that capture. Custom renderers that use tensor processors must return a ``CaptureInfo``.
