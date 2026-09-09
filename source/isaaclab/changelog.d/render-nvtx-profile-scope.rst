Added
^^^^^

* Added an NVTX range around the renderer call inside
  :meth:`~isaaclab.renderers.render_context.RenderContext.render_into_camera`, gated by the ``ISAACLAB_RENDER_PROFILE``
  environment variable, so any rendering backend can be profiled through the same scope name
  (:data:`~isaaclab.renderers.render_context.RENDER_PROFILE_SCOPE`).
