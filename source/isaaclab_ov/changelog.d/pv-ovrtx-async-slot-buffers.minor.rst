Added
^^^^^

* Added an opt-in asynchronous OVRTX render path controlled by
  :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.async_rendering`.
  ``True`` trades one frame of camera latency for pipelined rendering: rendering then overlaps the
  next step's simulation and Python work, improving throughput, and camera outputs are one step
  stale. Scene writes and renders are grouped into frames by
  :meth:`~isaaclab.renderers.BaseRenderer.announce_frame` announcements, which the framework
  makes with the physics step count. Each camera's first frame is consumed immediately, so its
  first read returns a valid frame.
