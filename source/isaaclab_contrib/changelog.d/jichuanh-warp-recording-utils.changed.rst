* Changed :class:`~isaaclab_contrib.sensors.VisuoTactileSensor` to refresh through
  :func:`~isaaclab.utils.warp.capture.eager`, since its refresh renders on the host: a
  :class:`~isaaclab.utils.warp.capture.GraphedCallable` that refreshes it runs without a CUDA graph.
