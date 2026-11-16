* Changed :class:`~isaaclab_contrib.sensors.VisuoTactileSensor` to mark its refresh with
  :func:`~isaaclab.utils.warp.capture.eager_method`, since it renders on the host: a
  :class:`~isaaclab.utils.warp.capture.GraphedCallable` that refreshes it runs without a CUDA graph.
