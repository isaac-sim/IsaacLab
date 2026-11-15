* Changed :class:`~isaaclab_contrib.sensors.VisuoTactileSensor` to refresh eagerly at its place on every replay of a
  :class:`~isaaclab.utils.warp.capture.GraphedCallable` recording, since its refresh renders on the host.
