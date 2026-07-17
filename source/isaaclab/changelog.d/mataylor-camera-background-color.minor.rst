Added
^^^^^

* Added :attr:`~isaaclab.sensors.camera.CameraCfg.background_color` to
  :class:`~isaaclab.sensors.camera.CameraCfg` as a unified cross-backend way to set the camera
  background to a solid color. Accepts normalized RGB floats ``(r, g, b)`` in ``[0, 1]``;
  defaults to ``None`` (each backend's original default background).
