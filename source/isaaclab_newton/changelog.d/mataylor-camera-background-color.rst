Added
^^^^^

* Added support for :attr:`~isaaclab.sensors.camera.CameraCfg.background_color` in
  :class:`~isaaclab_newton.renderers.NewtonWarpRenderer`. When set, converts the normalized RGB
  color to an ARGB clear color passed to ``SensorTiledCamera.ClearData`` on each render call.
