Added
^^^^^

* Added support for :attr:`~isaaclab.sensors.camera.CameraCfg.background_color` in
  :class:`~isaaclab_ov.renderers.OVRTXRenderer`. When set, authors
  ``omni:rtx:background:source:type = "color"`` and ``omni:rtx:background:source:color`` on the
  USD render product instead of the default ``"domeLight"`` background.
