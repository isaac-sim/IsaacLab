Added
^^^^^

* Added support for :attr:`~isaaclab.sensors.camera.CameraCfg.background_color` in
  :class:`~isaaclab_physx.renderers.IsaacRtxRenderer`. When set, applies
  ``/rtx/background/source/type = 2`` (Color) and ``/rtx/background/source/color`` via the
  settings manager during camera setup.
