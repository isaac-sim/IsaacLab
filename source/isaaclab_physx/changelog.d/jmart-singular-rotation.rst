Fixed
^^^^^

* Fixed the acceleration-arrow debug visualizer in
  :class:`~isaaclab_physx.sensors.pva.Pva` drawing arrows in undefined directions for
  bodies with effectively zero acceleration. Such bodies are now skipped from the
  visualization.
