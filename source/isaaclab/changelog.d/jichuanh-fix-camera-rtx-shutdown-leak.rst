Fixed
^^^^^

* Fixed a crash on shutdown (``SIGSEGV``) after training a camera-based task with the RTX
  renderer. :class:`~isaaclab.sensors.Camera` released its renderer resources from a finalizer,
  which does not run while any other reference to the camera survives.

Added
^^^^^

* Added :meth:`~isaaclab.sensors.SensorBase.close` to release a sensor's simulator-side
  resources, and :meth:`~isaaclab.scene.InteractiveScene.close` to close every sensor in the
  scene. Both are called during simulation teardown.
