Fixed
^^^^^

* Fixed a crash on shutdown (``SIGSEGV``) after training a camera-based task with the RTX
  renderer. :class:`~isaaclab.sensors.Camera` released its renderer resources from a finalizer,
  which does not run while any other reference to the camera survives.
  :class:`~isaaclab.renderers.RenderContext` now owns render data and releases it during
  simulation teardown.
