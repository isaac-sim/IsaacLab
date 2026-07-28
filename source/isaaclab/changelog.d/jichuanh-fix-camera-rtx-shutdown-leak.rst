Fixed
^^^^^

* Fixed a crash on shutdown (``SIGSEGV``) after training a camera-based task with the
  RTX renderer. Renderer resources for a :class:`~isaaclab.sensors.Camera` were released
  from its finalizer, which only runs once the last reference to the camera is dropped.
  Tasks commonly keep an extra reference -- an environment attribute, or an observation
  term that caches the sensor -- so the finalizer never ran and shader processors and
  pipeline layouts were still registered when the application shut down.
  :class:`~isaaclab.renderers.RenderContext` now owns render data through its new
  :meth:`~isaaclab.renderers.RenderContext.create_render_data` and
  :meth:`~isaaclab.renderers.RenderContext.close` methods, and releases it during
  simulation teardown regardless of any surviving references.
