Changed
^^^^^^^

* :class:`~isaaclab_physx.app.KitLauncher` auto-starts XR only when the run has no Kit visualizer, whether it comes
  from the config or ``--visualizer``, so a Kit visualizer declared in the config keeps its window with ``--xr``
  instead of being forced headless.
