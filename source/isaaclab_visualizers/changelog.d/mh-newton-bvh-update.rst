Fixed
^^^^^

* Fixed the Newton viewer silently running headless when no display is available: the implicit
  EGL fallback in :class:`~isaaclab_visualizers.newton.NewtonVisualizer` now prints a warning
  explaining that no window will open and how to enable one.
