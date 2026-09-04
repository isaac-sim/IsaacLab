Fixed
^^^^^

* Fixed :class:`~isaaclab_visualizers.newton.newton_visualizer.NewtonVisualizer` releasing its viewer
  without calling the viewer's :meth:`close`, which left the RTX backend's ordered GPU teardown to the
  garbage collector and intermittently leaked render step results and attribute bindings on shutdown.
