Fixed
^^^^^

* Fixed :class:`~isaaclab_visualizers.newton.newton_visualizer.NewtonRTXVisualizer` releasing its viewer
  without first neutralizing picking callbacks and calling the viewer's :meth:`close`, which left its ordered
  GPU teardown to the garbage collector and intermittently leaked render step results and attribute bindings
  on shutdown.
