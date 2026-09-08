Fixed
^^^^^

* Fixed :class:`~isaaclab_visualizers.newton.NewtonRTXVisualizer` hanging the process when
  combined with a Kit-based physics backend (``physx`` or ``ovphysx``). OVRTX is a kitless
  renderer and previously crashed inside the render thread on the first ``step()``, which left
  the process stuck instead of exiting. It now raises a clear ``RuntimeError`` from
  ``initialize()`` naming the incompatible combination and the supported alternatives.
