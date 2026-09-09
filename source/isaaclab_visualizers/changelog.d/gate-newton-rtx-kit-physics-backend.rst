Fixed
^^^^^

* Fixed :class:`~isaaclab_visualizers.newton.NewtonRTXVisualizer` hanging the process when
  combined with the Kit-based ``physx`` physics backend (i.e. ``presets=isaacsim_physx``).
  OVRTX is a kitless renderer and previously crashed inside the render thread on the first
  ``step()``, which left the process stuck instead of exiting. It now raises a clear
  ``RuntimeError`` from ``initialize()`` naming the incompatible combination and the supported
  alternatives. The kitless ``ovphysx`` backend is unaffected and remains supported.
