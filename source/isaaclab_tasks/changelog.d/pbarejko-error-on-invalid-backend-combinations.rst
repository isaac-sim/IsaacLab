Added
^^^^^

* Added validation in :mod:`isaaclab_tasks.utils.sim_launcher` that raises a descriptive
  error when an unsupported physics/renderer/visualizer combination is requested
  (e.g. the kitless OVRTX renderer paired with Isaac Sim PhysX or the Kit visualizer),
  pointing users at the correct preset instead of failing later with an opaque runtime error.
