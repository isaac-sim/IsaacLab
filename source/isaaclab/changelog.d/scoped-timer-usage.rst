Fixed
^^^^^

* Fixed ray-caster diagnostic timings to synchronize the simulation device before and after sensor updates,
  excluding previously queued simulation work. Consolidated synchronized benchmark boundaries through
  :class:`~isaaclab.utils.timer.Timer` while preserving Torch-device synchronization.
