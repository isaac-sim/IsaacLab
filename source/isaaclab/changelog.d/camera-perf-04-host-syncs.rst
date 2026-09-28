Changed
^^^^^^^

* Removed a per-step host synchronization from termination tracking and moved marker ID validation
  to the backend that consumes the IDs.
* Skipped the camera mask device-to-host copy when ``update_period`` is zero, since every step
  marks all cameras outdated.
* Changed uniform ``add`` noise with a zero-width range to return its input without drawing samples.
* Cached :class:`~isaaclab.managers.ActionManager` term dimensions instead of recomputing them on every
  action step.
