Fixed
^^^^^

* Fixed a duplicate-USD crash (``pxrInternal_v0_*`` Python class re-registration,
  e.g. ``extension class wrapper for base class ... has not been created yet``)
  hit while constructing environments when a pip ``usd-core`` is installed
  alongside Isaac Sim's bundled USD. Module-level ``pxr``/``isaacsim`` imports in
  :mod:`isaaclab.cloner`, :mod:`isaaclab.sensors`, :mod:`isaaclab.terrains`,
  :mod:`isaaclab.scene_data`, and :mod:`isaaclab.sim.utils.stage` are now deferred
  into the functions that use them, so Kit registers a single USD version on launch.
