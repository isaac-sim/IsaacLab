Fixed
^^^^^

* Fixed camera world-pose resolution stalling at high environment counts under the
  PhysX backend, which caused multi-second pauses between rendered frames and
  benchmark timeouts.

Added
^^^^^

* Added :meth:`close` to the PhysX Fabric frame view, removing its per-view Fabric
  index attributes so that views recreated over the same prims no longer accumulate
  attributes. Views dropped without closing are cleaned up on garbage collection,
  with a warning.
