Fixed
^^^^^

* Fixed camera world-pose resolution stalling (and benchmark timeouts) at high environment
  counts under the PhysX backend. The Fabric frame view no longer rebuilds a whole-scene
  path-to-index map on every environment reset; resolved indices are cached and recomputed
  only when the prim selection changes.
