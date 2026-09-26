Fixed
^^^^^

* Fixed an ``IndexError`` in ``NewtonSiteFrameView`` when a world-attached (body-less) site is
  resolved per environment under the flat (non-replicated, ``replicate_physics=False``) builder.
  ``NewtonManager._world_xforms`` has exactly one entry for the flat builder regardless of
  environment count, so callers indexing it with an environment index greater than 0 (e.g. a
  streaming/follow camera targeting a per-environment prim with no body ancestor) hit an
  out-of-bounds index. Such callers now fall back to the single shared world.
