Fixed
^^^^^

* Fixed camera world-pose resolution stalling (and benchmark timeouts) at high environment
  counts under the PhysX backend. The Fabric frame view now tags its prims with per-view
  Fabric index attributes so prim selections match only the view's prims instead of every
  xformable in the stage, and rebuilds the view-to-Fabric index mapping on the GPU on each
  access instead of resolving prim paths on the host on every environment reset.
