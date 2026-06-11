Added
^^^^^

* Added :class:`~isaaclab.utils.warp.ParticleMeshCounter` for fast, training-time counting of
  particles inside closed (watertight) region meshes via robust winding-number point queries.
  The counter supports multiple, independently posed region meshes per environment, sanitizes
  non-finite particle positions, and returns both per-region counts and the per-particle
  containment mask.
* Added the :func:`~isaaclab.utils.warp.make_box_region_mesh` and
  :func:`~isaaclab.utils.warp.make_frustum_region_mesh` helpers for building watertight,
  outward-oriented region meshes (axis-aligned boxes and capped circular frusta / cup cavities).
