Added
^^^^^

* Added :class:`~isaaclab.sim.spawners.shapes.CableCfg` and
  :func:`~isaaclab.sim.spawners.shapes.spawn_cable` for authoring 1D cable / rod
  prims as ``UsdGeomBasisCurves``. Physics is materialized by the Newton
  replicate hook in the contrib package; see
  :class:`~isaaclab_contrib.cable.CableObject`.
