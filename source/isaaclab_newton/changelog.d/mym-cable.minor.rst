Added
^^^^^

* Added :class:`~isaaclab_newton.sim.spawners.materials.NewtonCableMaterialCfg`
  for cable rod material parameters (stretch / bend stiffness, damping, density).
* Added a per-cable ignore-paths block in :func:`newton_physics_replicate` so
  ``add_usd`` skips cable ``BasisCurves`` prims (materialized via the per-world
  builder hook instead).
