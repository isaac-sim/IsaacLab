Added
^^^^^

* Added the Newton mesh-collision cooking fragments:
  :class:`~isaaclab_newton.sim.schemas.NewtonMeshCollisionCfg` (``newton:maxHullVertices`` via
  ``NewtonMeshCollisionAPI``) and :class:`~isaaclab_newton.sim.schemas.NewtonSDFCollisionCfg`
  (Newton SDF generation and hydroelastic-contact attributes via ``NewtonSDFCollisionAPI``). Each is
  a single-namespace :class:`~isaaclab.sim.schemas.MeshCollisionFragment` dispatched via
  :func:`~isaaclab.sim.schemas.apply_mesh_collision_properties`.
