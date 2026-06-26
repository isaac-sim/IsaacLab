Added
^^^^^

* Added the collision schema-fragment API: the
  :class:`~isaaclab.sim.schemas.CollisionFragment` marker and
  :class:`~isaaclab.sim.schemas.UsdPhysicsCollisionCfg` (the ``physics:collisionEnabled``
  single-namespace fragment). Each fragment carries ``_usd_namespace`` / ``_usd_applied_schema``
  metadata and a ``func`` applier so a prim can carry collision properties from multiple USD
  namespaces at once.
* Added :func:`~isaaclab.sim.schemas.apply_collision_properties`, which applies a list of
  collision fragments with ``UsdPhysics.CollisionAPI`` as the implicit anchor.

Changed
^^^^^^^

* Changed the spawner ``collision_props`` slot
  (:attr:`~isaaclab.sim.spawners.RigidObjectSpawnerCfg.collision_props`) and the mesh-converter
  ``collision_props`` slot to also accept a list of
  :class:`~isaaclab.sim.schemas.CollisionFragment` fragments. Legacy single cfgs continue to work
  through a transition bridge in the spawn writers.
