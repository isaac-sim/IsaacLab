Added
^^^^^

* Added support for ``collision_props`` alongside ``deformable_props`` on mesh spawners, so a
  deformable's collider can be tuned with
  :class:`~isaaclab.sim.schemas.CollisionFragment` fragments. The fragments are applied to the
  simulation mesh, which is the prim carrying ``UsdPhysics.CollisionAPI``. Previously the two
  could not be combined and PhysX collision offsets were unreachable for deformables.

Changed
^^^^^^^

* **Breaking:** Removed ``PhysxDeformableCollisionPropertiesCfg`` from the
  :mod:`isaaclab.sim.schemas` compatibility re-exports, following its removal from
  :mod:`isaaclab_physx.sim.schemas`. Pass collision offsets through the spawner's
  ``collision_props`` instead, for example
  ``collision_props=[PhysxCollisionCfg(rest_offset=0.0005, contact_offset=0.005)]``.
