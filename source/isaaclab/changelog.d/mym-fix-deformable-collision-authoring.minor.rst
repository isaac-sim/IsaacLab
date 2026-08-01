Added
^^^^^

* Added support for ``collision_props`` alongside ``deformable_props`` on mesh spawners. The collision
  fragments are applied to the deformable's simulation mesh, which is the prim carrying
  ``UsdPhysics.CollisionAPI``. Previously the two could not be combined.

Changed
^^^^^^^

* **Breaking:** Removed ``PhysxDeformableCollisionPropertiesCfg`` from the :mod:`isaaclab.sim.schemas`
  compatibility re-exports, following its removal from :mod:`isaaclab_physx.sim.schemas`. Pass collision
  offsets through the mesh spawner's ``collision_props`` instead.
