Changed
^^^^^^^

* **Breaking:** Removed ``PhysxDeformableCollisionPropertiesCfg`` and the ``contact_offset`` /
  ``rest_offset`` fields it contributed to
  :class:`~isaaclab_physx.sim.schemas.PhysxDeformableBodyPropertiesCfg`. They were authored onto
  the deformable body prim, but PhysX reads collision offsets off the collider, which for a
  deformable is its simulation mesh, so the values never reached the solver and it fell back to
  the PhysX defaults. Pass the offsets through the mesh spawner's ``collision_props`` instead, for
  example ``collision_props=[PhysxCollisionCfg(rest_offset=0.0005, contact_offset=0.005)]``.
