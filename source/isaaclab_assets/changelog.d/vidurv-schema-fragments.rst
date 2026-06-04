Changed
^^^^^^^

* Migrated the robot configurations off the deprecated ``RigidBodyPropertiesCfg`` to the new
  rigid-body fragment list (``rigid_props=[UsdPhysicsRigidBodyCfg(...), PhysxRigidBodyCfg(...)]``).
  The ``rigid_props`` attribute on the pre-defined robot cfgs is now a list of fragments rather
  than a single cfg object.
