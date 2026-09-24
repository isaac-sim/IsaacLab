Changed
^^^^^^^

* Changed the reach table collider, the UR10 particle-push colliders, and the NIST factory Newton
  Franka rigid-body properties to author their physics schemas with schema fragments
  (:class:`~isaaclab.sim.schemas.UsdPhysicsCollisionCfg`,
  :class:`~isaaclab_newton.sim.schemas.NewtonCollisionCfg`, and
  :class:`~isaaclab_newton.sim.schemas.MujocoRigidBodyCfg`) instead of the deprecated legacy
  property configs, so loading these tasks no longer emits their ``DeprecationWarning``. The
  authored USD is unchanged. Configurations that tune these spawner slots in place should select
  the fragment that owns the field (e.g. the :class:`~isaaclab_newton.sim.schemas.NewtonCollisionCfg`
  entry of the UR10 particle-push ``collision_props`` list for ``contact_margin``).
