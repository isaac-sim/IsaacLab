Added
^^^^^

* Added the single-namespace schema-fragment API: :class:`~isaaclab.sim.schemas.SchemaFragment`,
  the :class:`~isaaclab.sim.schemas.RigidBodyFragment` marker, and
  :class:`~isaaclab.sim.schemas.UsdPhysicsRigidBodyCfg`. Each fragment carries
  ``_usd_namespace`` / ``_usd_applied_schema`` metadata and a ``func`` applier so a prim can
  carry rigid-body properties from multiple USD namespaces at once.
* Added :func:`~isaaclab.sim.schemas.apply_namespaced` (generic fragment writer) and
  :func:`~isaaclab.sim.schemas.apply_rigid_body_properties` (applies a list of rigid-body
  fragments with ``UsdPhysics.RigidBodyAPI`` as the implicit anchor).

Changed
^^^^^^^

* Changed the spawner ``rigid_props`` slot
  (:attr:`~isaaclab.sim.spawners.RigidObjectSpawnerCfg.rigid_props`) to also accept a list of
  :class:`~isaaclab.sim.schemas.RigidBodyFragment` fragments. Legacy single cfgs continue to
  work through a transition bridge in the spawn writers.
