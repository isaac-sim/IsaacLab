Changed
^^^^^^^

* Extended :func:`~isaaclab.sim.schemas.apply_fixed_tendon_properties` to target both
  ``PhysxTendonAxisRootAPI`` and ``PhysxTendonAxisAPI`` instances, allowing backend fragments to
  configure whole fixed tendons and their individual joint-axis contributions separately.

Fixed
^^^^^

* Fixed the legacy fixed- and spatial-tendon writers authoring properties under the applied API
  type name. They now use the schema-owned ``physxTendon:<instance>:*`` namespace that PhysX reads.
