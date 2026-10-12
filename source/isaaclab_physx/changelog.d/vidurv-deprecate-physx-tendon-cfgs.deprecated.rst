* Deprecated :class:`~isaaclab_physx.sim.schemas.PhysxFixedTendonPropertiesCfg` and its
  ``FixedTendonPropertiesCfg`` alias in favor of
  :class:`~isaaclab_physx.sim.schemas.PhysxTendonAxisRootCfg` in the spawner's
  ``fixed_tendons_props`` slot. They will be removed in 3.2.
* Deprecated :class:`~isaaclab_physx.sim.schemas.PhysxSpatialTendonPropertiesCfg` and its
  ``SpatialTendonPropertiesCfg`` alias in favor of
  :class:`~isaaclab_physx.sim.schemas.PhysxTendonAttachmentRootCfg` in the spawner's
  ``spatial_tendons_props`` slot. They will be removed in 3.2. The aliases previously
  pointed to the ``Physx*TendonPropertiesCfg`` classes, which are deprecated as well.
