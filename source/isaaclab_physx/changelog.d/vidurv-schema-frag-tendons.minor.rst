Added
^^^^^

* Added :class:`~isaaclab_physx.sim.schemas.PhysxFixedTendonCfg` and
  :class:`~isaaclab_physx.sim.schemas.PhysxSpatialTendonCfg`, the PhysX tendon schema
  fragments. They override ``func`` with
  :func:`~isaaclab_physx.sim.schemas.apply_fixed_tendon` /
  :func:`~isaaclab_physx.sim.schemas.apply_spatial_tendon`, which delegate to the existing
  multi-instance tendon writers to tune every applied ``PhysxTendonAxisRootAPI`` /
  ``PhysxTendonAttachmentRootAPI`` / ``PhysxTendonAttachmentLeafAPI`` instance.

Changed
^^^^^^^

* Reworked :class:`~isaaclab_physx.sim.schemas.PhysxFixedTendonCfg` /
  :class:`~isaaclab_physx.sim.schemas.PhysxSpatialTendonCfg` appliers to tune the multi-instance
  PhysX tendon schemas directly, removing the dependency on the legacy
  ``modify_*_tendon_properties`` writers and the legacy ``Physx*TendonPropertiesCfg`` reconstruction.
  Callers relying on :class:`~isaaclab_physx.sim.schemas.PhysxFixedTendonPropertiesCfg`
  reconstruction inside the applier should pass a
  :class:`~isaaclab_physx.sim.schemas.PhysxFixedTendonCfg` fragment directly to
  :func:`~isaaclab.sim.schemas.apply_fixed_tendon_properties` instead.
