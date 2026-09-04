Added
^^^^^

* Added :class:`~isaaclab_physx.sim.schemas.PhysxTendonAxisCfg` for configuring the
  ``PhysxTendonAxisAPI`` properties of existing fixed-tendon instances.
* Added ``lower_limit`` and ``upper_limit`` to
  :class:`~isaaclab_physx.sim.schemas.PhysxTendonAxisRootCfg` and
  :class:`~isaaclab_physx.sim.schemas.PhysxFixedTendonPropertiesCfg`.

Changed
^^^^^^^

* Renamed ``PhysxFixedTendonCfg`` to
  :class:`~isaaclab_physx.sim.schemas.PhysxTendonAxisRootCfg` and
  ``PhysxSpatialTendonCfg`` to
  :class:`~isaaclab_physx.sim.schemas.PhysxTendonAttachmentRootCfg` so every fragment name matches
  its USD schema. No compatibility aliases are provided.
* Added ``instance_names`` to :class:`~isaaclab_physx.sim.schemas.PhysxTendonAxisRootCfg` and
  :class:`~isaaclab_physx.sim.schemas.PhysxTendonAttachmentRootCfg`. Pass one name or a list to select
  existing tendon instances; the default ``None`` preserves the previous broadcast behavior.
* Changed :class:`~isaaclab_physx.sim.schemas.PhysxTendonAttachmentRootCfg` to configure only
  ``PhysxTendonAttachmentRootAPI`` instances. Leaf and intermediate attachment topology remains
  asset-authored.

Removed
^^^^^^^

* Removed the per-prim ``apply_fixed_tendon`` and ``apply_spatial_tendon`` functions from
  :mod:`isaaclab_physx.sim.schemas`. Configure tendon fragments through
  :func:`isaaclab.sim.schemas.apply_fixed_tendon_properties` and
  :func:`isaaclab.sim.schemas.apply_spatial_tendon_properties`, respectively.
