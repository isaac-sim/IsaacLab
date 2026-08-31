Fixed
^^^^^

* Fixed PhysX fixed- and spatial-tendon schema fragments authoring settings outside the schema-declared
  ``physxTendon`` namespace.

* Fixed :class:`~isaaclab_physx.sim.schemas.PhysxSpatialTendonCfg` writing its settings to
  attachment-leaf instances. Stiffness, damping and the other fields it carries are declared only by
  ``PhysxTendonAttachmentRootAPI``; leaf and intermediate attachments carry per-element geometry, so
  the values were authored under names PhysX does not read. Only the attachment root is tuned now.
