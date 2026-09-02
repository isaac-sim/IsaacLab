Changed
^^^^^^^

* Changed fixed tendons to be named after their ``PhysxTendonAxisRootAPI`` instance rather than the
  joint prim carrying it, matching OVPhysX and Newton. Code that looked a tendon up by its joint
  name, through ``find_fixed_tendons`` or ``SceneEntityCfg.fixed_tendon_names``, must use the
  instance name.

Fixed
^^^^^

* Fixed PhysX fixed- and spatial-tendon schema fragments authoring settings outside the schema-declared
  ``physxTendon`` namespace.

* Fixed :class:`~isaaclab_physx.sim.schemas.PhysxSpatialTendonCfg` writing its settings to
  attachment-leaf instances. Stiffness, damping and the other fields it carries are declared only by
  ``PhysxTendonAttachmentRootAPI``; leaf and intermediate attachments carry per-element geometry, so
  the values were authored under names PhysX does not read. Only the attachment root is tuned now.
