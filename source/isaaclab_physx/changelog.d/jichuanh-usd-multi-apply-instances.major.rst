Changed
^^^^^^^

* **Breaking:** :class:`~isaaclab_physx.sim.schemas.PhysxFixedTendonCfg` and
  :class:`~isaaclab_physx.sim.schemas.PhysxSpatialTendonCfg` now require ``instance_names``, which
  selects the tendon instances the fragment tunes: one name, a list of names, or ``None`` for every
  instance on the prim. A fragment without it raises at spawn instead of silently tuning every
  instance, so two tendons rooted on one joint can be configured independently. Migration: add
  ``instance_names=None`` to keep the previous behavior.

* Changed the fixed- and spatial-tendon fragments to dispatch through the generic
  :func:`~isaaclab.sim.schemas.apply_schema_instances`; the schema and attribute layout are static data
  on the fragment.

Removed
^^^^^^^

* **Breaking:** Removed ``apply_fixed_tendon`` and ``apply_spatial_tendon`` from
  :mod:`isaaclab_physx.sim.schemas`. They only bound the fragments to a PhysX-side writer that no
  longer exists. Migration: call :func:`~isaaclab.sim.schemas.apply_schema_instances`, or the family
  writer :func:`~isaaclab.sim.schemas.apply_fixed_tendon_properties`.

Added
^^^^^

* Added ``lower_limit`` and ``upper_limit`` to :class:`~isaaclab_physx.sim.schemas.PhysxFixedTendonCfg`
  and :class:`~isaaclab_physx.sim.schemas.PhysxFixedTendonPropertiesCfg`, completing the tendon-length
  limits the ``PhysxTendonAxisRootAPI`` schema declares.

Fixed
^^^^^

* Fixed the PhysX fixed- and spatial-tendon fragments authoring their settings under
  ``PhysxTendonAxisRootAPI:<instance>:*`` / ``PhysxTendonAttachmentRootAPI:<instance>:*``, names
  PhysX never reads. They now land in the ``physxTendon:<instance>:*`` namespace.

* Fixed :class:`~isaaclab_physx.sim.schemas.PhysxSpatialTendonCfg` writing its settings to
  attachment-leaf instances, which declare per-attachment geometry rather than the tendon's dynamics.
  Only attachment roots are tuned.
