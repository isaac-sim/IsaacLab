Changed
^^^^^^^

* **Breaking:** Moved deformable body schema and material APIs from
  :mod:`isaaclab_physx.sim` to :mod:`isaaclab.sim`, and moved deformable object
  configuration from :mod:`isaaclab_physx.assets` to :mod:`isaaclab.assets`.
  Import :class:`~isaaclab.sim.DeformableBodyPropertiesCfg`,
  :func:`~isaaclab.sim.define_deformable_body_properties`,
  :func:`~isaaclab.sim.modify_deformable_body_properties`,
  :class:`~isaaclab.sim.DeformableObjectSpawnerCfg`,
  :class:`~isaaclab.sim.DeformableBodyMaterialCfg`,
  :class:`~isaaclab.sim.SurfaceDeformableBodyMaterialCfg`, and
  :func:`~isaaclab.sim.spawn_deformable_body_material` from :mod:`isaaclab.sim`
  instead of :mod:`isaaclab_physx.sim`; import
  :class:`~isaaclab.assets.DeformableObjectCfg` from :mod:`isaaclab.assets`
  instead of :mod:`isaaclab_physx.assets`.
* Changed PhysX deformable API documentation to direct users to the
  backend-neutral :mod:`isaaclab.assets` and :mod:`isaaclab.sim` imports.

Fixed
^^^^^

* Fixed :class:`~isaaclab_physx.assets.DeformableObject` state writer methods
  to accept ``ProxyArray`` inputs without requiring manual conversion.
