* Fixed the OVPhysX :class:`~isaaclab_ov.sensors.ray_caster.RayCaster`,
  :class:`~isaaclab_ov.sensors.ray_caster.RayCasterCamera`, :class:`~isaaclab_ov.sensors.ray_caster.MultiMeshRayCaster`
  and :class:`~isaaclab_ov.sensors.ray_caster.MultiMeshRayCasterCamera` reporting other environments' frames and
  tracked meshes on GPU when the scene clones several prototypes, for example through
  :class:`~isaaclab.sim.spawners.wrappers.wrappers_cfg.MultiAssetSpawnerCfg`. Their bindings now follow the numeric
  environment order, like the other OVPhysX tensor bindings.
