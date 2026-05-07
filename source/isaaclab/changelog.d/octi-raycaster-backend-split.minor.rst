Added
^^^^^

* Added :class:`~isaaclab.sensors.ray_caster.BaseRayCaster`,
  :class:`~isaaclab.sensors.ray_caster.BaseRayCasterCamera`,
  :class:`~isaaclab.sensors.ray_caster.BaseMultiMeshRayCaster`, and
  :class:`~isaaclab.sensors.ray_caster.BaseMultiMeshRayCasterCamera`
  carrying the backend-agnostic ray-caster logic. Backend subclasses
  override only the body-tracker and target-mesh-tracker hooks.

Changed
^^^^^^^

* :class:`~isaaclab.sensors.ray_caster.RayCaster`,
  :class:`~isaaclab.sensors.ray_caster.RayCasterCamera`,
  :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCaster`, and
  :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCasterCamera` are now
  :class:`~isaaclab.utils.backend_utils.FactoryBase` shims dispatching
  to PhysX / Newton implementations. Cfg surface and runtime semantics
  unchanged.
