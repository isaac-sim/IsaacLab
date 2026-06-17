Added
^^^^^

* Added :mod:`isaaclab_ovphysx.sensors.ray_caster` with
  :class:`~isaaclab_ovphysx.sensors.ray_caster.RayCaster`,
  :class:`~isaaclab_ovphysx.sensors.ray_caster.RayCasterCamera`,
  :class:`~isaaclab_ovphysx.sensors.ray_caster.MultiMeshRayCaster`, and
  :class:`~isaaclab_ovphysx.sensors.ray_caster.MultiMeshRayCasterCamera`.
  Mirrors :mod:`isaaclab_physx.sensors.ray_caster` structure: a single
  ``_OvPhysxRayCasterMixin`` carries the backend-specific pose-tracking
  surface, reading body poses via the ovphysx
  ``create_tensor_binding(pattern=..., tensor_type=RIGID_BODY_POSE)``
  API. Static (non-physics) sensor frames fall back to a one-time USD
  pose snapshot. Unblocks ``Isaac-Velocity-Rough-Anymal-D-v0`` (the
  height_scanner now dispatches under OVPhysX).
