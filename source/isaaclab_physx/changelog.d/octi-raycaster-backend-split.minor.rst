Added
^^^^^

* Added PhysX backend for :class:`~isaaclab.sensors.ray_caster.RayCaster` /
  :class:`~isaaclab.sensors.ray_caster.RayCasterCamera` /
  :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCaster` /
  :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCasterCamera`. Sensor
  body and tracked target meshes both run off ``RigidObjectView`` —
  per-step compose via small warp kernels, no
  :class:`~isaaclab_physx.sim.views.FabricFrameView` path. Static
  parents/targets serve cached per-env ``wp.transformf`` arrays.

Fixed
^^^^^

* Fixed all four ray-caster sensors (:class:`~isaaclab.sensors.ray_caster.RayCaster`,
  :class:`~isaaclab.sensors.ray_caster.RayCasterCamera`,
  :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCaster`,
  :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCasterCamera`) returning
  their spawn-time pose forever when parented under a rigid body. Previous
  path went through :class:`~isaaclab_physx.sim.views.FabricFrameView`
  which regressed in #5179; the new backend reads body pose directly from
  PhysX. The same fix applies to tracked target meshes
  (``track_mesh_transforms=True``) parented under rigid bodies.
* Fixed PhysX tracked target mesh updates to write directly into Warp mesh
  pose tables instead of staging through torch views.
