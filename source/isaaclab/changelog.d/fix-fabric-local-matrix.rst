Added
^^^^^

* Added :func:`~isaaclab.utils.warp.fabric.decompose_indexed_fabric_transforms`
  and :func:`~isaaclab.utils.warp.fabric.compose_indexed_fabric_transforms`
  Warp kernels.  They mirror the existing
  ``decompose_fabric_transformation_matrix_to_warp_arrays`` /
  ``compose_fabric_transformation_matrix_from_warp_arrays`` kernels but
  operate on :class:`wp.indexedfabricarray`, so the view-to-fabric mapping
  is baked into the array and the kernel just dereferences
  ``ifa[view_index]`` instead of taking a separate ``mapping`` argument.
* Added :func:`~isaaclab.utils.warp.fabric.update_indexed_local_matrix_from_world`
  and :func:`~isaaclab.utils.warp.fabric.update_indexed_world_matrix_from_local`
  Warp kernels that propagate ``local = world * inv(parent)`` and
  ``world = local * parent`` directly on Fabric storage matrices (no
  explicit transposes).  Used by
  :class:`~isaaclab_physx.sim.views.FabricFrameView` to keep child world and
  local matrices consistent across writes without round-tripping through USD.
* Added :meth:`~isaaclab.sim.SimulationContext.get_service` and
  :meth:`~isaaclab.sim.SimulationContext.set_service` — a typed singleton
  service locator on :class:`~isaaclab.sim.SimulationContext`.  Backend-specific
  caches (e.g. Fabric hierarchy handles) register themselves here instead of
  living as class-level globals.  Services are automatically cleared on
  :meth:`~isaaclab.sim.SimulationContext.clear_instance`.
