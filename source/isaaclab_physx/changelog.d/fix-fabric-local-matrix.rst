Fixed
^^^^^

* Fixed :meth:`~isaaclab_physx.sim.views.FabricFrameView.get_local_poses`
  returning stale USD values after Fabric world-pose writes.  Local poses
  are now read directly from Fabric's ``omni:fabric:localMatrix`` via
  :class:`wp.indexedfabricarray`, and are kept consistent with worldMatrix
  through Warp kernels that propagate either direction on writes.

Changed
^^^^^^^

* Reworked :class:`~isaaclab_physx.sim.views.FabricFrameView` to use three
  persistent ``PrimSelection`` instances (one per access mode), path-based
  view → fabric index mapping (no custom prim attributes), and Warp kernels
  that operate on :class:`wp.indexedfabricarray` so the kernels just index
  ``ifa[view_index]`` instead of taking a separate mapping array.
* Moved the ``IFabricHierarchy`` handle cache out of ``FabricFrameView`` (class-level
  global) into a new :class:`~isaaclab_physx.sim.fabric_stage_cache.FabricStageCache`,
  registered as a service on :class:`~isaaclab.sim.SimulationContext`.  The cache is
  automatically cleared on stage teardown.
* :meth:`~isaaclab_physx.sim.views.FabricFrameView.set_local_poses` now
  writes ``omni:fabric:localMatrix`` directly through Fabric.  The next
  ``get_world_poses`` runs a Warp kernel that recomputes
  ``child_world = parent_world * child_local``.  Symmetrically,
  ``set_world_poses`` runs a kernel that recomputes
  ``child_local = inv(parent_world) * child_world`` so subsequent
  ``get_local_poses`` calls return consistent values.
