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
  ``world = local * parent`` directly on Fabric storage matrices.

* Added Fabric-accelerated local-pose read/write paths to
  :class:`~isaaclab_physx.sim.views.FabricFrameView`.  Local-pose
  operations now use :class:`wp.indexedfabricarray` to read and write
  ``omni:fabric:localMatrix`` directly on the GPU, propagating between
  parent world matrices and child local/world matrices via Warp kernels
  without round-tripping through USD.

* Added topology-change recovery via automatic ``PrepareForReuse`` detection
  and per-selection index rebuild.

Deprecated
^^^^^^^^^^

* Deprecated ``get_scales`` / ``set_scales`` on ``FabricFrameView``.  For
  reads, use the explicit ``get_local_scales`` (operates on
  ``localMatrix``) or ``get_world_scales`` (composed world-space scale).
  For writes, use the writer scope's ``set_scales``.  The deprecated
  methods still work but emit a ``DeprecationWarning``; ``FabricFrameView``
  defaults to world (preserving prior behavior).
