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

* Added Fabric-accelerated ``get_local_poses`` / ``set_local_poses`` to
  :class:`~isaaclab_physx.sim.views.FabricFrameView`.

  Local-pose operations now use ``wp.indexedfabricarray`` to read/write
  ``omni:fabric:localMatrix`` directly on the GPU, propagating between
  parent world matrices and child local/world matrices via Warp kernels
  without round-tripping through USD.

* Added lazy per-view dirty tracking: ``set_local_poses`` marks the world
  matrix dirty and vice-versa, triggering automatic re-propagation only on
  the next read (no eager kernel launches on the write path).

* Added interleave detection: interleaving ``set_world_poses`` and
  ``set_local_poses`` on the same view within a frame flushes the stale
  direction automatically and emits a one-time performance warning.

* Added topology-change recovery via automatic ``PrepareForReuse`` detection
  and per-selection index rebuild.

Deprecated
^^^^^^^^^^

* Deprecated ``get_scales`` / ``set_scales`` on all ``BaseFrameView`` subclasses.
  Use the new explicit ``get_local_scales`` / ``set_local_scales`` (operates on
  ``xformOp:scale`` / ``localMatrix``) or ``get_world_scales`` /
  ``set_world_scales`` (operates on composed world-space scale) instead.
  The deprecated methods still work but emit a ``DeprecationWarning``;
  ``UsdFrameView`` defaults to local, ``FabricFrameView`` defaults to world
  (preserving prior behavior).
