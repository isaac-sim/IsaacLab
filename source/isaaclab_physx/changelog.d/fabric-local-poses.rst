Added
^^^^^

* Added Fabric-accelerated ``get_local_poses`` / ``set_local_poses`` to
  :class:`~isaaclab_physx.sim.views.FabricFrameView`.

  Local-pose operations now use ``wp.indexedfabricarray`` to read/write
  ``omni:fabric:localMatrix`` directly on the GPU, propagating between
  parent world matrices and child local/world matrices via Warp kernels
  without round-tripping through USD.

* Added per-view dirty tracking: ``set_local_poses`` marks the world matrix
  dirty and vice-versa, triggering automatic re-propagation on the next read.

* Added ``_rebuild_fabric_arrays`` topology-change recovery.
