Added
^^^^^

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
