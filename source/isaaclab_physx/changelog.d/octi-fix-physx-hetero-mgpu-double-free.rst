Fixed
^^^^^

* Fixed intermittent ``double free or corruption`` (SIGABRT) in
  :class:`~isaaclab_physx.cloner.PhysxReplicateContext` when running fully-heterogeneous
  scenes with one variant per environment on multiple GPUs. Calling
  ``rep.replicate()`` once per source with a single self-target is known to trigger
  native heap corruption under mGPU due to per-call PhysX-internal allocations.
  For layouts where every source maps only to its own environment no cross-env
  replication is needed; the replicator registration is now skipped so PhysX parses
  the source prims directly from the stage.
