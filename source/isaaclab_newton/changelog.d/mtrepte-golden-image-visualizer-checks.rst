Fixed
^^^^^

* Fixed stale ``_scene_data_mapping`` in :meth:`~isaaclab_newton.physics.NewtonManager.update_visualization_state`
  being reused after the visualization model was rebuilt for a stage with a different body count
  (e.g. switching from a 4-env tiled capture to a 1-env viewport capture within the same process).
  The mapping is now invalidated when its length does not match the current model's body count,
  preventing wrong body transforms from being written into the shadow ``state_0``.

* Fixed :meth:`~isaaclab_newton.physics.NewtonManager.sync_transforms_to_usd` caching a partial
  ``SelectPrims`` result when only some body prims had the ``newton:index`` Fabric attribute
  propagated to the GPU at first call (async propagation), causing subsequent writes to miss
  prims whose attribute arrived later and leaving those bodies invisible.  The per-call
  ``SelectPrims`` now runs unconditionally every frame; a new ``_newton_fabric_ready`` flag is
  set once the first successful write completes, and the dirty flag is kept True until then so
  retries succeed without requiring external intervention.
