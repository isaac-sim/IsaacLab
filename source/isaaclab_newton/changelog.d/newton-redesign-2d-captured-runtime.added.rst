* Added :class:`~isaaclab_newton.physics.CapturedGraph`, returned by ``newton_backend.capture_graph``, which records
  Torch and Warp work on one stream and keeps both libraries' allocations alive.
* Added ``SolverSpec.captures_model_changes`` and an optional ``env_mask`` to
  :meth:`~isaaclab_newton.physics.NewtonManager.add_model_change`. MJWarp records the solver refresh into a
  conditional CUDA graph, so an edit whose mask selects no world skips the refresh without a host synchronization.
* Added ``NewtonBackend.live``, true inside a step and while a CUDA graph is recorded, when derived asset data is
  recomputed on read instead of taken from timestamp caches.
