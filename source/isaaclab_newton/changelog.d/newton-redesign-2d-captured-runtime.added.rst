* Added :class:`~isaaclab_newton.physics.CapturedGraph`, which retained Torch and Warp graph allocations and replayed
  mixed work on one stream through :meth:`~isaaclab_newton.physics.CapturedGraph.launch`.
* Added ``NewtonSolver.captures_model_changes`` and an optional ``env_mask`` to
  :meth:`~isaaclab_newton.physics.NewtonBackend.mark_model_changed` and
  :meth:`~isaaclab_newton.physics.NewtonManager.add_model_change`. MJWarp recorded conditional solver refreshes that
  skipped edits whose masks selected no world without a host synchronization.
* Added ``NewtonBackend.live`` to identify step and graph-recording intervals in which derived asset data needed
  fresh computation instead of timestamp caches.
