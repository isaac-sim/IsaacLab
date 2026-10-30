* Added :class:`~isaaclab_newton.physics.CapturedGraph`, returned by ``newton_backend.capture_graph``. It records
  Torch and Warp work on one stream, keeps both libraries' graph allocations alive, and replays with
  :meth:`~isaaclab_newton.physics.CapturedGraph.launch`.
* Added an optional ``env_mask`` to :meth:`~isaaclab_newton.physics.NewtonManager.add_model_change` and
  ``NewtonSolver.captures_model_changes``. MJWarp records the solver refresh as a conditional CUDA graph that skips
  edits whose masks select no world, without a host synchronization.
* Added ``NewtonBackend.live``, true while a step runs or a CUDA graph is recorded; derived asset data then recomputes
  on read instead of using its timestamp cache.
