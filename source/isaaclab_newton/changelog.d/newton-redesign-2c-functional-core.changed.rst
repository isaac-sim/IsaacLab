* **Breaking:** Removed ``NewtonQueries``. Refit shared BVHs once per state with :func:`newton_backend.refit_bvh
  <isaaclab_newton.physics.newton_backend.refit_bvh>` and capture queries with :func:`newton_backend.capture_graph
  <isaaclab_newton.physics.newton_backend.capture_graph>`.
* **Breaking:** Moved ``NewtonManager.transforms_may_change_on_graph_replay`` to
  :attr:`NewtonBackend.transforms_may_change_on_graph_replay
  <isaaclab_newton.physics.NewtonBackend.transforms_may_change_on_graph_replay>`.
* Changed :class:`~isaaclab_newton.physics.NewtonManager` to drive its backend through the ``newton_backend``
  functions. The backend holds the step callbacks, solver-init callbacks, pending model changes and the captured
  step, so a hard reset discards them with the model.
