* Changed ``newton_backend.capture_graph`` and ``newton_backend.run_query`` to return
  :class:`~isaaclab_newton.physics.CapturedGraph`, replayed with ``launch()`` instead of :func:`warp.capture_launch`,
  and to record work issued during an outer capture instead of launching graphs inside it.
* Changed ``newton_backend.invalidate_fk`` to map view rows to worlds through the model's ``articulation_world``, so
  views over a subset of worlds flag the right worlds and global articulations request forward kinematics only.
* Changed derived asset data properties such as ``root_link_vel_w`` to recompute when read inside a step or a graph
  capture instead of raising during capture.
* Changed queued model-property changes to reach the solver at :meth:`~isaaclab_newton.physics.NewtonManager.forward`
  and scene-data state reads, not only at the next step.
* Changed state-force callbacks to live on the backend with the other step callbacks, so a hard reset discards them;
  consumers register again on ``PHYSICS_READY``.
* **Breaking:** Removed ``NewtonQueries``; use ``newton_backend.run_query`` and ``newton_backend.capture_graph``.
