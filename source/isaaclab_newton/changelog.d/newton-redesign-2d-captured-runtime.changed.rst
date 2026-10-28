* **Breaking:** Changed ``newton_backend.capture_graph`` to return
  :class:`~isaaclab_newton.physics.CapturedGraph`; replay its result with ``graph.launch()`` instead of
  :func:`warp.capture_launch`, and retain it for the graph's lifetime.
* Changed backend steps and sensor queries to record work directly into an enclosing CUDA graph, including BVH
  refits whose freshness could not depend on host timestamps during replay.
* Changed :meth:`~isaaclab_newton.physics.NewtonBackend.invalidate_fk` to map articulation-view rows to worlds through
  ``model.articulation_world``, so views over a subset of worlds flagged the right worlds and global articulations
  requested forward kinematics without resetting the global world.
* Changed derived asset properties to recompute inside a step or graph recording instead of raising during capture.
* Changed queued model-property edits to reach the solver before state reads and
  :meth:`~isaaclab_newton.physics.NewtonManager.forward`, as well as before the next step.
