* **Breaking:** Changed ``newton_backend.capture_graph`` to return :class:`~isaaclab_newton.physics.CapturedGraph`
  instead of :class:`warp.Graph`. Replace ``wp.capture_launch(graph)`` with ``graph.launch()`` and keep the returned
  object alive while it is replayed.
* Changed backend steps, BVH refits and sensor queries called inside an enclosing CUDA graph capture to record into
  it instead of starting their own capture. Recorded BVH refits run on every replay.
* Changed :meth:`~isaaclab_newton.physics.NewtonBackend.invalidate_fk` to map articulation-view rows to worlds through
  ``model.articulation_world``: views over a subset of worlds flag the right worlds, and global articulations get FK
  without resetting the global world.
* Changed derived asset properties to recompute inside a step or capture instead of raising during capture.
* Changed queued model-property edits to reach the solver before state reads and
  :meth:`~isaaclab_newton.physics.NewtonManager.forward`, not only before the next step. Fixed-root pose writes queue
  their edit instead of notifying the solver immediately.
