* Changed backend steps, BVH refits and sensor queries called inside an enclosing CUDA graph capture to record into it
  instead of starting or replaying their own graph. Recorded BVH refits ran on every replay.
* Changed :meth:`~isaaclab_newton.physics.NewtonBackend.invalidate_fk` to map articulation-view rows to worlds through
  ``model.articulation_world``, so views over a subset of worlds flagged the right worlds and global articulations got
  FK without resetting the global world.
* Changed derived asset properties to recompute inside a step or capture instead of raising during capture.
* Changed fixed-root pose writes to queue their model-property edit instead of notifying the solver immediately.
