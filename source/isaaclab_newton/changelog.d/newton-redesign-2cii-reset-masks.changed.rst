* **Breaking:** Changed :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_fk` and
  :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_body_state` to select environments by mask only and removed
  their ``env_ids`` argument. Pass an environment mask instead, with ``consume_mask=True`` for a scratch mask that the
  invalidation should clear.
* Changed :meth:`NewtonBackend.invalidate_fk <isaaclab_newton.physics.NewtonBackend.invalidate_fk>` and
  :meth:`NewtonBackend.invalidate_body_state <isaaclab_newton.physics.NewtonBackend.invalidate_body_state>` to select
  environments by mask only, and removed the id-indexed reset kernels.
* Changed the ``*_index`` state writes of Newton articulations, rigid objects, rigid object collections and cables to
  flag the written environments inside the write kernel, so invalidating them needed no extra launch.
