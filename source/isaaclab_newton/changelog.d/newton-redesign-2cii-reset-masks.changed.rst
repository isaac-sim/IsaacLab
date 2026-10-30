* Changed :meth:`NewtonBackend.invalidate_fk <isaaclab_newton.physics.NewtonBackend.invalidate_fk>` and
  :meth:`NewtonBackend.invalidate_body_state <isaaclab_newton.physics.NewtonBackend.invalidate_body_state>` to select
  environments by mask only. :class:`~isaaclab_newton.physics.NewtonManager` converts ``env_ids`` from asset writes
  with :meth:`NewtonBackend.mask_from_ids <isaaclab_newton.physics.NewtonBackend.mask_from_ids>`, and the id-indexed
  reset kernels are removed.
