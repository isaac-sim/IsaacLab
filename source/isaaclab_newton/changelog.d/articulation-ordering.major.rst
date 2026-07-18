Added
^^^^^

* Added backend joint/body ordering introspection properties to
  :class:`~isaaclab_newton.assets.Articulation`.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.register_post_step_callback`
  and :meth:`~isaaclab_newton.physics.NewtonManager.unregister_post_step_callback`
  for hooks that must run inside the stepped (and CUDA-graph-captured) region
  after the last solver substep. Articulations with a non-identity ordering use
  them to publish backend-order state to their public-order buffers every step
  and to release the hook when the articulation is destroyed.

Removed
^^^^^^^

* Removed the ``write_joint_state_data_index``, ``write_joint_state_data_mask``,
  ``write_joint_vel_data_index``, and ``write_joint_vel_data_mask`` kernels from
  ``isaaclab_newton.assets.articulation.kernels``. Prefer the public-order asset
  write APIs (:meth:`~isaaclab.assets.Articulation.write_joint_position_to_sim_index`
  and its siblings), which apply the ordering conversion internally. Code that
  works directly with raw solver views can instead launch the public elementwise
  reorder kernels (the ``reorder_2d_user_to_backend`` /
  ``reorder_2d_backend_to_user`` and ``reorder_3d_user_to_backend`` /
  ``reorder_3d_backend_to_user`` family) from
  ``isaaclab.assets.articulation.ordering_kernels`` together with the asset's
  ordering maps.
