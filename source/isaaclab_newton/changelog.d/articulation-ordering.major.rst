Added
^^^^^

* Added backend joint/body ordering introspection properties to
  :class:`~isaaclab_newton.assets.Articulation`.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.register_post_step_callback`
  for hooks that must run inside the stepped (and CUDA-graph-captured) region
  after the last solver substep. Articulations with a non-identity ordering use
  it to publish backend-order state to their public-order buffers every step.

Removed
^^^^^^^

* Removed the ``write_joint_state_data_index``, ``write_joint_state_data_mask``,
  ``write_joint_vel_data_index``, and ``write_joint_vel_data_mask`` kernels from
  ``isaaclab_newton.assets.articulation.kernels``. Use the shared ordering-aware
  kernels ``write_joint_state_user_to_backend_with_indices``,
  ``write_joint_state_user_to_backend_with_mask``,
  ``write_joint_vel_user_to_backend_with_indices``, and
  ``write_joint_vel_user_to_backend_with_mask`` from
  ``isaaclab.assets.articulation.ordering_kernels`` instead.
