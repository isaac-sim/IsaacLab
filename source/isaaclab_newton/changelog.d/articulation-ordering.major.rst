Added
^^^^^

* Added backend joint/body ordering introspection properties to
  :class:`~isaaclab_newton.assets.articulation.Articulation`.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.register_post_step_callback`
  for hooks that must run inside the stepped (and CUDA-graph-captured) region
  after the last solver substep. Articulations with a non-identity ordering use
  it to publish backend-order state to their public-order buffers every step.
