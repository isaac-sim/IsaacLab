* Fixed :meth:`~isaaclab_newton.assets.Articulation.reset` with only ``env_mask`` resetting the actuator state
  (delay buffers, network history, and Newton-native actuator state) of every environment.
* Fixed :meth:`~isaaclab_newton.assets.RigidObject.reset` and
  :meth:`~isaaclab_newton.assets.RigidObjectCollection.reset` ignoring ``env_mask`` and clearing the external
  wrenches of every environment.
* Fixed the ``write_body_*_to_sim_mask`` methods of :class:`~isaaclab_newton.assets.RigidObjectCollection`
  converting masks to indices on the host, which prevented capturing them in CUDA graphs as documented. They now
  write through masked kernels.
