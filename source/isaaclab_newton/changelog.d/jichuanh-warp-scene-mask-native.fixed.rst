* Fixed :meth:`~isaaclab_newton.assets.Articulation.reset` with only ``env_mask`` resetting the actuator state
  (delay buffers, network history, and Newton-native actuator state) of every environment.
* Fixed :meth:`~isaaclab_newton.assets.RigidObject.reset` and
  :meth:`~isaaclab_newton.assets.RigidObjectCollection.reset` ignoring ``env_mask`` and clearing the external
  wrenches of every environment.
