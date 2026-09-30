* Fixed ``body_com_acc_w`` of :class:`~isaaclab_newton.assets.Articulation`,
  :class:`~isaaclab_newton.assets.RigidObject`, and :class:`~isaaclab_newton.assets.RigidObjectCollection`
  dividing by the physics time step when an update spans several physics steps, such as when Newton owns
  the decimation loop. The finite difference now uses the elapsed time since the previous update.
