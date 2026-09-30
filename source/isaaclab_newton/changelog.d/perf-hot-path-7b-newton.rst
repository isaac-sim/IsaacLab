Changed
^^^^^^^

* Removed the redundant per-step solver-internal reset from :meth:`~isaaclab_newton.physics.NewtonManager.step`.
  The reset still runs through :meth:`~isaaclab_newton.physics.NewtonManager.forward` for worlds flagged by
  state writes.
* Skipped the device-to-host readback in joint position-limit writes of
  :class:`~isaaclab_newton.assets.Articulation` when the clamped-default message would not be logged.

Fixed
^^^^^

* Fixed ``body_com_acc_w`` of :class:`~isaaclab_newton.assets.Articulation`,
  :class:`~isaaclab_newton.assets.RigidObject`, and :class:`~isaaclab_newton.assets.RigidObjectCollection`
  dividing by the physics time step when an update spans several physics steps, such as when Newton owns
  the decimation loop. The finite difference now uses the elapsed time since the previous update.
