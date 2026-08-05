Fixed
^^^^^

* Fixed body-frame external wrenches (``set_external_force_and_torque(..., is_global=False)``) being
  applied unrotated on the Newton backend. The composed wrench was written directly into Newton's
  world-frame ``body_f`` buffer without a body-to-world rotation, so forces and torques requested in
  a body frame acted in the wrong direction for any non-axis-aligned body. The wrench is now composed
  into the world frame before being written. Affects :class:`~isaaclab.assets.Articulation`,
  :class:`~isaaclab.assets.RigidObject`, and :class:`~isaaclab.assets.RigidObjectCollection` on the
  Newton backend.
