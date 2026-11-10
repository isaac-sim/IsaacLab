* Fixed :meth:`~isaaclab_contrib.assets.Multirotor.reset` rejecting the ``env_mask`` argument of the articulation
  reset. A masked reset now resets the thrusters and thrust targets of the masked environments only.
