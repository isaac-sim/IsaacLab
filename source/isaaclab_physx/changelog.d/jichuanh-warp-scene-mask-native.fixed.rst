* Fixed :meth:`~isaaclab_physx.assets.Articulation.reset` with only ``env_mask`` resetting the actuator state of every
  environment.
* Fixed :meth:`~isaaclab_physx.assets.RigidObject.reset` and
  :meth:`~isaaclab_physx.assets.RigidObjectCollection.reset` ignoring ``env_mask`` and clearing the external
  wrenches of every environment.
