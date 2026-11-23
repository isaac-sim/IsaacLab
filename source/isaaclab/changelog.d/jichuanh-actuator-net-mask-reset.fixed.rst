* Fixed :meth:`~isaaclab.actuators.ActuatorCollection.reset` of a CUDA collection failing for an ``env_mask`` on the
  host; the mask is copied to the collection's device once.
