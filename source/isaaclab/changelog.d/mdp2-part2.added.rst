* Added :attr:`~isaaclab.actuators.ActuatorBase.capturable`, which says whether a CUDA graph can replay an actuator's
  compute and masked reset with Torch work inside it. Delayed PD actuators qualify: their masked resets draw delays
  from a device stream, which a replayed graph advances.
