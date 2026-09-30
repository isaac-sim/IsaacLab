* Added ``env_mask`` to :meth:`~isaaclab.actuators.ActuatorCollection.reset`. Isaac Lab actuator models reset only
  the masked environments, converting the mask to indices once for all groups, and backend-native actuator state
  resets from the mask through :meth:`~isaaclab.actuators.ActuatorControl.reset_native_actuators_mask`.
* Added ``env_mask`` to :meth:`~isaaclab.assets.BaseCableObject.reset`, matching the other scene assets.
