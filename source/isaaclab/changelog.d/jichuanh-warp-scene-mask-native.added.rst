* Added ``env_mask`` to :meth:`~isaaclab.actuators.ActuatorCollection.reset`. Isaac Lab actuator models reset only
  the masked environments from the mask itself, without converting it to indices, and backend-native actuator state
  resets from the mask through :meth:`~isaaclab.actuators.ActuatorControl.reset_native_actuators_mask`.
* Added boolean-mask selection to :meth:`~isaaclab.utils.buffers.DelayBuffer.set_time_lag` and
  :meth:`~isaaclab.utils.buffers.DelayBuffer.reset`, which select batches on the device without synchronizing the
  host. Tensor lags set through a mask are full-sized and are not range-checked.
* Added ``env_mask`` to :meth:`~isaaclab.assets.BaseCableObject.reset`, matching the other scene assets.
