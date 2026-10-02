* Added ``env_mask`` to :meth:`~isaaclab.actuators.ActuatorCollection.reset` and
  :meth:`~isaaclab.actuators.ActuatorControl.reset_native_actuators`. Isaac Lab actuator models and backend-native
  actuator state reset only the masked environments from the mask itself, without converting it to indices.
* Added ``batch_mask`` to :meth:`~isaaclab.utils.buffers.DelayBuffer.set_time_lag` and
  :meth:`~isaaclab.utils.buffers.DelayBuffer.reset`, which select batches on the device without synchronizing the
  host. Tensor lags set through a mask are full-sized and are not range-checked.
* Added ``env_mask`` to :meth:`~isaaclab.assets.BaseCableObject.reset`, matching the other scene assets.
* Added :class:`~isaaclab.utils.seed.WarpRng`, the per-environment Warp random number generator state that
  environments and sensors share, one per process. :func:`~isaaclab.utils.seed.configure_seed` reseeds it in place,
  and environments call :meth:`~isaaclab.utils.seed.WarpRng.initialize` at construction.
