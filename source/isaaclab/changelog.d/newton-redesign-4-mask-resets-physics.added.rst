* Added ``env_mask`` to :meth:`~isaaclab.actuators.ActuatorCollection.reset`,
  :meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.reset`, the stateful tensor processors and
  :class:`~isaaclab.utils.noise.NoiseModelWithAdditiveBias`. A mask resets only the selected environments and does
  not synchronize the device.
* Added boolean-mask support to the ``batch_ids`` of :meth:`~isaaclab.utils.buffers.DelayBuffer.reset`: a Torch
  boolean tensor resets the selected batches without synchronizing the device.
