* Added ``CameraCfg.tensor_processors``, which compute camera outputs from render outputs and earlier
  processor outputs after each capture, and reset and close with the camera.
* Added :class:`~isaaclab.utils.tensor_processors.TensorProcessorArgs`, the base for a processor call's arguments
  besides its data, and :class:`~isaaclab.sensors.SensorTensorProcessorArgs`, which cameras pass to processors as
  ``args``. Its ``env_mask`` identifies the environments due for each capture. Asynchronous rendering sets it to
  None because the current capture mask does not describe the published image.
