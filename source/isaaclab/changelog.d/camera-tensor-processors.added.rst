* Added ``CameraCfg.tensor_processors``, which compute camera outputs from render outputs and earlier
  processor outputs after each capture, and reset and close with the camera.
* Added :class:`~isaaclab.utils.tensor_processors.TensorProcessorCallInfo`, which cameras pass to processors as
  ``info``. Its ``env_mask`` identifies the environments due for each capture. Asynchronous rendering sets it to
  None because the current capture mask does not describe the published image.
