* Added ``CameraCfg.tensor_processors``, which compute camera outputs from render outputs and earlier
  processor outputs after each capture, and reset and close with the camera. Processors can read
  the reserved ``CAPTURE_ENV_MASK_NAME`` input, which marks the environments due for each capture when ``update_period``
  staggers them; processors reading it need synchronous rendering.
