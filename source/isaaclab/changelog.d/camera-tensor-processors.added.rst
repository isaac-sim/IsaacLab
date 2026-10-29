* Added ``CameraCfg.tensor_processors``, which compute camera outputs from render outputs and earlier
  processor outputs after each published capture, and reset and close with the camera.
* Added the reserved ``env_mask`` input for ``CameraCfg.tensor_processors``, a boolean mask of the environments
  with a new capture, so stateful processors can skip rows that still hold the previous capture when
  ``update_period`` staggers captures. Processors reading it need synchronous rendering.
