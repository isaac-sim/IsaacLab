* Added ``CameraCfg.tensor_processors``, which compute camera outputs from render outputs and earlier
  processor outputs after each capture, and reset and close with the camera.
* Camera processors accept an ``env_mask`` keyword identifying the environments due for each capture.
  Asynchronous rendering passes ``env_mask=None`` because the current capture mask does not describe the
  published image.
