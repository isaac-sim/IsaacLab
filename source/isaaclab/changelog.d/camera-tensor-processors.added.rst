* Added ``CameraCfg.tensor_processors``, which compute camera outputs from render outputs and earlier
  processor outputs after each capture, and reset and close with the camera.
* Camera processors accept an ``env_mask`` keyword identifying the environments due for each capture.
  Processors require synchronous rendering by default; stateless processors can set ``requires_fresh_capture=False``
  to allow asynchronous rendering, which passes ``env_mask=None``.
