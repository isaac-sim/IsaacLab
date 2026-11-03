* Added ``CameraCfg.tensor_processors``, which compute camera outputs from render outputs and earlier
  processor outputs after each published capture, and reset and close with the camera. Processors can read
  the reserved ``env_mask`` input, which marks the environments due for each capture when ``update_period``
  staggers them, matching the published image with asynchronous rendering.
* Added :class:`~isaaclab.renderers.CaptureInfo`, which describes a capture that a renderer published.
