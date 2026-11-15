* Added ``CameraCfg.tensor_processors``, which compute camera outputs from render outputs and earlier
  processor outputs after each capture, and reset and close with the camera.
* Added the ``env_mask`` keyword to :meth:`~isaaclab.utils.tensor_processors.TensorProcessor.__call__`. Cameras pass
  the environments due for each capture when ``update_period`` staggers them, and ``None`` with asynchronous rendering.
