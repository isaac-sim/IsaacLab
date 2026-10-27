* Added ``CameraCfg.tensor_processors``, which compute camera outputs from render outputs and earlier
  processor outputs after each published capture. Processor classes can prepare camera prims before scene
  export through ``prepare_camera``.
* Added the reserved ``env_mask`` input for ``CameraCfg.tensor_processors``, a boolean mask of the environments
  with a new capture, so stateful processors can skip rows that still hold the previous capture when
  ``update_period`` staggers captures.
