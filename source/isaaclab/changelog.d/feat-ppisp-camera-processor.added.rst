* Added ``CameraCfg.tensor_processors``, which compute camera outputs from render outputs and earlier
  processor outputs after each published capture. Processor classes can prepare camera prims before scene
  export through ``prepare_camera``.
