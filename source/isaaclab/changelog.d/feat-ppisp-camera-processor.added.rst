* Added ``CameraCfg.tensor_processors``, which run in order after each published capture and compute
  camera outputs from render outputs and the outputs of earlier processors. Processor classes can prepare
  camera prims before scene export through ``prepare_camera``.
