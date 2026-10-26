* Added :attr:`~isaaclab.sensors.camera.CameraCfg.tensor_processors` to compute camera outputs from render
  outputs after each capture. The camera requests processor inputs, such as ``rgb_hdr``, from the renderer
  without publishing them, and publishes the names in ``data_types``. Configuration validation rejects
  processor outputs that are neither published nor read by a later processor. Processor classes can author
  camera prim settings before the scene is cloned or exported through an optional ``prepare_camera`` class
  method.
