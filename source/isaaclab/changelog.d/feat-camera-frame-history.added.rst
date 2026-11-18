* Added :attr:`~isaaclab.sensors.camera.CameraCfg.frame_history_length` and the optional
  :attr:`~isaaclab.sensors.camera.CameraData.frame_history` / ``frame_history_mask`` camera outputs.
  Renderers that capture one image per update are pushed into the history by the camera;
  a renderer that predicts several frames per update sets
  :attr:`~isaaclab.renderers.BaseRenderer.provides_frame_history` and fills it itself.
