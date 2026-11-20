* Added :attr:`~isaaclab.sensors.camera.CameraCfg.frame_history_length` and the optional
  :attr:`~isaaclab.sensors.camera.CameraData.frame_history` / ``frame_history_mask`` camera outputs, which keep
  the most recent captures of every requested output.
  Renderers that capture one image per update are pushed into the history by the camera;
  a renderer that predicts several frames per update fills it itself and sets ``provides_frame_history``
  on its configuration.
