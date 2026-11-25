* Added gaze and foveation controls for foveated cameras to the Newton viewer, below the Camera Source selector while
  a foveated camera is the window's source, with sliders kept in sync with the camera's gaze and a discrete,
  log-spaced ``cmf_a`` selector (:attr:`~isaaclab_fovi.sensors.ViewerGazeControls.cmf_a_values`).
* Added :meth:`~isaaclab_fovi.sensors.FoveatedCamera.set_cmf_a`, which changes the CMF constant of fixed-count
  Newton foveated cameras at runtime.
