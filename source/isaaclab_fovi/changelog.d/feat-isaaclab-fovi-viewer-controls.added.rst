* Added gaze and foveation controls to the Newton viewer's side panel for every foveated camera
  (``viewer_gaze_controls``), with sliders kept in sync with the camera's gaze and a discrete, log-spaced
  ``cmf_a`` selector (``viewer_cmf_a_values``).
* Added :meth:`~isaaclab_fovi.sensors.FoveatedCamera.set_cmf_a`, which changes the CMF constant of fixed-count
  Newton foveated cameras at runtime.
