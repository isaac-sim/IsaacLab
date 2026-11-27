* Added physical, hybrid and dual (experimental) gaze to :class:`~isaaclab_fovi.mdp.GazeActionCfg` through its
  ``mode`` and :class:`~isaaclab_fovi.mdp.GazeRigCfg`: the gaze actions turn a rig's joints, the retina while the
  rig follows, or both independently.
* Added :data:`~isaaclab_fovi.assets.PAN_TILT_CFG`, a two-joint pan-tilt mount for a camera.
* Added :meth:`~isaaclab_fovi.sensors.FoveatedCamera.update_mount`, which reads the camera's mount before a lazily
  updated camera renders; hybrid gaze and the tracked-mount teacher aim from it.
* Added :attr:`~isaaclab_fovi.sensors.FoveatedCamera.gaze_writes`, which counts
  :meth:`~isaaclab_fovi.sensors.FoveatedCamera.set_gaze_angles` calls, so hybrid gaze starts a new fixation after a
  gaze set from elsewhere.
