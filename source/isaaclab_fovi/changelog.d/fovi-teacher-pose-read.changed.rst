* Changed the gaze teacher to read the camera's pose without capturing an image. It reads
  :attr:`~isaaclab_fovi.sensors.FoveatedCamera.mount_pos_w`, which
  :meth:`~isaaclab_fovi.sensors.FoveatedCamera.update_mount` refreshes, where reading the camera's data rendered a
  frame for every label, also between the frames a policy observes.
