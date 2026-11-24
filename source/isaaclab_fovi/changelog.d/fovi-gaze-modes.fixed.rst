* Fixed :class:`~isaaclab_fovi.mdp.gaze_teacher_labels` rendering a frame for every label, also between the frames
  a policy observes: it read the camera's position from its data, which updates the image. It now reads
  :attr:`~isaaclab_fovi.sensors.FoveatedCamera.mount_pos_w`, which
  :meth:`~isaaclab_fovi.sensors.FoveatedCamera.update_mount` refreshes.
* Fixed the gaze teacher of a tracked mount aiming from the mount of the step before: a lazily updated camera
  records its mount only when its data is read. The teacher now reads it with
  :meth:`~isaaclab_fovi.sensors.FoveatedCamera.update_mount`.
* Fixed :class:`~isaaclab_fovi.mdp.gaze_teacher_labels` raising a ``TypeError`` when its target names every body
  of the asset, as naming a rigid object's only body does, which resolves the body ids to a slice.
