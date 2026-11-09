* Fixed ``pos_w`` and ``quat_w`` of the Newton contact sensor returning never-written zeros of the wrong shape when
  ``track_pose`` was enabled. Newton does not track the sensor pose, so reading them now raises
  ``NotImplementedError`` as documented, and the configuration warns about ``track_pose``.
