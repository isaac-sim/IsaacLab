Fixed
^^^^^

* Fixed the Kuka Allegro wrist camera rendering from its reset pose for the whole episode in the
  ``duo_camera`` presets. The camera is mounted on the palm, so ``update_latest_camera_pose`` is now
  enabled and its rendered view follows the arm.
