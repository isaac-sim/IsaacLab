* Fixed Newton GL perspective startup with depth-only scene cameras. The automatic selector offered
  cameras supporting all requested channels, while explicit choices were validated at initialization.
* Fixed scene-camera navigation to use live poses after parent-body motion, independently of cached
  measurement poses and ``CameraCfg.update_latest_camera_pose``.
