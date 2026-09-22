Added
^^^^^

* Added opt-in ``XrCameraFeedLayoutCfg.use_scene_partition`` to isolate shared SceneUI
  and the XR presentation camera from robot-camera rendering. This requires Kit XR
  scene-partition propagation and runtime-updated mesh bounds fixes. Enabled PiP
  preparation configures selected Isaac RTX cameras with ``enable_scene_partitioning=False``
  and requests the global renderer setting ``show_all_partitions_by_default=False``.
  Other tasks and non-XR runs retain their defaults.

Fixed
^^^^^

* Fixed head-locked XR camera panels to follow the complete display pose.
