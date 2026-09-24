Added
^^^^^

* Added opt-in ``XrCameraFeedLayoutCfg.use_scene_partition`` to exclude shared SceneUI
  from robot cameras using their existing environment partitions. The unpartitioned
  XR camera retained its all-partitions spectator view. Only the ``/ui`` partition
  was temporarily authored and restored after the final isolated panel closed;
  camera configuration and global renderer settings remained unchanged.

Fixed
^^^^^

* Fixed head-locked XR camera panels to follow the complete display pose.
* Hid panels during tracking loss or partition conflicts and restored them after recovery.
* Fixed first-panel startup to establish the SceneUI partition before waiting for panel visibility.
* Refreshed partition inheritance when SceneUI children appeared, preventing recursive camera feeds after startup
  or stage replacement without rewriting the partition every frame.
* Preserved positional construction of ``XrCameraFeedLayoutCfg``.
* Preserved requested PiP denoising settings without importing Isaac Sim during configuration preparation.
