Added
^^^^^

* Added opt-in ``XrCameraFeedLayoutCfg.use_scene_partition`` to exclude shared SceneUI
  from robot cameras. Preparation disabled environment partitioning and all-partitions
  rendering before camera initialization, keeping the single environment shared while
  the XR camera and SceneUI used a dedicated PiP partition. Prior camera, renderer, and
  session-layer settings were restored after the final owner closed, including failures
  during environment construction.

Fixed
^^^^^

* Fixed head-locked XR camera panels to follow the complete display pose.
* Hid panels during tracking loss or partition conflicts and restored them after recovery.
* Fixed first-panel startup to establish the SceneUI partition before waiting for panel visibility.
* Refreshed partition inheritance when SceneUI children appeared, preventing recursive camera feeds after startup
  or stage replacement without rewriting the partition every frame.
* Preserved positional construction of ``XrCameraFeedLayoutCfg``.
* Preserved requested PiP denoising settings without importing Isaac Sim during configuration preparation.
