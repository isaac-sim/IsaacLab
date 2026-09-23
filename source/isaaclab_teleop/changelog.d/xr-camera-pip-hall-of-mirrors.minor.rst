Added
^^^^^

* Added opt-in ``XrCameraFeedLayoutCfg.use_scene_partition`` to isolate shared SceneUI
  and the XR presentation camera from robot-camera rendering. Enabled PiP preparation
  temporarily disables per-environment partitioning for selected Isaac RTX
  cameras and owns ``showAllPartitionsByDefault=False`` until the final isolated session
  closes. Additional cameras must disable per-environment partitioning; conflicting
  renderer visibility overrides are rejected. Custom launchers must close prepared
  sessions even when environment construction fails. Other tasks and non-XR runs
  retain their defaults.

Fixed
^^^^^

* Fixed head-locked XR camera panels to follow the complete display pose.
* Hid panels during tracking loss or partition conflicts and restored them after recovery.
* Fixed first-panel startup to establish the SceneUI partition before waiting for panel visibility.
* Refreshed partition inheritance when SceneUI children appeared, preventing recursive camera feeds after startup
  or stage replacement without rewriting the partition every frame.
* Preserved positional construction of ``XrCameraFeedLayoutCfg``.
* Preserved requested PiP denoising settings without importing Isaac Sim during configuration preparation.
