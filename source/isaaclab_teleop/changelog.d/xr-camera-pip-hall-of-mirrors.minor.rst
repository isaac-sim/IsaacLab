Added
^^^^^

* Added opt-in ``XrCameraFeedLayoutCfg.use_scene_partition`` to isolate shared SceneUI
  and the XR presentation camera from robot-camera rendering. This requires Kit XR
  scene-partition propagation and runtime-updated mesh bounds fixes. Enabled PiP
  preparation temporarily disables per-environment partitioning for selected Isaac RTX
  cameras and owns ``showAllPartitionsByDefault=False`` until the final isolated session
  closes. Additional cameras must disable per-environment partitioning; conflicting
  renderer visibility overrides are rejected. Custom launchers must close prepared
  sessions even when environment construction fails. Other tasks and non-XR runs
  retain their defaults. A released-runtime minimum is not yet established; validation
  currently requires a local Kit build containing both fixes.

Fixed
^^^^^

* Fixed head-locked XR camera panels to follow the complete display pose.
* Hid panels during tracking loss or partition conflicts and restored them after recovery.
* Preserved positional construction of ``XrCameraFeedLayoutCfg``.
