Added
^^^^^

* Added XR camera feedback to ``teleop_se3_agent.py`` and ``record_demos.py``, with task-configured
  existing-camera selection, declarative layouts, and viewer-start, head-locked, or explicit-world
  placement.
* Added a PiP-owned CUDA Replicator source with staged camera-buffer fallback so camera feedback works
  with CPU physics without changing the core camera output device.
* Added optional feed-local DLSS Ray Reconstruction and execution-mode settings, applied by the
  PiP adapter without extending the core camera or renderer configuration APIs.
* Added lazy Kit Scene UI loading so kitless teleoperation warns and continues without PiP.
* Added :class:`~isaaclab_teleop.XrCameraFeedSession` as the supported camera-feedback lifecycle
  API for teleoperation entry points.

Changed
^^^^^^^

* Changed ``--disable_external_cameras`` into the master camera-rendering and PiP gate for
  ``teleop_se3_agent.py`` and ``record_demos.py``. To keep task cameras enabled without PiP, leave
  this flag unset and configure ``xr_camera_feeds`` as an empty list or with every feed disabled.

Fixed
^^^^^

* Fixed camera feedback so it refreshes immediately after environment resets.
* Fixed pre-6.1 PiP compatibility by falling back to classic DLSS when a selected feed requests
  Ray Reconstruction on a runtime without responsive denoising.
