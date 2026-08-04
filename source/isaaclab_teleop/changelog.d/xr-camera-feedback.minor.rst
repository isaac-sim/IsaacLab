Added
^^^^^

* Added XR camera feedback to ``teleop_se3_agent.py`` and ``record_demos.py``, with task-configured
  existing-camera selection, declarative layouts, and viewer-start, head-locked, or explicit-world
  placement.
* Added a PiP-owned CUDA Replicator source with staged camera-buffer fallback so camera feedback works
  with CPU physics without changing the core camera output device.
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

* Fixed repeated IWER reset messages and refreshed camera feedback immediately after resets.
