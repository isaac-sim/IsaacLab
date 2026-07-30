Added
^^^^^

* Added XR camera feedback to ``teleop_se3_agent.py`` and ``record_demos.py``, with task-configured
  existing-camera selection, declarative layouts, and viewer-start, head-locked, or explicit-world
  placement.
* Added direct and staged CUDA image presentation paths so camera feedback works with CPU physics.

Changed
^^^^^^^

* Changed Kit Scene UI loading to be lazy so kitless teleoperation warns and continues without PiP.
* Changed ``--disable_external_cameras`` into the master camera-rendering and PiP gate for
  ``teleop_se3_agent.py`` and ``record_demos.py``.

Fixed
^^^^^

* Fixed repeated IWER reset messages and refreshed camera feedback immediately after resets.
