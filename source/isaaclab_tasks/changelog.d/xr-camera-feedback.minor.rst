Added
^^^^^

* Added recorded ``robot_pov_cam`` policy observations to the GR1T2 pick-place and G1
  locomanipulation tasks for XR camera feedback.
* Configured the recorded cameras to produce their RTX pixel output on ``cuda:0`` while preserving
  CPU physics.

Fixed
^^^^^

* Fixed the ExhaustPipe ``robot_pov_cam`` rotation after the WXYZ-to-XYZW quaternion migration.
