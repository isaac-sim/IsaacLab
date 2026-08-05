Added
^^^^^

* Added recorded ``robot_pov_cam`` policy observations to the GR1T2 pick-place and G1
  locomanipulation tasks for XR camera feedback.

Fixed
^^^^^

* Fixed the ExhaustPipe ``robot_pov_cam`` rotation after the WXYZ-to-XYZW quaternion migration.
* Fixed the GR1T2 and G1 XR cameras to follow their robot attachment points and restored
  the calibrated G1 camera view with head-locked panel placement.
