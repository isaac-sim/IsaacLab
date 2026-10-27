* Fixed the SO-101 stacking pipeline for Isaac Capture 1.7 by removing the retired
  ``orientation_offset`` argument and calibration constant. Orientation follows the SDK's
  engage-relative clutch model seeded from the configured home transform, with the existing
  eight-element pose and analog-gripper action order preserved.
