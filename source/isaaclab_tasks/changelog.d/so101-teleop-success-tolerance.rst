Fixed
^^^^^

* Fixed the SO-101 joint-teleop cube-stack task often failing to auto-reset after a completed
  stack. The success termination accepted the gripper as open only within 0.2 rad of
  ``SO101_GRIPPER_OPEN``, which is the top 0.2 rad of the jaw's 1.92 rad range, so a leader arm
  whose calibrated full-open reading fell short never triggered success. The tolerance is now
  0.5 rad, which still requires more opening than releasing a cube needs.
