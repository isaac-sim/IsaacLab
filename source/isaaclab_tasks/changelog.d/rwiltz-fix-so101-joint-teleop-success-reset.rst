Fixed
^^^^^

* Fixed the ``IsaacContrib-Stack-Cube-SO101-Joint-Teleop-v0`` environment not resetting after a
  successful cube stack. The success termination used a 100 µrad gripper-open tolerance
  (``atol=0.0001``) that the follower cannot reach when the leader arm's raw encoder angle does
  not exactly match ``SO101_GRIPPER_OPEN`` due to calibration offsets or joint soft-limit ceilings.
  The tolerance is now set to ``gripper_threshold`` (0.2 rad) in the joint-teleop env, consistent
  with the threshold used elsewhere to classify the gripper as open vs. closed.
