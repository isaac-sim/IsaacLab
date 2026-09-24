Changed
^^^^^^^

* **Breaking:** Changed the ``Isaac-Stack-Cube-SO101-IK-Abs-v0`` teleop pipeline to the
  consolidated engage-relative :class:`~isaacteleop.retargeters.SO101ClutchRetargeter` from Isaac
  Teleop 1.5. The clutch now
  rebases the full end-effector pose instead of position only, and engaging additionally requires
  the controller squeeze -- operators must hold the grip button to drive the arm and release it to
  re-clutch. Removed the ``orientation_offset`` calibration argument, which no longer exists: the
  commanded home orientation now comes from the task's configured home transform. That orientation
  still carries a ``TODO(measure-in-sim)`` and must be measured from
  ``ee_frame.data.target_pose_source`` before the task commands the correct gripper orientation.
  Requires ``isaacteleop`` 1.5 or newer. Against 1.4 the constructor call is still accepted and
  silently runs the previous position-only behaviour, so check the installed version if
  orientation control appears unchanged.
