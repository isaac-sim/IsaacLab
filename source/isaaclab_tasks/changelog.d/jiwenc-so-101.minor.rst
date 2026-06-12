Added
^^^^^

* Added ``Isaac-Stack-Cube-SO101-v0`` and ``Isaac-Stack-Cube-SO101-IK-Abs-v0`` cube-stacking
  environments for the SO-101 5-DOF arm. The IK-Abs variant uses absolute task-space
  differential inverse kinematics and is seated on the table.
* Generalized the cube-stack MDP gripper observations and terminations to support single-jaw
  grippers in addition to two-finger parallel grippers.
* Added XR teleoperation for ``Isaac-Stack-Cube-SO101-IK-Abs-v0`` via an IsaacTeleop retargeting
  pipeline built from the SO-101 retargeters in ``isaacteleop.retargeters``. Controller poses are
  rebased into the robot base frame upstream via
  :attr:`~isaaclab_teleop.IsaacTeleopCfg.target_frame_prim_path` (set to the seated, +90 deg-yawed
  base prim), so the retargeters work directly in the IK command frame. End-effector motion is
  clutch-rebased around the pose captured on engage: the clutch latches on the first running frame
  (the headset "Play") and seeds its home from a static reset-origin (the gripper's pose in the
  base frame at the seated init pose) so engaging with a steady controller does not move the arm,
  then keeps a running home so a mid-task re-clutch resumes from the last commanded pose. The
  gripper jaw tracks the controller trigger continuously (analog).

Changed
^^^^^^^

* **Breaking:** Reworked the ``Isaac-Stack-Cube-SO101-IK-Abs-v0`` teleop control to command the
  full end-effector SE3 pose. The arm action is now an 8-D
  ``[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper]`` (orientation xyzw),
  replacing the previous 6-D ``[pos_x, pos_y, pos_z, pitch, roll, gripper]``. A single
  full-pose differential IK (``SO101PoseIKController``, ``command_type="pose"``) now solves all
  **5** arm joints (``shoulder_pan``, ``shoulder_lift``, ``elbow_flex``, ``wrist_flex``,
  ``wrist_roll``) over a 6-row task: 3 linear rows track position exactly (weight 1) and 3
  orientation rows are soft-weighted by ``orientation_task_weight`` (default ``0.5``). This
  replaces the reduced 4-row ``[x, y, z, pitch]`` IK over 4 joints plus a separate
  ``wrist_roll`` action. The manipulability-aware damped least squares and null-space
  joint-limit avoidance are retained (the damping ramp keys off the full weighted task Jacobian,
  so it is coupled to the orientation weight).

  Migration: the controller / action classes were renamed
  ``SO101PositionPitchIK{Controller,ControllerCfg,Action,ActionCfg}`` ->
  ``SO101PoseIK{...}`` (modules ``position_pitch_ik_{controller,action}.py`` ->
  ``pose_ik_{controller,action}.py``), the cfg field ``pitch_task_weight`` ->
  ``orientation_task_weight``, and the ``wrist_roll_action`` term was removed (``wrist_roll`` is
  now solved by the IK). The SO-101 ``SO101WristRetargeter`` (and its ``SO101RollRetargeter``
  alias) was removed from ``isaacteleop.retargeters``; the clutch retargeter now drives the full
  pose with a fixed orientation calibration offset.
