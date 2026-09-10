Fixed
^^^^^

* Fixed :class:`~isaaclab.controllers.OperationalSpaceController` using ``pose_abs`` target quaternions
  without normalizing them. Unnormalized policy outputs scaled the orientation error and the commanded
  efforts by the quaternion norm. Targets are now normalized, and degenerate (zero or non-finite)
  quaternions fall back to the current end-effector orientation, matching the absolute-pose handling of
  :class:`~isaaclab.controllers.DifferentialIKController`.
