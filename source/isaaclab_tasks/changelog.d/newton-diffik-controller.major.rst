Fixed
^^^^^

* Fixed the SO-101 pose IK action to construct its specialized controller through the configured class type.

Changed
^^^^^^^

* **Breaking:** Updated SO-101 pose IK and keyboard reset IK to supply fixed controller joint counts and
  initial joint limits. Standalone SO-101 controller callers must pass ``num_joints=`` and,
  when avoidance is enabled, ``joint_pos_limits=`` at construction.
