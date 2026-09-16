Fixed
^^^^^

* Fixed the SO-101 pose IK action to construct its specialized controller through the configured class type.

Changed
^^^^^^^

* **Breaking:** SO-101 pose IK required ``num_joints=`` at construction. Updated task callers;
  standalone callers must supply the selected joint count.
