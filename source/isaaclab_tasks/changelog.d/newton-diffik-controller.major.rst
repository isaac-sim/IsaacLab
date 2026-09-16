Fixed
^^^^^

* Fixed the SO-101 pose IK action to construct its specialized controller through the configured class type.

Changed
^^^^^^^

* **Breaking:** SO-101 pose IK required ``cfg.num_joints`` before construction. Action terms populated it
  from resolved joints; standalone callers must set this field.
