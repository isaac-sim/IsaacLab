Fixed
^^^^^

* Fixed Newton joint wrench sensors to report fixed connections within an articulation, including
  welded wrist sensors and tool flanges. Free joints, world-fixed roots, and loop-closing constraints
  remained excluded. Select entries by ``body_names`` or ``find_bodies`` because the reported body count
  and ordering may change.
