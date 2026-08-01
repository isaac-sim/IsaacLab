Fixed
^^^^^

* Fixed URDF and MJCF conversion producing assets with no physics, which left spawned
  articulations without joints, articulation roots, or mass properties. The converters now
  select the ``"Physics"`` variant that the importer leaves unset.
