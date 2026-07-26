Added
^^^^^

* Added exhaustive standalone demo and tutorial smoke-test coverage across
  declared physics, renderer, and visualizer combinations, with the headless
  demo matrix enabled in CI.

Changed
^^^^^^^

* **Breaking:** Changed standalone demos that require Isaac Sim PhysX to use
  ``isaacsim_physx`` explicitly. Replace ``--physics physx`` with
  ``--physics isaacsim_physx`` when launching these demos.

Fixed
^^^^^

* Fixed the Newton bin-packing demo dropping contacts and constraints when its
  MJWarp buffers overflowed.
